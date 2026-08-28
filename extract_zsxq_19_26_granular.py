# -*- coding: utf-8 -*-
"""从 19-26年留存 星球提取细粒度美股事件因子（带 text_type / materiality_tier / horizon）。
输入: backtest_output/zsxq_group_48418411254128_web.json
输出: data/duckdb/zsxq_19_26_granular_events.parquet + cache jsonl
"""
import json, os, re, sys, time, argparse, csv
from pathlib import Path
from datetime import datetime
import requests
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROJ = Path(r"F:\even-codex\lianghua+IMA")
OUT = _PROJ / "backtest_output"

# 加载 .env.llm
_envf = _PROJ / ".env.llm"
if _envf.exists():
    for line in _envf.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

# 美股中文/常见名映射
CN2T = {
    "英伟达":"NVDA","超威半导体":"AMD","AMD":"AMD","美光":"MU","西部数据":"WDC","希捷":"STX",
    "迈威尔":"MRVL","应用材料":"AMAT","康宁":"GLW","伟创力":"FLEX","戴尔":"DELL",
    "Arm":"ARM","新易盛":"EOPT","博通":"AVGO","台积电":"TSM","高通":"QCOM",
    "英特尔":"INTC","超微":"SMCI","Meta":"META","苹果":"AAPL","微软":"MSFT",
    "亚马逊":"AMZN","特斯拉":"TSLA","谷歌":"GOOGL","礼来":"LLY","Moderna":"MRNA",
    "默沙东":"MRK","雅培":"ABT","奈飞":"NFLX","优步":"UBER","Roblox":"RBLX",
    "Palantir":"PLTR","Coinbase":"COIN","英伟达大全":"NVDA","纳斯达克":"QQQ",
    "应用光电":"AAOI","安森美":"ON","微芯":"MCHP","德州仪器":"TXN","亚德诺":"ADI",
    "新思科技":"SNPS","铿腾":"CDNS","拉姆研究":"LRCX","科磊":"KLAC","迈威尔科技":"MRVL",
    "美满":"MRVL","芯源":"MPWR","泛林":"LRCX",
}

US_TICKERS = set(CN2T.values())

# 记录已耗尽（402）的模型，避免重复尝试
EXHAUSTED_MODELS = set()

MODEL_FALLBACKS = ['deepseek-v4-flash-202605', 'deepseek-v4-flash-0731', 'glm-5-turbo', 'glm-5', 'glm-5.1', 'glm-5v-turbo', 'mimo-v2.5-pro', 'minimax-m3', 'minimax-m2.7', 'hy-mt2-pro', 'hy-mt2-plus', 'kimi-k2.6', 'deepseek/deepseek-v4-flash-vision-exp', 'deepseek-v4-pro', 'deepseek-v4-pro-202606']

SCHEMA_DESCRIPTION = """{\"type\": \"array\", \"items\": {\"type\": \"object\", \"properties\": {\"is_us_stock\": {\"type\": \"boolean\"}, \"ticker\": {\"type\": \"string\"}, \"text_type\": {\"type\": \"string\", \"enum\": [\"research_report\", \"news_summary\", \"single_event\", \"personal_opinion\", \"noise\"]}, \"materiality_tier\": {\"type\": \"string\", \"enum\": [\"tier_1_hard_data\", \"tier_2_soft_logic\", \"tier_3_macro_industry\"]}, \"sentiment_score\": {\"type\": \"number\", \"minimum\": -1.0, \"maximum\": 1.0}, \"expected_horizon_days\": {\"type\": \"integer\"}, \"confidence\": {\"type\": \"number\", \"minimum\": 0.0, \"maximum\": 1.0}, \"evidence\": {\"type\": \"string\"}}, \"required\": [\"is_us_stock\", \"ticker\", \"text_type\", \"materiality_tier\", \"sentiment_score\", \"expected_horizon_days\", \"confidence\", \"evidence\"]}}"""

PROMPT = f"""你是美股量化研究员。请分析下面这段投资星球内容，按美股标的抽取结构化信号。
输出要求：
- 只输出合法 JSON 数组，格式严格如下 schema，不要任何解释。
- 每个数组元素代表一个美股标的信号。
- 如果内容不直接关联美股标的，输出 []
- text_type: research_report(研报/目标价/EPS), news_summary(科技日报/新闻汇总), single_event(单一催化剂), personal_opinion(个人观点), noise(闲聊)
- materiality_tier: tier_1_hard_data(有明确数字), tier_2_soft_logic(逻辑推演), tier_3_macro_industry(行业宏观)
- expected_horizon_days: research_report=20, single_event=3, news_summary/personal_opinion=1, noise=0
JSON Schema: {SCHEMA_DESCRIPTION}
内容：{{text}}"""

def load_web_json():
    path = OUT / "zsxq_group_48418411254128_web.json"
    return json.loads(path.read_text(encoding="utf-8"))

def norm_date(ts):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", ts or "")
    return m.group(1) if m else ""

def parse_time(ts):
    """转换为 UTC 时间戳字符串。"""
    if not ts:
        return None
    try:
        dt = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M")
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None

def prefilter(text):
    if not text or len(text) < 20:
        return False
    if re.search(r"\b[A-Z]{1,6}\.US\b", text):
        return True
    if re.search(r"\b(NVDA|AMD|INTC|MU|WDC|STX|MRVL|AMAT|GLW|FLEX|DELL|NBIS|ARM|LITE|AVGO|SNPS|ADI|KLAC|LRCX|MCHP|ON|NXPI|TXN|QCOM|SMCI|META|AAPL|MSFT|GOOGL|AMZN|TSLA|LLY|MRNA|MRK|COIN|PLTR|TSM|UBER|NFLX|RBLX)\b", text):
        return True
    if any(k in text for k in CN2T if len(k) >= 2):
        return True
    inst = ["摩根大通","大摩","高盛","瑞银","野村","美银","巴克莱","摩根士丹利","花旗","杰富瑞","Raymond James","德意志","汇丰","法兴","瑞信","伯恩斯坦"]
    tech_kw = ["AI", "GPU", "芯片", "半导体", "算力", "光模块", "数据中心", "云计算", "大模型", "美股"]
    if any(x in text for x in inst) and any(x in text for x in tech_kw):
        return True
    return False

def call_llm(text, key, base, model, timeout=120):
    url = base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "You return only valid JSON arrays."},
            {"role": "user", "content": PROMPT.replace("{text}", text[:6000])}
        ]
    }
    for attempt in range(2):
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=timeout)
            if r.status_code == 402:
                print(f"  [{model}] 402 Payment Required, switch")
                EXHAUSTED_MODELS.add(model)
                return None
            if r.status_code == 429:
                time.sleep(4 * (attempt + 1))
                continue
            r.raise_for_status()
            txt = (r.json()["choices"][0]["message"]["content"] or "").strip()
            txt = re.sub(r"^```(?:json)?\s*", "", txt)
            txt = re.sub(r"\s*```$", "", txt)
            try:
                arr = json.loads(txt)
            except Exception:
                m = re.search(r"\[.*\]", txt, re.S)
                arr = json.loads(m.group(0)) if m else []
            return arr if isinstance(arr, list) else []
        except Exception as e:
            print(f"  LLM error [{model}] attempt {attempt+1}: {e}")
            time.sleep(3 * (attempt + 1))
    return None  # None 表示失败，需要切换模型

def call_llm_with_fallback(text, key, base, primary_model):
    models = [primary_model] + [m for m in MODEL_FALLBACKS if m != primary_model]
    for model in models:
        if model in EXHAUSTED_MODELS:
            continue
        result = call_llm(text, key, base, model)
        if result is not None:
            return result, model
    return [], primary_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "glm-5-turbo"))
    a = ap.parse_args()

    items = load_web_json()
    print(f"[load] {len(items)} topics")

    key = os.environ.get("LLM_API_KEY", "").strip()
    base = os.environ.get("LLM_BASE_URL", "").strip()
    if not key:
        print("[error] no LLM_API_KEY")
        sys.exit(2)
    print(f"[llm] primary={a.model} @ {base}")

    cache_path = OUT / "zsxq_19_26_granular_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                cache[obj.get("sig", "")] = obj
            except Exception:
                pass
    print(f"[cache] {len(cache)} cached")

    filtered = [(i, it) for i, it in enumerate(items) if prefilter(it.get("text", ""))]
    if a.limit:
        filtered = filtered[:a.limit]
    print(f"[filter] {len(filtered)} topics after prefilter")

    if a.dry:
        for i, it in filtered[:5]:
            print(f"--- {it.get('create_time')} ---")
            print(it.get("text", "")[:300])
        return

    rows = []
    fcache = open(cache_path, "a", encoding="utf-8")

    processed = 0
    for idx, it in filtered:
        text = it.get("text", "")
        date = norm_date(it.get("create_time", ""))
        sig = f"{date}|{text[:80]}"

        if sig in cache:
            events = cache[sig].get("events", [])
            used_model = cache[sig].get("model", a.model)
        else:
            events, used_model = call_llm_with_fallback(text, key, base, a.model)
            cache_obj = {"sig": sig, "create_time": it.get("create_time"), "events": events, "model": used_model}
            fcache.write(json.dumps(cache_obj, ensure_ascii=False) + "\n")
            fcache.flush()

        for ev in events:
            if not ev.get("is_us_stock"):
                continue
            ticker = ev.get("ticker", "").upper().strip()
            if not ticker or ticker not in US_TICKERS:
                continue
            event_time = parse_time(it.get("create_time"))
            if not event_time:
                continue
            sentiment = float(ev.get("sentiment_score", 0))
            confidence = float(ev.get("confidence", 0))
            row = {
                "event_time": event_time,
                "date": date,
                "ticker": ticker,
                "text_type": ev.get("text_type", "noise"),
                "materiality_tier": ev.get("materiality_tier", "tier_3_macro_industry"),
                "sentiment_score": sentiment,
                "confidence": confidence,
                "raw_signal": sentiment * confidence,
                "horizon_days": int(ev.get("expected_horizon_days", 1)),
                "evidence": ev.get("evidence", ""),
                "source_text": text[:500],
                "author": it.get("author", ""),
                "create_time": it.get("create_time", ""),
            }
            rows.append(row)

        processed += 1
        if processed % 50 == 0:
            print(f"[progress] {processed}/{len(filtered)} -> {len(rows)} events")
        time.sleep(a.sleep)

    fcache.close()

    if rows:
        df = pd.DataFrame(rows)
        out_parquet = _PROJ / "data" / "duckdb" / "zsxq_19_26_granular_events.parquet"
        df.to_parquet(out_parquet, index=False)
        print(f"[saved] {out_parquet}: {len(df)} events")
    else:
        print("[warn] no events extracted")

    print(f"[done] processed {processed}, events {len(rows)}")

if __name__ == "__main__":
    main()
