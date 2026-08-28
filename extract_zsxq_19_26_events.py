# -*- coding: utf-8 -*-
"""从 19-26年留存 星球提取美股事件因子（LLM 结构化）。
输入: backtest_output/zsxq_group_48418411254128_web.json
输出: backtest_output/zsxq_19_26_events.csv + cache jsonl
"""
import json, os, re, sys, time, argparse, csv
from pathlib import Path
import requests
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

PROMPT = """你是美股量化研究员。下面是一段来自投资星球的内容，可能是外资研报、美股科技日报或短评。
请只抽取【明确提到且在美股上市】的公司信号，不猜、不输出普通单词。
对每个股票输出一个事件，字段说明：
  ticker: 标准美股代码（如 NVDA）。若只给了中文名，用常见映射；无法确认美股代码就跳过。
  action: 取值之一 [upgrade, downgrade, reinitiate, price_up, price_down, positive, negative, neutral]
  direction: 1(看多/上调) -1(看空/下调) 0(中性)
  strength: 0到1 的确定性/强度
  pt_delta_pct: 若明确给了目标价变化幅度写百分比，否则 0
  evidence: 一句原文证据（≤60字）
只输出合法 JSON 数组，例如：
[{"ticker":"NVDA","action":"price_up","direction":1,"strength":0.8,"pt_delta_pct":5.0,"evidence":"目标价上调至280美元"}]
没有明确信号输出 []。不要输出解释。

内容：{text}"""

def load_web_json():
    path = OUT / "zsxq_group_48418411254128_web.json"
    return json.loads(path.read_text(encoding="utf-8"))

def norm_date(ts):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", ts or "")
    return m.group(1) if m else ""

def prefilter(text):
    """预过滤：只保留可能含美股信号的文本。"""
    if not text or len(text) < 20:
        return False
    # 显式美股代码或 .US
    if re.search(r"\b[A-Z]{1,6}\.US\b", text):
        return True
    # 常见美股代码（全大写，长度2-5）
    if re.search(r"\b(NVDA|AMD|INTC|MU|WDC|STX|MRVL|AMAT|GLW|FLEX|DELL|NBIS|ARM|LITE|AVGO|SNPS|ADI|KLAC|LRCX|MCHP|ON|NXPI|TXN|QCOM|SMCI|META|AAPL|MSFT|GOOGL|AMZN|TSLA|LLY|MRNA|MRK|COIN|PLTR|TSM|UBER|NFLX|RBLX)\b", text):
        return True
    # 中文名
    if any(k in text for k in CN2T if len(k) >= 2):
        return True
    # 外资投行 + 科技关键词
    inst = ["摩根大通","大摩","高盛","瑞银","野村","美银","巴克莱","摩根士丹利","花旗","杰富瑞","Raymond James","德意志","汇丰","法兴","瑞信","伯恩斯坦"]
    tech_kw = ["AI", "GPU", "芯片", "半导体", "算力", "光模块", "数据中心", "云计算", "大模型", "美股"]
    if any(x in text for x in inst) and any(x in text for x in tech_kw):
        return True
    return False

def call_llm(text, key, base, model, timeout=120):
    url = base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model, "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "You return only valid JSON arrays."},
            {"role": "user", "content": PROMPT.replace("{text}", text[:6000])}
        ]
    }
    for attempt in range(5):
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=timeout)
            if r.status_code == 429:
                time.sleep(4 * (attempt + 1)); continue
            r.raise_for_status()
            txt = (r.json()["choices"][0]["message"]["content"] or "").strip()
            txt = re.sub(r"^```(?:json)?\s*", "", txt); txt = re.sub(r"\s*```$", "", txt)
            try:
                arr = json.loads(txt)
            except Exception:
                m = re.search(r"\[.*\]", txt, re.S)
                arr = json.loads(m.group(0)) if m else []
            return arr if isinstance(arr, list) else []
        except Exception as e:
            print(f"  LLM error attempt {attempt+1}: {e}")
            time.sleep(3 * (attempt + 1))
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.8)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    items = load_web_json()
    print(f"[load] {len(items)} topics")

    key = os.environ.get("LLM_API_KEY", "").strip()
    base = os.environ.get("LLM_BASE_URL", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip()
    if not key: print("[error] no LLM_API_KEY"); sys.exit(2)
    print(f"[llm] {model} @ {base}")

    cache_path = OUT / "zsxq_19_26_events_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
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
    fcsv = open(OUT / "zsxq_19_26_events.csv", "w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(fcsv, fieldnames=["date", "ticker", "action", "direction", "strength", "pt_delta_pct", "evidence", "source_text", "author", "create_time"])
    writer.writeheader()

    fcache = open(cache_path, "a", encoding="utf-8")

    processed = 0
    for idx, it in filtered:
        text = it.get("text", "")
        date = norm_date(it.get("create_time", ""))
        sig = f"{date}|{hash(text[:100])}"

        if sig in cache:
            events = cache[sig].get("events", [])
        else:
            events = call_llm(text, key, base, model)
            cache_obj = {"sig": sig, "create_time": it.get("create_time"), "events": events, "model": model}
            fcache.write(json.dumps(cache_obj, ensure_ascii=False) + "\n")
            fcache.flush()

        for ev in events:
            ticker = ev.get("ticker", "")
            if not ticker or ticker not in US_TICKERS:
                continue
            row = {
                "date": date,
                "ticker": ticker,
                "action": ev.get("action", ""),
                "direction": ev.get("direction", 0),
                "strength": ev.get("strength", 0),
                "pt_delta_pct": ev.get("pt_delta_pct", 0),
                "evidence": ev.get("evidence", ""),
                "source_text": text[:500],
                "author": it.get("author", ""),
                "create_time": it.get("create_time", ""),
            }
            writer.writerow(row)
            rows.append(row)

        processed += 1
        if processed % 50 == 0:
            print(f"[progress] {processed}/{len(filtered)} -> {len(rows)} events")
        time.sleep(a.sleep)

    fcsv.close()
    fcache.close()
    print(f"[done] processed {processed}, events {len(rows)}, saved {OUT / 'zsxq_19_26_events.csv'}")

if __name__ == "__main__":
    main()
