# -*- coding: utf-8 -*-
"""从知识星球(浑水调研Plus等)抽取外资研报/US事件因子（LLM 结构化）。
输出 backtest_output/zsxq_events_{...}.csv + 缓存 jsonl。
"""
import json, os, re, sys, time, argparse, datetime, csv
from pathlib import Path
import requests
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROJ = Path(r"F:\even-codex\lianghua+IMA")
OUT = _PROJ / "backtest_output"

# .env.llm
_envf = Path(__file__).resolve().parent / ".env.llm"
if _envf.exists():
    for line in _envf.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

# 美股中文/常见名映射（重点覆盖策略 universe）
CN2T = {
    "英伟达":"NVDA","超威半导体":"AMD","AMD":"AMD","美光":"MU","西部数据":"WDC","希捷":"STX",
    "迈威尔":"MRVL","应用材料":"AMAT","康宁":"GLW","伟创力":"FLEX","戴尔":"DELL",
    "Arm":"ARM","新易盛":"EOPT","博通":"AVGO","台积电":"TSM","高通":"QCOM",
    "英特尔":"INTC","超微":"SMCI","Meta":"META","苹果":"AAPL","微软":"MSFT",
    "亚马逊":"AMZN","特斯拉":"TSLA","谷歌":"GOOGL","礼来":"LLY","Moderna":"MRNA",
    "默沙东":"MRK","雅培":"ABT","奈飞":"NFLX","优步":"UBER","Roblox":"RBLX",
    "Palantir":"PLTR","Coinbase":"COIN","英伟达大全":"NVDA","纳斯达克":"QQQ",
}
TICKER_CNM2 = {v:k for k,v in CN2T.items()}

PROMPT = """你是美股量化研究员。下面是一段来自投资星球的内容（可能是外资研报、美股科技日报或短评）。
请只抽取【明确提到且在美股上市】的公司信号，不猜、不输出普通单词。
对每个股票输出一个事件，字段说明：
  ticker: 标准美股代码（如 NVDA）。若只给了中文名，用我提供的映射；无法确认美股代码就跳过。
  action: 取值之一 [upgrade, downgrade, reinitiate, price_up, price_down, positive, negative, neutral]
  direction: 1(看多/上调) -1(看空/下调) 0(中性)
  strength: 0到1 的确定性/强度
  pt_delta_pct: 若明确给了目标价变化幅度(相对现价或原目标价)写百分比，否则 0
  evidence: 一句原文证据（≤60字）
只输出合法 JSON 数组，例如：
[{"ticker":"NVDA","action":"price_up","direction":1,"strength":0.8,"pt_delta_pct":5.0,"evidence":"目标价280美元"}]
没有明确信号输出 []。不要输出解释。

内容：{text}"""

def load(items_path):
    return json.loads(Path(items_path).read_text(encoding="utf-8"))

def norm_date(ts):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", ts or "")
    return m.group(1) if m else ""

def parse_iso(ts):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", ts or "")
    return m.group(0) if m else (norm_date(ts) or "")

def prefilter(text):
    inst = ["摩根大通","大摩","高盛","瑞银","野村","美银","巴克莱","摩根士丹利","花旗","杰富瑞","Raymond James","德意志","汇丰","法兴","瑞信","伯恩斯坦"]
    us_codes = ["NVDA","AMD","INTC","MU","WDC","STX","MRVL","AMAT","GLW","FLEX","DELL","NBIS","ARM","LITE","AVGO","SNPS","ADI","KLAC","LRCX","MCHP","ON","NXPI","TXN","QCOM","SMCI","META","AAPL","MSFT","GOOGL","AMZN","TSLA","LLY","MRNA","MRK","COIN","PLTR","TSM"]
    return (re.search(r"[A-Z]{1,6}\.US", text) is not None
            or any(x in text for x in inst)
            or any(x in text for x in us_codes or [])
            or any(k in text for k in CN2T if len(k) >= 2))

def call_llm(text, key, base, model, timeout=120):
    url = base.rstrip("/") + "/chat/completions"
    payload = {"model": model, "temperature": 0.0,
               "messages":[{"role":"system","content":"You return only valid JSON arrays."},
                            {"role":"user","content": PROMPT.replace("{text}", text[:4500])}]}
    for attempt in range(5):
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type":"application/json"}, json=payload, timeout=timeout)
            if r.status_code == 429:
                time.sleep(4*(attempt+1)); continue
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
            time.sleep(3*(attempt+1))
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=1.2)
    ap.add_argument("--src", default="浑水调研Plus")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    items = load(OUT / f"zsxq_{a.src}.json")
    key = os.environ.get("LLM_API_KEY","").strip(); base = os.environ.get("LLM_BASE_URL","").strip(); model = os.environ.get("LLM_MODEL","").strip()
    if not key: print("no LLM_API_KEY"); sys.exit(2)

    # 只保留 US/外资相关，按时间倒序
    candidates = []
    for it in items:
        text = ((it.get("title") or "") + " " + (it.get("content") or ""))
        if not text.strip(): continue
        if prefilter(text):
            d = parse_iso(it.get("create_time") or "")
            candidates.append((d, it, text))
    candidates.sort(key=lambda x: x[0], reverse=True)
    if a.limit: candidates = candidates[:a.limit]
    if a.dry:
        print("candidates", len(candidates))
        for d, it, txt in candidates[:3]:
            print("---", d, (it.get("title") or "")[:60]); print(PROMPT.replace("{text}", txt[:600]))
        return

    cache = OUT / f"zsxq_events_{a.src}_cache.jsonl"
    done = set()
    if cache.exists():
        for ln in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
            try: done.add(json.loads(ln)["media_id"])
            except Exception: pass

    rows = []
    for i,(d, it, txt) in enumerate(candidates, 1):
        mid = it.get("topic_id","")
        if mid in done: continue
        arr = call_llm(txt, key, base, model)
        newrows = []
        for e in arr:
            tk = str(e.get("ticker","")).strip().upper()
            if not tk or not re.fullmatch(r"[A-Z]{1,6}", tk): continue
            newrows.append({"media_id": mid, "title": it.get("title","")[:120], "pdf_date": d,
                            "source_folder": it.get("group",{}).get("name","") if isinstance(it.get("group"),dict) else a.src,
                            "ticker": tk, "action": e.get("action",""), "direction": int(e.get("direction",0)),
                            "strength": float(e.get("strength",0.5)), "pt_delta_pct": float(e.get("pt_delta_pct",0) or 0),
                            "evidence": str(e.get("evidence",""))[:80]})
        rows.extend(newrows)
        with cache.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"media_id": mid, "sig": newrows}, ensure_ascii=False)+"\n")
        done.add(mid)
        print(f"[{i}/{len(candidates)}] {d} {mid} -> {len(newrows)}", flush=True)
        time.sleep(a.sleep)

    csvpath = OUT / f"zsxq_events_{a.src}.csv"
    with csvpath.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["media_id","title","pdf_date","source_folder","ticker","action","direction","strength","pt_delta_pct","evidence"]); w.writeheader(); w.writerows(rows)
    print("rows", len(rows), "saved", csvpath)

if __name__ == "__main__":
    main()
