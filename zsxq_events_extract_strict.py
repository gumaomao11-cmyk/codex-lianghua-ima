# -*- coding: utf-8 -*-
"""严格版星球事件抽取：只抽评级变动/目标价调整，不抽笼统positive/negative。
输出 backtest_output/zsxq_events_浑水调研Plus_strict_cache.jsonl
"""
import json, os, re, sys, time, argparse, datetime, csv
from pathlib import Path
import requests
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROJ = Path(r"F:\even-codex\lianghua+IMA")
OUT = _PROJ / "backtest_output"

_envf = Path(__file__).resolve().parent / ".env.llm"
if _envf.exists():
    for line in _envf.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

CN2T = {
    "英伟达":"NVDA","超威半导体":"AMD","AMD":"AMD","美光":"MU","西部数据":"WDC","希捷":"STX",
    "迈威尔":"MRVL","应用材料":"AMAT","康宁":"GLW","伟创力":"FLEX","戴尔":"DELL",
    "Arm":"ARM","新易盛":"EOPT","博通":"AVGO","台积电":"TSM","高通":"QCOM",
    "英特尔":"INTC","超微":"SMCI","Meta":"META","苹果":"AAPL","微软":"MSFT",
    "亚马逊":"AMZN","特斯拉":"TSLA","谷歌":"GOOGL","礼来":"LLY","Moderna":"MRNA",
    "默沙东":"MRK","雅培":"ABT","奈飞":"NFLX","优步":"UBER","Roblox":"RBLX",
    "Palantir":"PLTR","Coinbase":"COIN","纳斯达克":"QQQ",
}

PROMPT = """你是美股量化研究员。下面是一段来自投资星球的内容。只抽取【明确的评级变动或目标价调整】，不抽笼统情绪判断。

只抽取以下5类信号（没有就输出空数组[]）：
1. upgrade: 分析师明确上调评级（如Hold→Buy，Underweight→Overweight，Sell→Hold）
2. downgrade: 分析师明确下调评级
3. reinitiate: 首次覆盖或恢复覆盖（"启动覆盖""恢复评级""initiate"）
4. price_up: 目标价上调（必须给出具体新目标价数字）
5. price_down: 目标价下调（必须给出具体新目标价数字）

严禁抽取：笼统的"看好/看空/正面/负面"(positive/negative)、中性评论、财报数据、行业趋势。

输出JSON数组，每个元素：
  ticker: 标准美股代码
  action: upgrade|downgrade|reinitiate|price_up|price_down
  direction: 1(上调) -1(下调)
  strength: upgrade/downgrade=0.9, reinitiate=0.8, price_up/price_down=0.7
  pt_delta_pct: 目标价变动百分比(如100→120则20.0)，无目标价则0
  evidence: 原文证据(≤60字，必须包含具体评级词或目标价数字)

示例：[{"ticker":"NVDA","action":"upgrade","direction":1,"strength":0.9,"pt_delta_pct":0,"evidence":"大摩上调NVDA至Overweight"}]
没有明确评级/目标价变动就输出[]。"""

def load(items_path):
    return json.loads(Path(items_path).read_text(encoding="utf-8"))

def parse_iso(ts):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", ts or "")
    return m.group(0) if m else ""

def prefilter(text):
    rating_kw = ["上调","下调","升级","降级","买入","卖出","增持","减持","Overweight","Underweight",
                 "Buy","Sell","Hold","Outperform","Underperform","Neutral","目标价","维持","覆盖",
                 "initiate","upgrade","downgrade","reinitiate","raising","cutting","target"]
    us_codes = ["NVDA","AMD","INTC","MU","WDC","STX","MRVL","AMAT","GLW","FLEX","DELL","NBIS","ARM",
                "AVGO","QCOM","SMCI","META","AAPL","MSFT","GOOGL","AMZN","TSLA","LLY","MRK","COIN","PLTR","TSM"]
    return (any(x in text for x in rating_kw) and
            (any(x in text for x in us_codes) or any(k in text for k in CN2T if len(k)>=2)))

def call_llm(text, key, base, model, timeout=120):
    url = base.rstrip("/") + "/chat/completions"
    payload = {"model": model, "temperature": 0.0,
               "messages":[{"role":"system","content":"You return only valid JSON arrays. No explanation."},
                            {"role":"user","content": PROMPT + "\n\n内容：" + text[:4500]}]}
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
            print(f"  LLM error attempt {attempt+1}: {str(e)[:80]}", flush=True)
            time.sleep(3*(attempt+1))
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.7)
    ap.add_argument("--src", default="浑水调研Plus")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    items = load(OUT / f"zsxq_{a.src}.json")
    key = os.environ.get("LLM_API_KEY","").strip()
    base = os.environ.get("LLM_BASE_URL","").strip()
    model = os.environ.get("LLM_MODEL","").strip()
    if not key: print("no LLM_API_KEY"); sys.exit(2)
    print(f"LLM: {model} @ {base}", flush=True)

    candidates = []
    for it in items:
        text = ((it.get("title") or "") + " " + (it.get("content") or ""))
        if not text.strip(): continue
        if prefilter(text):
            d = parse_iso(it.get("create_time") or "")
            candidates.append((d, it, text))
    candidates.sort(key=lambda x: x[0], reverse=True)
    if a.limit: candidates = candidates[:a.limit]
    print(f"prefiltered candidates: {len(candidates)} (from {len(items)} total)")

    if a.dry:
        for d, it, txt in candidates[:3]:
            print(f"--- {d} {(it.get('title') or '')[:60]}")
        return

    cache = OUT / f"zsxq_events_{a.src}_strict_cache.jsonl"
    done = set()
    if cache.exists():
        for ln in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
            try: done.add(json.loads(ln)["media_id"])
            except: pass
    print(f"already done: {len(done)}, remaining: {len(candidates)-len(done)}", flush=True)

    rows = []
    for i,(d, it, txt) in enumerate(candidates, 1):
        mid = it.get("topic_id","")
        if mid in done: continue
        arr = call_llm(txt, key, base, model)
        newrows = []
        for e in arr:
            tk = str(e.get("ticker","")).strip().upper()
            if not tk or not re.fullmatch(r"[A-Z]{1,6}", tk): continue
            act = str(e.get("action","")).strip()
            if act not in ("upgrade","downgrade","reinitiate","price_up","price_down"): continue
            newrows.append({"media_id": mid, "title": it.get("title","")[:120], "pdf_date": d,
                            "source_folder": a.src,
                            "ticker": tk, "action": act,
                            "direction": int(e.get("direction",0)),
                            "strength": float(e.get("strength",0.7)),
                            "pt_delta_pct": float(e.get("pt_delta_pct",0) or 0),
                            "evidence": str(e.get("evidence",""))[:80]})
        rows.extend(newrows)
        with cache.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"media_id": mid, "sig": newrows}, ensure_ascii=False)+"\n")
        done.add(mid)
        if i % 10 == 0 or i <= 3:
            print(f"[{i}/{len(candidates)}] {d} {mid} -> {len(newrows)} (total rows: {len(rows)})", flush=True)
        time.sleep(a.sleep)

    csvpath = OUT / f"zsxq_events_{a.src}_strict.csv"
    with csvpath.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["media_id","title","pdf_date","source_folder","ticker","action","direction","strength","pt_delta_pct","evidence"])
        w.writeheader(); w.writerows(rows)
    print(f"done! rows={len(rows)} saved {csvpath}", flush=True)

if __name__ == "__main__":
    main()
