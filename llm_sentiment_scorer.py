# -*- coding: utf-8 -*-
"""LLM sentiment scorer for IMA abstracts (sequential + retry on 429)."""
import json, os, sys, argparse, datetime, time, re
from pathlib import Path
import requests
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# auto load .env.llm
_env_file = Path(__file__).resolve().parent / ".env.llm"
if _env_file.exists():
    for line in _env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line=line.strip()
        if line and not line.startswith("#") and "=" in line:
            k,v=line.split("=",1); k=k.strip(); v=v.strip()
            if k and k not in os.environ:
                os.environ[k]=v

DATA = Path(r"F:\even-codex\us-stock-data")
META = DATA / "ima_all_meta.json"
OUT  = Path(__file__).resolve().parent / "backtest_output"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "kb_llm_sentiment_cache.jsonl"
RESULT = OUT / "kb_llm_sentiment.csv"

PROMPT = """你是美股量化研究员。下面是一篇美股科技投研日报/研报的 AI 摘要。
请只输出明确提到、且你能判断多空倾向的美股代码信号。
不要猜、不要输出普通单词。如果只有中性提及，direction=0。
输出必须是可以 json.loads 的数组：
[{"ticker": "NVDA", "direction": 1, "strength": 0.7, "reason": "原文提到定制芯片需求加速"}]
direction: 1=看多, -1=看空, 0=中性。strength: 0到1。
若没有明确信号，输出 []。不要输出解释。

日报摘要：
{abstract}"""

def load_items():
    return json.loads(META.read_text(encoding="utf-8"))

def load_done():
    done=set()
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                obj=json.loads(line)
                if isinstance(obj, dict) and obj.get("media_id"):
                    done.add(obj["media_id"])
            except Exception:
                continue
    return done

def parse_date(it):
    ts=str(it.get("create_time") or "0")
    if ts.isdigit():
        ms=int(ts); sec=ms/1000 if ms>100000000000 else ms
        try:
            return datetime.datetime.fromtimestamp(sec).strftime("%Y-%m-%d")
        except Exception:
            return ""
    return ""

def call_llm(abstract, key, base, model, timeout=120):
    url = base.rstrip("/") + "/chat/completions"
    payload = {"model": model, "temperature": 0.0,
               "messages":[{"role":"system","content":"You return only valid JSON arrays."},
                            {"role":"user","content": PROMPT.replace("{abstract}", abstract[:4000])}]}
    last=None
    for attempt in range(6):
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type":"application/json"}, json=payload, timeout=timeout)
            if r.status_code == 429:
                wait = 4 * (attempt+1)
                print(f"    429, backoff {wait}s", flush=True)
                time.sleep(wait); continue
            r.raise_for_status()
            txt=(r.json()["choices"][0]["message"]["content"] or "").strip()
            txt=re.sub(r"^```(?:json)?\s*","",txt); txt=re.sub(r"\s*```$","",txt)
            try:
                arr=json.loads(txt)
            except Exception:
                m=re.search(r"\[.*\]", txt, re.S)
                if not m: return []
                arr=json.loads(m.group(0))
            return arr if isinstance(arr, list) else []
        except Exception as e:
            last=e; wait=3*(attempt+1)
            print(f"    attempt {attempt+1} err {type(e).__name__} {str(e)[:120]}, wait {wait}s", flush=True)
            time.sleep(wait)
    raise last if last else requests.exceptions.RequestException("failed")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dry-run",action="store_true"); ap.add_argument("--limit",type=int,default=0); ap.add_argument("--sleep",type=float,default=2.0)
    args=ap.parse_args()
    if args.dry_run:
        items=load_items(); print("total items",len(items)); print(PROMPT.replace("{abstract}",(items[0].get("abstract") or "")[:1200])); print("dry-run OK"); return
    key=os.environ.get("LLM_API_KEY","").strip()
    if not key:
        print("ERROR: no LLM_API_KEY"); sys.exit(2)
    base=os.environ.get("LLM_BASE_URL","https://open.bigmodel.cn/api/paas/v4").strip()
    model=os.environ.get("LLM_MODEL","glm-4-flash").strip()
    items=load_items(); done=load_done()
    todo=[it for it in items if it.get("media_id") not in done]
    src=os.environ.get("LLM_SOURCE","").strip()
    if src:
        todo=[it for it in todo if it.get("source_folder","")==src]

    if args.limit: todo=todo[:args.limit]
    print("total",len(items),"done",len(done),"todo",len(todo),"model",model,"base",base)
    rows=[]
    # merge previous cache so partial runs don't lose results
    seen_rows = set()
    if CACHE.exists():
        for _line in CACHE.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                _obj=json.loads(_line)
            except Exception:
                continue
            for _s in (_obj.get("sig") or []):
                if isinstance(_s, dict) and _s.get("ticker"):
                    _key=(_s.get("media_id"),_s.get("ticker"))
                    if _key not in seen_rows:
                        seen_rows.add(_key); rows.append(_s)

    for i,it in enumerate(todo,1):
        mid=it.get("media_id","")
        try:
            arr=call_llm(it.get("abstract",""), key, base, model)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] FAILED {mid[:30]} {str(e)[:100]}", flush=True)
            continue
        d=parse_date(it)
        new_rows=[]
        for sig in arr:
            tk=str(sig.get("ticker","")).strip().upper()
            if not tk or len(tk)<2: continue
            new_rows.append({"media_id":mid,"title":it.get("title",""),"pdf_date":d,"source_folder":it.get("source_folder",""),"ticker":tk,"direction":int(sig.get("direction",0)),"strength":float(sig.get("strength",0.5)),"reason":str(sig.get("reason",""))[:200],"model":model})
        rows.extend(new_rows)
        if new_rows:
            with CACHE.open("a",encoding="utf-8") as f:
                f.write(json.dumps({"media_id":mid,"sig":new_rows},ensure_ascii=False)+"\n")
        print(f"  [{i}/{len(todo)}] {mid[:30]} -> {len(new_rows)} signals", flush=True)
        time.sleep(args.sleep)
    import csv
    with RESULT.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["media_id","title","pdf_date","source_folder","ticker","direction","strength","reason","model"]); w.writeheader(); w.writerows(rows)
    print("done rows",len(rows),"saved",RESULT)

if __name__=="__main__":
    main()
