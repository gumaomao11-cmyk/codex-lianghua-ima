# -*- coding: utf-8 -*-
"""把 zsxq_events_*.csv 转成标准化因子 CSV（对齐 kb/llm schema）。
输出 backtest_output/zsxq_events_factors.csv
"""
import sys, csv, glob
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path("backtest_output")

rows=[]
for p in sorted(glob.glob(str(OUT/"zsxq_events_*.csv"))):
    if "factors" in p or "cache" in p: continue
    with open(p, encoding="utf-8-sig", newline="") as f:
        r=csv.DictReader(f)
        for it in r:
            d=int(float(it.get("direction") or 0)); st=float(it.get("strength") or 0)
            n_pos=st if d>0 else 0.0; n_neg=st if d<0 else 0.0
            rows.append({"source_folder":it.get("source_folder","星球"),
                         "title":it.get("title","")[:120],
                         "pdf_date":it.get("pdf_date",""),
                         "media_id":it.get("media_id",""),
                         "ticker":(it.get("ticker") or "").strip().upper(),
                         "n_pos":n_pos,"n_neg":n_neg,
                         "sign":1 if d>0 else (-1 if d<0 else 0)})
rows=[r for r in rows if r["ticker"] and r["pdf_date"]]
outp=OUT/"zsxq_events_factors.csv"
with outp.open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["source_folder","title","pdf_date","media_id","ticker","n_pos","n_neg","sign"]); w.writeheader(); w.writerows(rows)
import collections
print("event rows",len(rows),"saved",outp)
top=collections.Counter(r["ticker"] for r in rows).most_common(20)
print("top",top)
from datetime import date
ds=sorted(d for d in set(r["pdf_date"] for r in rows))
print("dates", (ds[0],ds[-1]) if ds else None, "n_days", len(ds))
