# -*- coding: utf-8 -*-
"""Event-study: forward returns after strong LLM sentiment signals (directional)."""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT=Path(r"F:\even-codex\lianghua+IMA\backtest_output")
DATA=Path(r"F:\even-codex\us-stock-data")
px=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce")
close=px.copy(); idx=px.index
df=pd.read_csv(OUT/"kb_llm_sentiment.csv",encoding="utf-8-sig")
df["pdf_date"]=pd.to_datetime(df["pdf_date"],errors="coerce")
df=df.dropna(subset=["pdf_date"])
strong=df[(df["direction"]!=0)&(df["strength"]>=0.5)].copy()
print("strong signals:", len(strong), "bull", int((strong.direction>0).sum()), "bear", int((strong.direction<0).sum()))

def fwd_for(ticker, pos, h):
    entry=pos+1
    end=entry+h
    if end>=len(idx): return np.nan
    return close.iloc[end][ticker]/close.iloc[entry][ticker]-1.0

rows=[]
for _,r in strong.iterrows():
    t=r["ticker"]
    if t not in close.columns: continue
    pos=idx.searchsorted(r["pdf_date"])
    if pos>=len(idx): continue
    d=1 if r["direction"]>0 else -1
    for h in [1,3,5,10,20]:
        ret=fwd_for(t,pos,h)
        if math.isnan(ret): continue
        rows.append({"signal_date":r["pdf_date"],"ticker":t,"direction":d,"h":h,"adj_ret":d*ret,"raw_ret":ret})
ev=pd.DataFrame(rows)
print("events", len(ev))

def market_bench(sig,h):
    pos=idx.searchsorted(sig); entry=pos+1
    if entry+h>=len(idx): return np.nan
    return (close.iloc[entry+h]/close.iloc[entry]-1.0).mean()
ev["bench"]=[market_bench(sig,h) for sig,h in zip(ev["signal_date"],ev["h"])]
reg=[entry == d for entry,d in zip(ev["adj_ret"],ev["bench"])]
print("broadcast size", len(ev))
stats=[]
for h in [1,3,5,10,20]:
    sub=ev[ev.h==h].dropna(subset=["adj_ret","bench"])
    if len(sub)==0: continue
    adj=(sub["adj_ret"]-sub["bench"]).values
    std=adj.std(ddof=0); t=adj.mean()/std*np.sqrt(len(adj)) if std>0 else np.nan
    stats.append({"horizon":h,"n":len(sub),"mean_adj_ret_pct":adj.mean()*100,"mean_bench_pct":sub["bench"].mean()*100,"mean_raw_pct":sub["raw_ret"].mean()*100,"hit":(adj>0).mean(),"t_stat":t})
    print(f"h={h}: n={len(sub)} mean_adj={adj.mean()*100:.2f}% bench={sub['bench'].mean()*100:.2f}% hit={(adj>0).mean():.2f} t={t:.2f}")
res=pd.DataFrame(stats); res.to_csv(OUT/"llm_event_study.csv",index=False,encoding="utf-8-sig")
print("\n--- by direction (h=5,20) ---")
for h in [5,20]:
    sub=ev[ev.h==h].dropna(subset=["adj_ret","bench"])
    for d,label in [(1,"bull"),(-1,"bear")]:
        x=sub[sub.direction==d]
        if len(x)==0: continue
        active=(x["adj_ret"]-x["bench"]).values
        print(f"h={h} {label}: n={len(x)} mean_active={active.mean()*100:.2f}% hit={(active>0).mean():.2f}")
print("saved llm_event_study.csv")
