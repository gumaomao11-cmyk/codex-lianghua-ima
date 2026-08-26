# -*- coding: utf-8 -*-
"""Next-day strong-bull LLM signal strategy with volatility filter."""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT=Path(r"F:\even-codex\lianghua+IMA\backtest_output"); DATA=Path(r"F:\even-codex\us-stock-data")
px=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce")
prime=px.loc[px.index>=pd.Timestamp("2019-01-01")]; px=px.loc[:,prime.notna().sum()>=300]
idx=px.index; close=px; ret=px.pct_change()
# 近30日滚动波动（日度）
vol30=ret.rolling(30).std()

df=pd.read_csv(OUT/"kb_llm_sentiment.csv",encoding="utf-8-sig"); df["pdf_date"]=pd.to_datetime(df["pdf_date"],errors="coerce"); df=df.dropna(subset=["pdf_date"])
# 只取强多信号；过滤指数类噪音
bull=df[(df["direction"]==1)&(df["strength"]>=0.5)].copy()
exclude={"QQQ","SPY","SPX","DIA","IWM","XLK","SOXX","QQQ"}
bull=bull[~bull["ticker"].isin(exclude)]
print("strong bull total", len(bull), "unique tickers", bull["ticker"].nunique())

def run(vol_thr=None, pct_thr=None):
    rows=[]
    for _,r in bull.iterrows():
        t=r["ticker"]
        if t not in close.columns: continue
        pos=idx.searchsorted(r["pdf_date"])
        if pos+2>=len(idx): continue
        entry=pos+1; exit_=entry+1
        ret1=close.iloc[exit_][t]/close.iloc[entry][t]-1.0
        if math.isnan(ret1): continue
        v=vol30.iloc[entry][t]
        if vol_thr is not None and (math.isnan(v) or v>vol_thr): continue
        if pct_thr is not None and not math.isnan(v):
            pct=vol30.iloc[entry].quantile(pct_thr)
            if v>pct: continue
        bench=(close.iloc[exit_]/close.iloc[entry]-1.0).mean()
        rows.append({"entry":idx[entry],"ticker":t,"fwd":ret1,"bench":bench,"vol":v})
    ev=pd.DataFrame(rows)
    if len(ev)==0: return None
    daily=ev.groupby("entry").agg(fwd=("fwd","mean"),bench=("bench","mean"),n=("fwd","size")).sort_index()
    active=(daily["fwd"]-daily["bench"])
    cum_v=(1+daily["fwd"]).cumprod()
    cum_b=(1+daily["bench"]).cumprod()
    dd=((cum_v/cum_v.cummax())-1).min()
    sharpe=daily["fwd"].mean()/daily["fwd"].std(ddof=0)*np.sqrt(252) if daily["fwd"].std()>0 else np.nan
    return {"filter":str(vol_thr or pct_thr),"n_signals":len(ev),"n_days":len(daily),"mean_fwd":daily["fwd"].mean()*100,"mean_bench":daily["bench"].mean()*100,"mean_active":active.mean()*100,"hit":(active>0).mean(),"cum_fwd":(cum_v.iloc[-1]-1)*100,"cum_bench":(cum_b.iloc[-1]-1)*100,"sharpe":sharpe,"maxdd":dd*100}

res=[]
for label, kw in [("All", {}), ("vol<=怎2.0%", {"vol_thr":0.02}), ("vol<=1.5%", {"vol_thr":0.015}), ("lowest80%vol", {"pct_thr":0.80})]:
    r=run(**kw)
    if r: res.append(r); print({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()})
pd.DataFrame(res).to_csv(OUT/"llm_nextday_strategy.csv",index=False,encoding="utf-8-sig")
print("saved llm_nextday_strategy.csv")
