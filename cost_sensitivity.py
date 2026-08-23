# -*- coding: utf-8 -*-
import os
"""季度用：成本敏感度（动量 6m skip1 top10）。"""
from pathlib import Path
import numpy as np, pandas as pd
DATA = Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data"); DAYS=252; START=20000.0
stk=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce").loc[:,lambda d:d.count()>=2400]
def ml(x): return x.resample("ME").last()
def mom(px,p,k): m=ml(px); return m.shift(k)/m.shift(p+k)-1
def run_fast(px,scores,top,cost_bps):
    cols=list(px.columns); M=len(cols); dr=px.pct_change().fillna(0.0)
    me=np.array(pd.DatetimeIndex(scores.index)); day=np.array(px.index)
    slot=np.searchsorted(me,day,side="right")-1
    exact=(slot>=0)&(day==me[np.clip(slot,0,len(me)-1)])
    slot=np.clip(slot-exact.astype(int),0,len(me)-1); T=len(me); Wmat=np.zeros((T,M))
    for s in range(1,T):
        d=pd.Timestamp(me[s]); w=np.zeros(M); sc=scores.loc[d].dropna()
        if len(sc)>0:
            ta=list(sc.sort_values(ascending=False).index[:top]); ix=[cols.index(c) for c in ta]; w[ix]=1.0/len(ta)
        Wmat[s]=w
    Wdf=pd.DataFrame(Wmat[slot],index=px.index,columns=cols)
    g=(Wdf*dr).sum(axis=1).fillna(0.0)
    cost=pd.Series(0.0,index=g.index); prev=np.zeros(M)
    for s in range(1,T):
        ix=np.searchsorted(day,me[s],side="right")
        if ix>=len(day): continue
        wn=Wmat[slot[ix]]; to=np.abs(wn-prev).sum()/2.0; cost.iloc[ix]=to*cost_bps/10000.0; prev=wn.copy()
    return (g-cost).clip(lower=-0.5)
def stats(ret):
    if len(ret)==0: return (0,0,0,0,START)
    nav=(1+ret).cumprod()*START
    ann=(nav.iloc[-1]/START)**(DAYS/len(nav))-1
    vol=ret.std(ddof=1)*np.sqrt(DAYS)
    sh=ann/vol if vol>0 else np.nan
    mdd=(nav/nav.cummax()-1).min()
    return (ann,vol,sh,mdd,nav.iloc[-1])
print("== 成本敏感度（6m skip1 top10，全期） ==")
print(f"{'cost':>6} | {'ann':>7} | {'vol':>6} | {'sharpe':>7} | {'mdd':>7} | {'final':>12}")
for c in [0,5,10,25,50,100]:
    r=run_fast(stk,mom(stk,6,1),10,c); a,v,s,d,f=stats(r)
    print(f"{c:>4}bps | {a*100:6.1f}% | {v*100:5.1f}% | {s:6.3f} | {d*100:6.1f}% | ${f:>11,.0f}")
