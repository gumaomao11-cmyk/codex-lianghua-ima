# -*- coding: utf-8 -*-
"""Selected robustness checks: low-vol blend, IS-chosen OOS validation."""
from pathlib import Path
import numpy as np, pandas as pd
DATA=Path(r"F:\even-codex\us-stock-data")
OUT=Path(r"F:\even-codex\lianghua2\backtest_output")
DAYS=252; START=20000.0
stk=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce").loc[:,lambda d:d.count()>=2400]
def ml(x): return x.resample("ME").last()
def mom(px,period,skip):
    m=ml(px); return m.shift(skip)/m.shift(skip+period)-1
def mv(px,look=60):
    d=px.pct_change(); return (d.rolling(look).std()*np.sqrt(DAYS)).resample("ME").last()
def blend(momsc, vm, w=0.6):
    mr=momsc.rank(axis=1,pct=True)
    vr=vm.rank(axis=1,pct=True)
    return w*mr.fillna(0.5)+(1-w)*(1-vr).fillna(0.5)

def run_fast(px,scores,top,cost_bps,vol_m=None,vol_target=None):
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
    ug=(Wdf*dr).sum(axis=1).fillna(0.0)
    if vol_target:
        scale=np.ones(len(day))
        for s in range(1,T):
            ix2=np.searchsorted(day,me[s],side="right")
            if ix2>=len(day): continue
            look=ug.loc[:pd.Timestamp(day[ix2])].iloc[-(61):-1].dropna()
            if len(look)>=21 and look.std(ddof=1)>0: scale[ix2]=min(1.0,vol_target/(look.std(ddof=1)*np.sqrt(DAYS)))
        for s in range(1,T):
            ix3=np.searchsorted(day,me[s],side="right")
            if ix3>=len(day): continue
            lo=day[ix3]; up=me[s+1] if s+1<T else day[-1]; sel=(day>=lo)&(day<=up)
            if np.any(sel): scale[sel]=scale[ix3]
        Wdf=Wdf.mul(scale,axis=0)
    g=(Wdf*dr).sum(axis=1).fillna(0.0)
    cost=pd.Series(0.0,index=g.index); prev=np.zeros(M)
    for s in range(1,T):
        ix=np.searchsorted(day,me[s],side="right")
        if ix>=len(day): continue
        wn=Wmat[slot[ix]]; to=np.abs(wn-prev).sum()/2.0; cost.iloc[ix]=to*cost_bps/10000.0; prev=wn.copy()
    ss=(g-cost).clip(lower=-0.5); nav=(1+ss).cumprod()*START
    return ss,nav
def stat(ret,nav):
    ann=(nav.iloc[-1]/START)**(DAYS/len(nav))-1; vol=ret.std(ddof=1)*np.sqrt(DAYS)
    return ann,vol,(ann/vol if vol>0 else np.nan),(nav/nav.cummax()-1).min(),nav.iloc[-1]

vm=mv(stk,60)
cands=[
    ("6m top10", mom(stk,6,0), 10),
    ("3m-1 top10", mom(stk,3,1), 10),
    ("9m top10 (OOS best)", mom(stk,9,0), 10),
    ("6m lowvol-blend top10", blend(mom(stk,6,0), vm, 0.5), 10),
    ("3m-1 lowvol-blend top10", blend(mom(stk,3,1), vm, 0.5), 10),
]
print(f"{'name':<28} {'IS.sh':>6} {'OOS.sh':>7} {'Full.sh':>8} {'ann':>6} {'vol':>6} {'mdd':>7} {'final':>10}")
for name,sc,top in cands:
    ret,nav=run_fast(stk,sc,top,10,vol_m=vm)
    isr=ret[(ret.index>=pd.Timestamp('2017-02-01'))&(ret.index<=pd.Timestamp('2021-12-31'))]
    isn=nav[isr.index]; oo=ret[(ret.index>=pd.Timestamp('2022-01-01'))]; oon=nav[oo.index]
    a,v,sh,m,f=stat(ret,nav); ais,_,ish,_,_=stat(isr,isn); aos,_,osh,_,_=stat(oo,oon)
    print(f"{name:<28} {ish:6.2f} {osh:7.2f} {sh:8.2f} {a*100:5.1f}% {v*100:5.1f}% {m*100:6.1f}% ${f:>9,.0f}")
