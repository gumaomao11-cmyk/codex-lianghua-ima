# -*- coding: utf-8 -*-
import os
"""影子策略对比：把 6m/3m/9m × 前10/15/20 等候选用本地数据从 paper 起始日模拟到现在，与实际 paper 账户对比。"""
import json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
DATA=Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data"); IDX=Path(os.environ.get("ETFS_REF_FILE") or r"F:\even-codex\panda\backtest\prices_2016.csv")
from _paths import OUT
LOG=OUT/"paper_log.csv"; STATE=OUT/"paper_state.json"
DAYS=252; START=20000.0
stk=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce").loc[:,lambda d:d.count()>=2400]
spy=pd.read_csv(IDX,index_col=0,parse_dates=True)['SPY']
def ml(x): return x.resample("ME").last()
def mom(px,p,k): m=ml(px); return m.shift(k)/m.shift(p+k)-1
def mvol(px,l=60): d=px.pct_change(); return (d.rolling(l).std()*np.sqrt(DAYS)).resample("ME").last()
def weekly_scores(px, p=6, k=1):
    """周频 6m-skip1 动量近似：每周最后一个交易日的日线动量(每周换仓)。"""
    span_p=p*21; span_k=k*21
    mom = px.div(px.shift(span_p+span_k)).sub(1.0)
    s = pd.Series(np.arange(len(px)), index=px.index)
    last = s.groupby(px.index.to_period("W")).last().values.astype(int)
    return mom.iloc[last]

def accel_scores(px, p=6, k=1, am=1, wm=0.5):
    """月频复合：wm*动量 + (1-wm)*近 am 个月收益"""
    m = mom(px, p, k)
    a = ml(px).pct_change(am)
    return wm*m + (1-wm)*a

def weekly_accel_scores(px, p=6, k=1, am=1, wm=0.5):
    """周频复合：wm*动量 + (1-wm)*近 am 个月日线收益"""
    span_p=p*21; span_k=k*21; span_a=am*21
    momw = px.div(px.shift(span_p+span_k)).add(-1.0)
    accw = px.div(px.shift(span_a)).add(-1.0)
    sc = wm*momw + (1-wm)*accw
    s = pd.Series(np.arange(len(px)), index=px.index)
    last = s.groupby(px.index.to_period("W")).last().values.astype(int)
    return sc.iloc[last]
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
    g=(Wdf*dr).sum(axis=1).fillna(0.0)
    if vol_target:
        scale=np.ones(len(day))
        for s in range(1,T):
            ix2=np.searchsorted(day,me[s],side="right")
            if ix2>=len(day): continue
            look=g.loc[:pd.Timestamp(day[ix2])].iloc[-(61):-1].dropna()
            if len(look)>=21 and look.std(ddof=1)>0: scale[ix2]=min(1.0,vol_target/(look.std(ddof=1)*np.sqrt(DAYS)))
        for s in range(1,T):
            ix3=np.searchsorted(day,me[s],side="right")
            if ix3>=len(day): continue
            lo=day[ix3]; up=me[s+1] if s+1<T else day[-1]; sel=(day>=lo)&(day<=up)
            if np.any(sel): scale[sel]=scale[ix3]
        Wdf=Wdf.mul(scale,axis=0); g=(Wdf*dr).sum(axis=1).fillna(0.0)
    cost=pd.Series(0.0,index=g.index); prev=np.zeros(M)
    for s in range(1,T):
        ix=np.searchsorted(day,me[s],side="right")
        if ix>=len(day): continue
        wn=Wmat[slot[ix]]; to=np.abs(wn-prev).sum()/2.0; cost.iloc[ix]=to*cost_bps/10000.0; prev=wn.copy()
    return (g-cost).clip(lower=-0.5)
def summarize(seg):
    if len(seg)==0: return (0,0,0,0,START)
    nav=(1+seg).cumprod()*START
    ret=nav.iloc[-1]/START-1
    vol=seg.std(ddof=1)*np.sqrt(DAYS)
    sh=ret/vol if vol>0 else np.nan
    dd=(nav/nav.cummax()-1).min()
    return (ret,vol,sh,dd,nav.iloc[-1])

state=json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"start_date":"2026-08-18"}
start=pd.Timestamp(state["start_date"])
print(f"Paper start date: {start.date()}")
spy_seg=spy.pct_change().loc[start:].fillna(0.0)
sr,sv,ss,sdd,sf=summarize(spy_seg); print(f"SPY since start: ret={sr*100:.2f}% sh={ss:.2f} mdd={sdd*100:.1f}%")
print()
vm=mvol(stk,60)
configs=[
    ("6m skip1 top10  (paper)", mom(stk,6,1), 10, 10, None),
    ("6m skip1 top10 月频+vol25", mom(stk,6,1), 10, 10, 0.25),
    ("6m skip1 top10  (周频)", weekly_scores(stk,6,1), 10, 10, None),
    ("6m skip1 top10 周频+vol25", weekly_scores(stk,6,1), 10, 10, 0.25),
    ("6m skip1 top15", mom(stk,6,1), 15, 10, None),
    ("6m skip1 top20  (稳健)", mom(stk,6,1), 20, 10, None),
    ("3m skip1 top10", mom(stk,3,1), 10, 10, None),
    ("9m top10", mom(stk,9,0), 10, 10, None),
    ("6m skip1 top10 + vol25", mom(stk,6,1), 10, 10, 0.25),
    ("6m accel top10 月频", accel_scores(stk,6,1), 10, 10, None),
    ("6m accel top10 月频+vol25", accel_scores(stk,6,1), 10, 10, 0.25),
    ("6m accel top10 周频", weekly_accel_scores(stk,6,1), 10, 10, None),
    ("6m accel top10 周频+vol25", weekly_accel_scores(stk,6,1), 10, 10, 0.25),
]
rows=[("SPY (reference)",)*3+("(ref)",)*2]
for name,sc,top,cost,vt in configs:
    r=run_fast(stk,sc,top,cost,vol_m=vm,vol_target=vt)
    seg=r.loc[start:]
    a,v,s,d,f=summarize(seg)
    print(f"{name:<26} ret={a*100:+6.2f}%  vol={v*100:5.1f}%  sh={s:5.2f}  mdd={d*100:6.1f}%  final=${f:,.0f}")
