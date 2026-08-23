# -*- coding: utf-8 -*-
"""Fast vectorized optimization & robustness experiments."""
from pathlib import Path
import numpy as np, pandas as pd

DATA=Path(r"F:\even-codex\us-stock-data"); IDX=Path(r"F:\even-codex\panda\backtest\prices_2016.csv")
OUT=Path(r"F:\even-codex\lianghua2\backtest_output"); OUT.mkdir(parents=True, exist_ok=True)
DAYS=252; START=20000.0

stk=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce").loc[:,lambda d:d.count()>=2400]
def ml(x): return x.resample("ME").last()
def mom(px,period,skip):
    m=ml(px); return m.shift(skip)/m.shift(skip+period)-1
def monthly_vol(px,look=60):
    d=px.pct_change(); v=d.rolling(look).std()*np.sqrt(DAYS); return v.resample("ME").last()

def run_fast(px, scores, top, cost_bps, vol_m=None, risk_weight=False, vol_target=None):
    cols=list(px.columns); M=len(cols); dr=px.pct_change().fillna(0.0)
    me=np.array(pd.DatetimeIndex(scores.index)); day=np.array(px.index)
    slot=np.searchsorted(me,day,side="right")-1
    exact=(slot>=0)&(day==me[np.clip(slot,0,len(me)-1)])
    slot=np.clip(slot-exact.astype(int),0,len(me)-1)
    T=len(me); Wmat=np.zeros((T,M))
    for s in range(1,T):
        d=pd.Timestamp(me[s]); w=np.zeros(M); sc=scores.loc[d].dropna()
        if len(sc)>0:
            ta=list(sc.sort_values(ascending=False).index[:top]); ix=[cols.index(c) for c in ta]
            w[ix]=1.0/len(ta)
            if risk_weight:
                row=vol_m.loc[d].reindex(ta).values if vol_m is not None else None
                if row is not None:
                    inv=np.where(np.isfinite(row) & (row>1e-6),1.0/np.maximum(row,1e-6),1.0)
                    w[ix]=inv/np.sum(inv)
        Wmat[s]=w
    Wdf=pd.DataFrame(Wmat[slot],index=px.index,columns=cols)
    ug=(Wdf*dr).sum(axis=1).fillna(0.0)
    if vol_target:
        scale=np.ones(len(day))
        for s in range(1,T):
            ix2=np.searchsorted(day,me[s],side="right")
            if ix2>=len(day): continue
            r0=day[ix2]
            look=ug.loc[:pd.Timestamp(r0)].iloc[-(61):-1].dropna()
            if len(look)>=21 and look.std(ddof=1)>0:
                scale[ix2]=min(1.0,vol_target/(look.std(ddof=1)*np.sqrt(DAYS)))
        for s in range(1,T):
            ix3=np.searchsorted(day,me[s],side="right")
            if ix3>=len(day): continue
            lo=day[ix3]
            up=me[s+1] if s+1<T else day[-1]
            sel=(day>=lo)&(day<=up)
            if np.any(sel):
                scale[sel]=scale[ix3]
        Wdf=Wdf.mul(scale,axis=0)
    g=(Wdf*dr).sum(axis=1).fillna(0.0)
    cost=pd.Series(0.0,index=g.index); prev=np.zeros(M)
    for s in range(1,T):
        ix=np.searchsorted(day,me[s],side="right")
        if ix>=len(day): continue
        wn=Wmat[slot[ix]]
        to=np.abs(wn-prev).sum()/2.0
        cost.iloc[ix]=to*cost_bps/10000.0
        prev=wn.copy()
    ss=(g-cost).clip(lower=-0.5); nav=(1+ss).cumprod()*START
    return ss,nav,(Wdf.abs().sum(axis=1)>1e-6).mean()

def stats(ret,nav):
    if len(nav)==0: return dict(ann=0,vol=0,sharpe=np.nan,mdd=0,final=START)
    ann=(nav.iloc[-1]/START)**(DAYS/len(nav))-1
    vol=ret.std(ddof=1)*np.sqrt(DAYS)
    sharpe=ann/vol if vol>0 else np.nan
    mdd=(nav/nav.cummax()-1).min()
    return dict(ann=ann,vol=vol,sharpe=sharpe,mdd=mdd,final=nav.iloc[-1])

vm=monthly_vol(stk,60)
rows=[]
for p in [3,6,9,12]:
    for k in [0,1]:
        sc=mom(stk,p,k)
        for top in [10,20]:
            ret,nav,act=run_fast(stk,sc,top,10,vol_m=vm)
            is_msk=(ret.index>=pd.Timestamp('2017-02-01'))&(ret.index<=pd.Timestamp('2021-12-31'))
            oo_msk=(ret.index>=pd.Timestamp('2022-01-01'))
            isre=stats(ret[is_msk],nav[is_msk]); oore=stats(ret[oo_msk],nav[oo_msk]); fre=stats(ret,nav)
            rows.append(dict(period=p,skip=k,top=top,is_sharpe=isre['sharpe'],oos_sharpe=oore['sharpe'],
                             full_sharpe=fre['sharpe'],is_ann=isre['ann'],oos_ann=oore['ann'],
                             full_ann=fre['ann'],oos_mdd=oore['mdd'],oos_final=oore['final']))
df=pd.DataFrame(rows).sort_values('oos_sharpe',ascending=False)
df.to_csv(OUT/"optimization_sweep.csv",index=False,encoding="utf-8-sig")
print("=== Parameter sweep (ranked by OOS Sharpe) ===")
print(df.round(3).to_string(index=False))
best=df.iloc[0]
print("\nBest OOS: lookback=%d skip=%d top=%d OOS.sh=%.2f IS.sh=%.2f" % (best.period,best.skip,best.top,best.oos_sharpe,best.is_sharpe))

def show(name,ret,nav,act):
    st=stats(ret,nav)
    print(f"| {name:<30} | ann={st['ann']*100:5.1f}% | vol={st['vol']*100:5.1f}% | sh={st['sharpe']:5.2f} | mdd={st['mdd']*100:6.1f}% | ${st['final']:>9,.0f} | in={act*100:3.0f}% |")

base=mom(stk,6,0); bsm=mom(stk,int(best.period),int(best.skip))
cases=[
    ("Base 6m top20", base,20,{}),
    ("6m top20 risk-parity", base,20,{'risk_weight':True}),
    ("6m top20 vol25", base,20,{'vol_target':0.25}),
    ("6m top20 rp+vol25", base,20,{'risk_weight':True,'vol_target':0.25}),
    ("BestOOS pure", bsm,int(best.top),{}),
    ("BestOOS + vol25", bsm,int(best.top),{'vol_target':0.25}),
]
print("\n=== Upgrade candidates ===")
for name,sc,top,kw in cases:
    ret,nav,act=run_fast(stk,sc,top,10,vol_m=vm,**kw)
    show(name,ret,nav,act)
