# -*- coding: utf-8 -*-
import os
"""Rolling walk-forward validation + cost sensitivity (monthly stock momentum)."""
from pathlib import Path
import numpy as np, pandas as pd
DATA=Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data"); from _paths import OUT
DAYS=252; START=20000.0
stk=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce").loc[:,lambda d:d.count()>=2400]
def ml(x): return x.resample("ME").last()
def mom(px,p,k):
    m=ml(px); return m.shift(k)/m.shift(p+k)-1

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
    ss=(g-cost).clip(lower=-0.5); nav=(1+ss).cumprod()*START
    return ss,nav

def shp(ret):
    r=ret.dropna()
    if len(r)<30: return np.nan
    ann=(1+r).prod()**(DAYS/len(r))-1
    vol=r.std(ddof=1)*np.sqrt(DAYS)
    return ann/vol if vol>0 else np.nan
def summarize(ret):
    nav=(1+ret).cumprod()*START
    ann=(nav.iloc[-1]/START)**(DAYS/len(nav))-1
    vol=ret.std(ddof=1)*np.sqrt(DAYS)
    mdd=(nav/nav.cummax()-1).min()
    return ann,vol,ann/vol if vol>0 else np.nan,mdd,nav.iloc[-1]

# candidate circle
periods=[3,6,9,12]; skips=[0,1]; tops=[10,20]
cands=[(p,k,t) for p in periods for k in skips for t in tops]
cache={}
for c in cands:
    cache[c]=run_fast(stk,mom(stk,c[0],c[1]),c[2],5)  # 5bps costs in walk-forward

# rolling choose by IS (data up to year-end), apply to next year
cutoffs=[pd.Timestamp(y,12,31) for y in range(2020,2026)]
oos=[]; history=[]
for i,cut in enumerate(cutoffs):
    best=None
    for c in cands:
        r,_=cache[c]
        m=(r.index>=pd.Timestamp('2017-02-01'))&(r.index<=cut)
        s=shp(r[m])
        if s is not None and (best is None or s>best[1]):
            best=(c,float(s))
    c=best[0]; sleft=pd.Timestamp(cut+pd.Timedelta(days=1))
    sright=cutoffs[i+1] if i+1<len(cutoffs) else pd.Timestamp(2026,8,14)
    seg=cache[c][0][(cache[c][0].index>=sleft)&(cache[c][0].index<=sright)]
    oos.append(seg)
    history.append(dict(selection_end=str(cut.date()), chosen=f"{c[0]}m skip{c[1]} top{c[2]}", is_sharpe=round(best[1],3)))
    print(f"Selected at {cut.date()}: {c[0]}m skip{c[1]} top{c[2]} | IS Sharpe {best[1]:.2f}")

wf_ret=pd.concat(oos).sort_index()
ann,vol,sh,mdd,final=summarize(wf_ret)
print("\n=== Rolling walk-forward strategy (2021 - 2026-08, costs 5bps) ===")
print(f"Annual return {ann*100:.2f}% | Vol {vol*100:.2f}% | Sharpe {sh:.2f} | MaxDD {mdd*100:.1f}% | Final ${final:,.0f}")

# comparisons on same OOS window
win=(pd.Timestamp('2021-01-01'),pd.Timestamp('2026-08-14'))
for name,c in [("Fixed 6m top20",(6,0,20)),("Fixed 9m top10",(9,0,10))]:
    r,_=cache[c]
    rr=r[(r.index>=win[0])&(r.index<=win[1])]
    a,v,s,m,f2=summarize(rr)
    print(f"{name:<22} OOS: ann={a*100:5.1f}%  vol={v*100:5.1f}%  sh={s:.2f}  mdd={m*100:6.1f}%  final=${f2:,.0f}")
spy=pd.read_csv(r"F:\even-codex\panda\backtest\prices_2016.csv",index_col=0,parse_dates=True)['SPY']
sr=spy.pct_change()[(spy.index>=win[0])&(spy.index<=win[1])]
a,v,s,m,f2=summarize(sr)
print(f"{'BuyHold SPY':<22} OOS: ann={a*100:5.1f}%  vol={v*100:5.1f}%  sh={s:.2f}  mdd={m*100:6.1f}%  final=${f2:,.0f}")

# yearly for walk-forward
yr=(1+wf_ret).groupby(wf_ret.index.year).prod()-1
print("\nWalk-forward yearly returns (%):")
print(yr.round(1).to_string())

hist=pd.DataFrame(history); hist.to_csv(OUT/"walkforward_selections.csv",index=False,encoding="utf-8-sig")
pd.DataFrame({"ret":wf_ret}).to_csv(OUT/"walkforward_returns.csv",encoding="utf-8-sig")

print("\n=== Cost sensitivity (fixed 9m top10, full period) ===")
for bps in [0,5,10,25,50,100]:
    r,nav=run_fast(stk,mom(stk,9,0),10,bps)
    a,v,s,m,f2=summarize(r)
    print(f"cost {bps:>4}bps | ann={a*100:5.1f}% | vol={v*100:5.1f}% | sharpe={s:.3f} | mdd={m*100:6.1f}% | final=${f2:>10,.0f}")
