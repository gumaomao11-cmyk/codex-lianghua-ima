# -*- coding: utf-8 -*-
"""Corrected IS/OOS metrics (sub-period NAV resets to $20k) for top10 aggressive variants."""
from pathlib import Path
import numpy as np, pandas as pd
DATA=Path(r"F:\even-codex\us-stock-data"); OUT=Path(r"F:\even-codex\lianghua2\backtest_output")
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
    return ss

def stats_on_ret(ret, start_val=START):
    if len(ret)==0: return (np.nan,np.nan,np.nan,np.nan,start_val)
    nav=(1+ret).cumprod()*start_val
    ann=(nav.iloc[-1]/start_val)**(DAYS/len(nav))-1
    vol=ret.std(ddof=1)*np.sqrt(DAYS)
    sh=ann/vol if vol>0 else np.nan
    mdd=(nav/nav.cummax()-1).min()
    return (ann,vol,sh,mdd,nav.iloc[-1])

rows=[]
for p in [3,6,9,12]:
    for k in [0,1]:
        sc=mom(stk,p,k)
        for top in [10,20]:
            ret=run_fast(stk,sc,top,10)
            full=stats_on_ret(ret)
            ism=(ret.index>=pd.Timestamp('2017-02-01'))&(ret.index<=pd.Timestamp('2021-12-31'))
            isr=ret[ism].reset_index(drop=True); is_=stats_on_ret(isr)
            om=(ret.index>=pd.Timestamp('2022-01-01'))
            orr=ret[om].reset_index(drop=True); oo=stats_on_ret(orr)
            rows.append(dict(period=p,skip=k,top=top,
                             full_sh=full[2], full_ann=full[0], full_mdd=full[3], full_end=full[4],
                             is_sh=is_[2], is_ann=is_[0],
                             oos_sh=oo[2], oos_ann=oo[0], oos_mdd=oo[3], oos_end=oo[4]))
df=pd.DataFrame(rows).sort_values('oos_sh',ascending=False)
df.to_csv(OUT/'optimization_sweep_corrected.csv',index=False,encoding='utf-8-sig')
print(df.round(3).to_string(index=False))
print("\n=== Aggressive top10 picks (corrected OOS, from $20k reset) ===")
for _,r in df[df.top==10].iterrows():
    print(f"{int(r.period)}m skip{int(r.skip)} top10: full sh={r.full_sh:.2f} ann={r.full_ann*100:.1f}% | IS sh={r.is_sh:.2f} | OOS sh={r.oos_sh:.2f} ann={r.oos_ann*100:.1f}% mdd={r.oos_mdd*100:.1f}% end=${r.oos_end:,.0f}")
