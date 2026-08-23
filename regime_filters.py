# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path(r"F:\even-codex\lianghua2\backtest_output")
DATA = Path(r"F:\even-codex\us-stock-data")
IDX = Path(r"F:\even-codex\panda\backtest\prices_2016.csv")
START=20000.0

etf = pd.read_csv(IDX, index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce").ffill()
stk = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
stk = stk.loc[:, stk.count() >= 2400]

def ml(px): return px.resample("ME").last()
def mom(px,period,skip):
    m=ml(px); return m.shift(skip)/m.shift(skip+period)-1
mret = ml(stk).pct_change().fillna(0.0)

# base strategy returns (no filter) - monthly
def base_returns(px, mret, sc, top, cost_bps):
    dates=list(mret.index); w_prev=pd.Series(0.0,index=px.columns); rets=[]; rng=[]
    for i in range(1,len(dates)):
        s=sc.loc[dates[i-1]].dropna()
        if len(s)==0: continue
        ta=s.sort_values(ascending=False).index[:top].tolist()
        w=pd.Series(0.0,index=px.columns); w[ta]=1.0/len(ta)
        to=(w-w_prev).abs().sum()/2; c=to*cost_bps/10000.0
        rets.append(float((w*mret.loc[dates[i]]).sum())-c); rng.append(dates[i]); w_prev=w.copy()
    return pd.Series(rets,index=pd.DatetimeIndex(rng))

sc6 = mom(stk,6,0)
base6 = base_returns(stk, mret, sc6, 20, 10)

# self filters computed at month-end from base strategy trailing sum
def self_filter(base, k):
    trail = base.rolling(k, min_periods=k).sum()
    return (trail > 0).reindex(mret.index).fillna(False)

def name_pos_filter(sc, period_days=0):
    return sc > 0

def run_with(px, mret, sc, top, cost_bps, filt=None, name_pos=False, sd=None, ed=None, cap=START):
    dates=[d for d in list(mret.index) if (sd is None or d>=sd) and (ed is None or d<=ed)]
    value=cap; rets=[]; rng=[]; jour=[]; w_prev=pd.Series(0.0,index=px.columns)
    for i in range(1,len(dates)):
        prev_d=dates[i-1]; s=sc.loc[prev_d].dropna()
        if name_pos: s=s[s>0]
        inv=True
        if filt is not None:
            inv=bool(filt.loc[prev_d]) if prev_d in filt.index else True
        if inv and len(s)>0:
            ta=s.sort_values(ascending=False).index[:top].tolist()
            w=pd.Series(0.0,index=px.columns); w[ta]=1.0/len(ta)
        else:
            w=pd.Series(0.0,index=px.columns)
        to=(w-w_prev).abs().sum()/2; c=to*cost_bps/10000.0
        r=float((w*mret.loc[dates[i]]).sum())-c; r=max(r,-0.5); value*=(1+r)
        rets.append(r); rng.append(dates[i]); jour.append((w.abs().sum()>1e-6)); w_prev=w.copy()
    if not rets: return None
    ret=pd.Series(rets,index=pd.DatetimeIndex(rng)); nav=(1+ret).cumprod()*cap
    ann=(nav.iloc[-1]/cap)**(12/len(ret))-1; vol=ret.std(ddof=1)*np.sqrt(12)
    sh=ann/vol if vol>0 else np.nan; dd=(nav/nav.cummax()-1).min(); cal=ann/abs(dd) if dd<0 else np.nan
    return dict(nav=nav,ret=ret,ann=ann,vol=vol,sharpe=sh,mdd=dd,calmar=cal,
                final=nav.iloc[-1],active=float(np.mean(jour)),cash=int(len(jour)-sum(jour)))

IS=(pd.Timestamp("2017-01-31"),pd.Timestamp("2021-12-31"))
OOS=(pd.Timestamp("2022-01-31"),pd.Timestamp("2026-08-31"))

cands = [
    ("Base: Mom6_top20", sc6, 20, None, False),
    ("+selffilter_6m", sc6, 20, self_filter(base6,6), False),
    ("+selffilter_12m", sc6, 20, self_filter(base6,12), False),
    ("+name_pos(mom>0)", sc6, 20, None, True),
    ("+SPY>200dSMA", sc6, 20, (etf['SPY']>etf['SPY'].rolling(200,min_periods=200).mean()).reindex(mret.index).fillna(False), False),
]
rows=[]
for name,sc,top,flt,np_ in cands:
    isr=run_with(stk,mret,sc,top,10,flt,np_,*IS)
    oosr=run_with(stk,mret,sc,top,10,flt,np_,*OOS)
    rows.append(dict(strategy=name,
        IS_ann=f"{isr['ann']*100:.1f}%", IS_sharpe=f"{isr['sharpe']:.2f}", IS_mdd=f"{isr['mdd']*100:.1f}%", IS_active=f"{isr['active']*100:.0f}%",
        OOS_ann=f"{oosr['ann']*100:.1f}%", OOS_sharpe=f"{oosr['sharpe']:.2f}", OOS_mdd=f"{oosr['mdd']*100:.1f}%",
        OOS_active=f"{oosr['active']*100:.0f}%", OOS_cash=oosr['cash'], OOS_final=f"${oosr['final']:,.0f}"))
rdf=pd.DataFrame(rows); rdf.to_csv(OUT/"regime_filters.csv",index=False,encoding="utf-8-sig")
print(rdf.to_string(index=False))
