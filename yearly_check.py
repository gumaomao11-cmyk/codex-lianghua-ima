# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path(r"F:\even-codex\lianghua2\backtest_output")
DATA = Path(r"F:\even-codex\us-stock-data")
IDX = Path(r"F:\even-codex\panda\backtest\prices_2016.csv")
DAYS=252; START=20000.0

etf = pd.read_csv(IDX, index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce").ffill()
stk = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
stk = stk.loc[:, stk.count() >= 2400]

def ml(x): return x.resample("ME").last()
def mom(px, period, skip):
    m=ml(px); return m.shift(skip)/m.shift(skip+period)-1
def bh(px,t):
    s=px[t].dropna(); r=s.pct_change().fillna(0); y=(1+r).groupby(r.index.year).prod()-1
    return y
yearly = {}
for t in ['SPY','QQQ','SPMO']:
    yearly["BH "+t] = bh(etf,t)

def monthly_run(px, score, top, cost_bps):
    cols=list(px.columns); dr=px.pct_change().fillna(0.0)
    me=pd.DatetimeIndex(score.index); W=pd.DataFrame(0.0,index=px.index,columns=cols); rd=[]
    for i in range(1,len(me)):
        d=me[i]; w=pd.Series(0.0,index=cols); s=score.loc[d].dropna()
        if len(s)>0:
            ta=s.sort_values(ascending=False).index[:top].tolist(); w[ta]=1.0/len(ta)
        end=me[i+1] if i+1<len(me) else px.index[-1]
        days=px.index[(px.index>d)&(px.index<=end)]
        if len(days):
            for c in cols: W.loc[days,c]=w[c]
            rd.append(days[0])
    g=(W*dr).sum(axis=1).fillna(0)
    cost=pd.Series(0.0,index=g.index); prev=pd.Series(0.0,index=cols)
    for r0 in rd:
        d0=r0-pd.Timedelta(days=1)
        wn=W.loc[d0] if d0 in W.index else W.loc[r0]
        to=(wn-prev).abs().sum()/2; cost.loc[r0]=to*cost_bps/10000.0; prev=wn.copy()
    ss=(g-cost).clip(lower=-0.5); nav=(1+ss).cumprod()*START
    y=(1+ss).groupby(ss.index.year).prod()-1
    return y, nav

m6s = mom(stk,6,0); m12s=mom(stk,12,1); m6e=mom(etf,6,0)
for name,px,sc,top,cost in [
    ("StockMom6_top20", stk, m6s, 20, 10),
    ("StockMom12-1_top10", stk, m12s, 10, 10),
    ("ETF_Mom6_top2", etf, m6e, 2, 5),
]:
    y,nav = monthly_run(px,sc,top,cost); yearly["ST | "+name]=y

dfy = pd.DataFrame(yearly).sort_index()*100
dfy.index = dfy.index.astype(str)
dfy.to_csv(OUT/"yearly_returns.csv", encoding="utf-8-sig")
print(dfy.round(1).to_string())
print("\nBest year (cell %) and worst year (cell %) by name:")
for c in dfy.columns:
    print(c, "best:", f"{dfy[c].max():.1f}%", dfy[c].idxmax(), "| worst:", f"{dfy[c].min():.1f}%", dfy[c].idxmin(), "| positive years:", int((dfy[c]>0).sum()), "/", len(dfy[c]))
