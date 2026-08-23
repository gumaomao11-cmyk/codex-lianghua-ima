# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(r"F:\even-codex\lianghua2\backtest_output")
DATA=Path(r"F:\even-codex\us-stock-data"); IDX=Path(r"F:\even-codex\panda\backtest\prices_2016.csv")
START=20000.0
etf=pd.read_csv(IDX,index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce").ffill()
stk=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce")
stk=stk.loc[:,stk.count()>=2400]
def ml(px): return px.resample("ME").last()
def mom(px,p,s):
    m=ml(px); return m.shift(s)/m.shift(s+p)-1
mret=ml(stk).pct_change().fillna(0.0); dr=stk.pct_change().fillna(0.0)
sc6=mom(stk,6,0)
cols=list(stk.columns); W=pd.DataFrame(0.0,index=stk.index,columns=cols)
me=pd.DatetimeIndex(mret.index)
for i in range(1,len(me)):
    d=me[i]; s=sc6.loc[d].dropna()
    w=pd.Series(0.0,index=cols)
    if len(s):
        ta=s.sort_values(ascending=False).index[:20].tolist(); w[ta]=1.0/len(ta)
    end=me[i+1] if i+1<len(me) else stk.index[-1]
    days=stk.index[(stk.index>d)&(stk.index<=end)]
    if len(days):
        for c in cols: W.loc[days,c]=w[c]

def sim(reg=None):
    eff = W if reg is None else W.mul(reg.shift(1), axis=0)
    gross=(eff*dr).sum(axis=1).fillna(0.0)
    cost=eff.diff().abs().sum(axis=1)/2*10/10000.0
    s=(gross-cost).clip(lower=-0.5)
    nav=(1+s).cumprod()*START
    dd=nav/nav.cummax()-1
    return nav, dd, ((eff.abs().sum(axis=1)>1e-6).astype(int))

# Strategies
reg50 = (etf['SPY']>etf['SPY'].rolling(50,min_periods=50).mean()).astype(float)
nav_base,dd_base,act_base = sim(None)
nav_50, dd_50, act_50 = sim(reg50)
nav_200m, dd_200m, act_200m = None,None,None  # keep monthly variant in table only

# benchmarks
spy_nav = etf['SPY']/etf['SPY'].iloc[0]*START
qqq_nav = etf['QQQ']/etf['QQQ'].iloc[0]*START

fig, axes = plt.subplots(2,2, figsize=(14,9), constrained_layout=True)
ax=axes[0,0]
for nm, n in [("SPY buy&hold",spy_nav),("QQQ buy&hold",qqq_nav),("Base momentum (no cash rule)",nav_base)]:
    ax.plot(n.index, n.values/START, lw=1.4, alpha=.8, label=nm)
ax.plot(nav_50.index, nav_50.values/START, lw=2.2, color='tab:red', label='Recommended: Mom6_top20 + daily SPY<50d -> cash')
ax.set_yscale('log'); ax.legend(fontsize=8); ax.set_title('Full period growth of $20,000 (net of costs)'); ax.grid(alpha=.3)

ax=axes[0,1]
oos = nav_50.index>=pd.Timestamp("2022-01-01")
for nm, n in [("SPY buy&hold",spy_nav),("QQQ buy&hold",qqq_nav),("Base momentum (no cash rule)",nav_base)]:
    ax.plot(n.index[oos], (n[oos]/START), lw=1.4, alpha=.8, label=nm)
ax.plot(nav_50.index[oos], nav_50[oos]/START, lw=2.2, color='tab:red', label='Recommended')
ax.set_title('Out-of-sample 2022+ growth of $20,000'); ax.legend(fontsize=8); ax.grid(alpha=.3)

ax=axes[1,0]
ax.plot(dd_base.index, dd_base.values*100, lw=1.4, alpha=.8, label='Base momentum (no cash rule)')
ax.plot(dd_50.index, dd_50.values*100, lw=2.0, color='tab:red', label='Recommended (cash when weak)')
y = dd_50.copy()
in_cash = act_50==0
ax.fill_between(y.index, -35, -33, where=in_cash, color='gray', alpha=.35, interpolate=True, label='Cash days')
ax.set_title('Drawdown (%) - Recommended vs Base'); ax.legend(fontsize=8); ax.grid(alpha=.3)

ax=axes[1,1]
monthly = nav_50.resample("ME").last().pct_change().dropna()
years = (1+monthly).groupby(monthly.index.year).prod()-1
yrs = years.loc[2017:2026]
ax.bar(yrs.index.astype(str), yrs.values*100, color=['#2a9d8f' if v>=0 else '#e76f51' for v in yrs.values])
ax.axhline(0,color='k',lw=.8); ax.set_title('Recommended strategy yearly returns (%)'); ax.grid(alpha=.3)
plt.savefig(OUT/"final_strategy_chart.png", dpi=150)
plt.close(fig)

# save nav csv
out = pd.DataFrame({"date": nav_base.index})
for nm,n in [("base_mom6_top20",nav_base),("recommended_daily50_cash",nav_50)]:
    out[nm]=n.values
out.to_csv(OUT/"final_nav.csv", index=False)
print("chart saved:", OUT/"final_strategy_chart.png")
print("final NAV CSV:", OUT/"final_nav.csv")
