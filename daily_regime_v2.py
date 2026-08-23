# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from pathlib import Path
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

# build monthly base weights (daily series, Mom6 top20)
cols=list(stk.columns); W=pd.DataFrame(0.0,index=stk.index,columns=cols)
me=pd.DatetimeIndex(mret.index); wprev=pd.Series(0.0,index=cols)
rebal=[]
for i in range(1,len(me)):
    d=me[i]; s=sc6.loc[d].dropna()
    if len(s):
        ta=s.sort_values(ascending=False).index[:20].tolist()
        w=pd.Series(0.0,index=cols); w[ta]=1.0/len(ta)
    else:
        w=pd.Series(0.0,index=cols)
    end=me[i+1] if i+1<len(me) else stk.index[-1]
    days=stk.index[(stk.index>d)&(stk.index<=end)]
    if len(days):
        for c in cols: W.loc[days,c]=w[c]
        rebal.append(days[0])
    wprev=w.copy()

daily_ret = dr
spy=etf['SPY']
def run_daily(filter_name, window):
    reg = (spy > spy.rolling(window,min_periods=window).mean()).astype(float).shift(1)
    reg = reg.reindex(W.index).fillna(0.0)
    eff = W.mul(reg, axis=0)
    gross=(eff*daily_ret).sum(axis=1).fillna(0.0)
    # turnover cost on daily position changes
    dW=eff.diff().abs().sum(axis=1)/2
    cost=dW*10/10000.0
    strat=(gross-cost).clip(lower=-0.5)
    nav=(1+strat).cumprod()*START
    def stats(lo,hi):
        sub=strat.loc[lo:hi]; subnav=nav.loc[lo:hi] if False else None
        nv=(1+sub).cumprod()*START
        ann=(nv.iloc[-1]/START)**(252/len(sub))-1 if len(sub)>0 else 0
        vol=sub.std(ddof=1)*np.sqrt(252); sh=ann/vol if vol>0 else np.nan
        dd=(nv/nv.cummax()-1).min(); cal=ann/abs(dd) if dd<0 else np.nan
        act=(eff.loc[lo:hi].abs().sum(axis=1)>1e-6).mean()
        return ann,vol,sh,dd,cal,nv.iloc[-1],act
    out={}
    for lab,sd,ed in [("IS",pd.Timestamp("2017-01-01"),pd.Timestamp("2021-12-31")),
                      ("OOS",pd.Timestamp("2022-01-01"),pd.Timestamp("2026-12-31"))]:
        a,v,sh,dd,cal,f,act=stats(sd,ed)
        out[lab]=dict(ann=a,vol=v,sharpe=sh,mdd=dd,calmar=cal,final=f,active=act)
    return out, nav, eff

rows=[]
for name,win in [("SPY>50dSMA",50),("SPY>100dSMA",100),("SPY>200dSMA",200)]:
    r,nav,eff = run_daily(name,win)
    rows.append(dict(strategy=f"Mom6_top20 + daily {name}",
        IS_ann=f"{r['IS']['ann']*100:.1f}%", IS_sharpe=f"{r['IS']['sharpe']:.2f}", IS_mdd=f"{r['IS']['mdd']*100:.1f}%", IS_active=f"{r['IS']['active']*100:.0f}%",
        OOS_ann=f"{r['OOS']['ann']*100:.1f}%", OOS_sharpe=f"{r['OOS']['sharpe']:.2f}", OOS_mdd=f"{r['OOS']['mdd']*100:.1f}%",
        OOS_active=f"{r['OOS']['active']*100:.0f}%", OOS_final=f"${r['OOS']['final']:,.0f}"))
# benchmark base (no filter) daily
W0=W
gross=(W0*daily_ret).sum(axis=1).fillna(0.0)
dW0=W0.diff().abs().sum(axis=1)/2; cost0=dW0*10/10000.0; strat0=(gross-cost0).clip(lower=-0.5)
nav0=(1+strat0).cumprod()*START
def bs(lo,hi):
    sub=strat0.loc[lo:hi]; nv=(1+sub).cumprod()*START; ann=(nv.iloc[-1]/START)**(252/len(sub))-1
    vol=sub.std(ddof=1)*np.sqrt(252); sh=ann/vol if vol>0 else np.nan; dd=(nv/nv.cummax()-1).min(); return ann,vol,sh,dd, nv.iloc[-1]
for lab,sd,ed in [("IS",pd.Timestamp("2017-01-01"),pd.Timestamp("2021-12-31")),("OOS",pd.Timestamp("2022-01-01"),pd.Timestamp("2026-12-31"))]:
    a,v,sh,dd,f=bs(sd,ed)
rows.append(dict(strategy="Base Mom6_top20 (no cash filter)",
    IS_ann=f"{a*100:.1f}%", IS_sharpe=f"{sh:.2f}", IS_mdd=f"{dd*100:.1f}%", IS_active="100%",
    OOS_ann=f"{a*100:.1f}%", OOS_sharpe=f"{sh:.2f}", OOS_mdd=f"{dd*100:.1f}%", OOS_active="100%", OOS_final=f"${f:,.0f}"))
res=pd.DataFrame(rows); res.to_csv(OUT/"daily_regime_filters.csv",index=False,encoding="utf-8-sig")
print(res.to_string(index=False))

