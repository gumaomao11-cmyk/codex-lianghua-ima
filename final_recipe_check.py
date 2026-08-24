# -*- coding: utf-8 -*-
"""最终配方验证：纯动量 vs 动量+IMA词频(win60/λ1.2)，月度10只等权5bp，无LLM、无止盈止损。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA=Path(r"F:\even-codex\us-stock-data"); OUT=Path(r"F:\even-codex\lianghua+IMA\backtest_output")

px=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce")
recent=px.loc[px.index>=pd.Timestamp("2025-01-01")]; px=px.loc[:,recent.notna().sum()>=150]
ml=px.resample("ME").last(); mom=(ml.shift(1)/ml.shift(7)-1.0).replace([np.inf,-np.inf],np.nan)
daily_ret=px.pct_change().fillna(0.0); cols=list(px.columns)
fac=pd.read_csv(OUT/"kb_abstract_factors.csv",encoding="utf-8-sig")
fac["pdf_date"]=pd.to_datetime(fac["pdf_date"],errors="coerce"); fac=fac.dropna(subset=["pdf_date"])

def wscore(d,window=60):
    s=fac[fac.pdf_date<d]; s=s[s.pdf_date>=d-pd.Timedelta(days=window)]
    if s.empty: return pd.Series(dtype=float)
    g=s.groupby("ticker").agg(n=("n_pos","sum"),d=("n_neg","sum"),sig=("sign","sum"))
    sc=g["n"]-g["d"]+g["sig"]*0.5
    return (sc-sc.mean())/(sc.std() if sc.std()>0 else 1.0)

rebal=list(ml.truncate("2025-09-30",px.index[-1]).index)
def run(kind, window=60, lam=1.2, cost_bps=5):
    weights=pd.DataFrame(0.0,index=px.index,columns=cols); prev=pd.Series(0.0,index=cols); costs=[]
    for i,d in enumerate(rebal):
        if d not in mom.index: continue
        m=mom.loc[d].dropna()
        if m.empty: continue
        if kind=="mom": score=m.rank(pct=True)
        else: score=m.rank(pct=True)+lam*wscore(d,window).reindex(m.index).fillna(0.0)
        sel=score.sort_values(ascending=False).index[:10]
        if len(sel)==0: continue
        w=pd.Series(0.0,index=cols); w[sel]=1/len(sel)
        end=rebal[i+1] if i+1<len(rebal) else px.index[-1]
        days=px.index[(px.index>d)&(px.index<=end)]
        if len(days):
            weights.loc[days,:]=w.reindex(cols).values
            costs.append((days[0],(w-prev).abs().sum()/2.0*cost_bps/10000)); prev=w
    ret=(weights*daily_ret).sum(axis=1)
    for day,c in costs: ret.loc[day]-=c
    ret=ret.fillna(0.0).clip(lower=-0.5); nav=(1+ret).cumprod()*20000
    ix=nav.index>=pd.Timestamp("2025-10-01"); nav2=nav[ix]; rr=ret[ix]
    ann=(nav2.iloc[-1]/nav2.iloc[0])**(252/len(rr))-1; vol=rr.std()*np.sqrt(252)
    sharpe=ann/vol if vol>0 else np.nan; maxdd=(nav2/nav2.cummax()-1).min()
    rets/100.0 if False else None
    return {"final":float(nav2.iloc[-1]),"ann":ann,"vol":vol,"sharpe":sharpe,"maxdd":float(maxdd),"n_rebal":len(costs)}

r1=run("mom"); r2=run("ima")
print("== 最终配方验证（2025-10 ~ 2026-08，5bp，月度10只等权）==")
print(f"纯动量          : final=${r1['final']:,.0f}  年化 {r1['ann']*100:.1f}%  波动 {r1['vol']*100:.1f}%  夏普 {r1['sharpe']:.2f}  最大回撤 {r1['maxdd']*100:.1f}%")
print(f"动量+IMA(60/1.2): final=${r2['final']:,.0f}  年化 {r2['ann']*100:.1f}%  波动 {r2['vol']*100:.1f}%  夏普 {r2['sharpe']:.2f}  最大回撤 {r2['maxdd']*100:.1f}%")
