# -*- coding: utf-8 -*-
"""Baseline momentum vs momentum + LLM bearish negative filter."""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT=Path(r"F:\even-codex\lianghua+IMA\backtest_output"); DATA=Path(r"F:\even-codex\us-stock-data")
px=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce")
recent=px.loc[px.index>=pd.Timestamp("2025-01-01")]; px=px.loc[:,recent.notna().sum()>=150]
ml=px.resample("ME").last(); mom=ml.shift(1)/ml.shift(7)-1.0
mom=mom.replace([np.inf,-np.inf],np.nan)
cols=list(px.columns); daily_ret=px.pct_change().fillna(0.0); rebal=list(ml.truncate("2025-09-30",px.index[-1]).index)

df=pd.read_csv(OUT/"kb_llm_sentiment.csv",encoding="utf-8-sig"); df["pdf_date"]=pd.to_datetime(df["pdf_date"],errors="coerce"); df=df.dropna(subset=["pdf_date"])

def bear_since(d, window, strength_thr):
    s=df[df.pdf_date<d]
    s=s[s.pdf_date>=d-pd.Timedelta(days=window)]
    s=s[(s["direction"]<0)&(s["strength"]>=strength_thr)]
    return set(s["ticker"])

def run(mode, window=60, strength_thr=0.5, cost_bps=5):
    weights=pd.DataFrame(0.0,index=px.index,columns=cols); prev=pd.Series(0.0,index=cols); costs=[]
    for i,d in enumerate(rebal):
        if d not in mom.index: continue
        m=mom.loc[d].dropna()
        if m.empty: continue
        if mode=="momentum":
            score=m.rank(pct=True)
        else:
            excl=bear_since(d, window, strength_thr)
            m2=m.drop(index=[t for t in excl if t in m.index])
            if len(m2)<10:
                # fallback back to full momentum if too few left
                score=m.rank(pct=True)
            else:
                score=m2.rank(pct=True)
        sel=score.sort_values(ascending=False).index[:10].tolist()
        if not sel: continue
        w=pd.Series(0.0,index=cols); w[sel]=1/len(sel)
        end=rebal[i+1] if i+1<len(rebal) else px.index[-1]
        days=px.index[(px.index>d)&(px.index<=end)]
        if len(days):
            weights.loc[days,:]=w.reindex(cols).values
            costs.append((days[0],(w-prev).abs().sum()/2.0*cost_bps/10000)); prev=w
    ret=(weights*daily_ret).sum(axis=1)
    for day,c in costs: ret.loc[day]-=c
    ret=ret.fillna(0.0).clip(lower=-0.5)
    nav=(1+ret).cumprod()*20000
    ix=nav.index>=pd.Timestamp("2025-10-01"); nav2=nav[ix]; rr=ret[ix]
    ann=(nav2.iloc[-1]/nav2.iloc[0])**(252/len(rr))-1; vol=rr.std()*np.sqrt(252)
    sharpe=ann/vol if vol>0 else np.nan; dd=(nav2/nav2.cummax()-1).min()
    return {"mode":f"{mode}|win{window}|thr{strength_thr}","final":float(nav2.iloc[-1]),"ann":ann,"sharpe":sharpe,"maxdd":float(dd),"n_rebal":len(costs)}

rows=[]
for mode in ["momentum"]:
    rows.append(run(mode))
for window in [15,30,60]:
    for thr in [0.4,0.5,0.7]:
        rows.append(run("momentum_llm_filter", window=window, strength_thr=thr))
res=pd.DataFrame(rows); res.to_csv(OUT/"momentum_llm_filter_grid.csv",index=False,encoding="utf-8-sig")
for _,r in res.iterrows():
    print(f"{r['mode']}: final=${r['final']:,.0f} ann={r['ann']*100:.1f}% sharpe={r['sharpe']:.2f} maxdd={r['maxdd']*100:.1f}%")
print("best by sharpe:")
print(res.sort_values("sharpe",ascending=False).head(6).to_string(index=False))
