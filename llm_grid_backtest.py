# -*- coding: utf-8 -*-
"""Quick grid: find if momentum+LLM sentiment helps under cleaner configs."""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT=Path(r"F:\even-codex\lianghua+IMA\backtest_output"); DATA=Path(r"F:\even-codex\us-stock-data")
px=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce")
recent=px.loc[px.index>=pd.Timestamp("2025-01-01")]; px=px.loc[:,recent.notna().sum()>=150]
df=pd.read_csv(OUT/"kb_llm_sentiment.csv",encoding="utf-8-sig"); df["pdf_date"]=pd.to_datetime(df["pdf_date"],errors="coerce"); df=df.dropna(subset=["pdf_date"])
ml=px.resample("ME").last(); mom=ml.shift(1)/ml.shift(7)-1; mom=mom.replace([np.inf,-np.inf],np.nan)
cols=list(px.columns); daily_ret=px.pct_change().fillna(0.0); rebal=list(ml.truncate("2025-09-30",px.index[-1]).index)

def score_for(d, window, strong):
    s=df[df.pdf_date<d]; s=s[s.pdf_date>=d-pd.Timedelta(days=window)]
    if strong:
        s=s[s["direction"]!=0]
        s=s[s["strength"]>=0.5]
    if s.empty: return pd.Series(dtype=float)
    s=s.copy(); s["score"]=s["direction"]*s["strength"]
    return s.groupby("ticker")["score"].sum()

def run(window, strong, lam, cost_bps=5):
    weights=pd.DataFrame(0.0,index=px.index,columns=cols); prev=pd.Series(0.0,index=cols); costs=[]
    for i,d in enumerate(rebal):
        if d not in mom.index: continue
        m=mom.loc[d].dropna()
        if m.empty: continue
        z=score_for(d,window,strong).reindex(m.index)
        if z.notna().sum()>0: z=(z-z.mean())/(z.std() if z.std()>0 else 1)
        else: z=z*0
        score=m.rank(pct=True)+lam*z.fillna(0.0)
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
    sharpe=ann/vol if vol>0 else np.nan; dd=(nav2/nav2.cummax()-1).min()
    return {"window":window,"strong":strong,"lam":lam,"final":float(nav2.iloc[-1]),"sharpe":sharpe,"maxdd":float(dd),"ann":ann}

rows=[]
for w in [15,30,60]:
    for strong in [True,False]:
        for lam in [0.7,1.0,1.5]:
            r=run(w,strong,lam); rows.append(r)
            print(f"win={w} strong={int(strong)} lam={lam}: final=${r['final']:,.0f} sharpe={r['sharpe']:.2f} maxdd={r['maxdd']*100:.1f}%")
pd.DataFrame(rows).to_csv(OUT/"llm_sentiment_grid.csv",index=False,encoding="utf-8-sig")
print("best by sharpe:", pd.DataFrame(rows).sort_values("sharpe",ascending=False).head(5).to_string(index=False))
