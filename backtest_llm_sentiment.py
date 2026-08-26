# -*- coding: utf-8 -*-
"""Compare LLM sentiment factor vs old word-count factor (forward IC) and run momentum+LLM backtest."""
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT=Path(r"F:\even-codex\lianghua+IMA\backtest_output"); DATA=Path(r"F:\even-codex\us-stock-data")

# prices
px=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce")
recent=px.loc[px.index>=pd.Timestamp("2025-01-01")]; px=px.loc[:,recent.notna().sum()>=150]
fwd=px.shift(-20)/px - 1.0

def ic_analysis(facmat, fwd):
    dates=[]; ics=[]
    for d in facmat.index:
        if d not in fwd.index: continue
        s=facmat.loc[d]; f=fwd.loc[d].reindex(s.index)
        m=pd.concat([s,f],axis=1).dropna()
        if len(m)<5: continue
        ic=m.iloc[:,0].rank().corr(m.iloc[:,1].rank())
        if math.isnan(ic): continue
        dates.append(d); ics.append(ic)
    ics=pd.Series(ics, index=dates)
    if len(ics)==0: return {"version":"","n":0,"mean_ic":float("nan"),"icir":float("nan"),"hit":float("nan")}
    return {"n":len(ics),"mean_ic":ics.mean(),"std_ic":ics.std(ddof=0),"icir":ics.mean()/ics.std(ddof=0) if ics.std()>0 else float("nan"),"hit":(ics>0).mean()}

def daily_factor_from_signal(gp, version):
    d0=gp.pivot_table(index="pdf_date",columns="ticker",values="score",aggfunc="sum").sort_index()
    if version=="surprise":
        base=d0.rolling(60,min_periods=5).mean(); return (d0-base).where(d0.notna())
    if version=="decay":
        out=d0.copy()*0; hl=10.0
        import math as _m
        for i,d in enumerate(d0.index):
            cur=d0.iloc[i].fillna(0)
            if i>0: cur = cur + out.iloc[i-1].fillna(0)*_m.exp(-1/hl)
            out.iloc[i]=cur
        return out
    return d0

def build_llm_gp():
    df=pd.read_csv(OUT/"kb_llm_sentiment.csv",encoding="utf-8-sig")
    df["pdf_date"]=pd.to_datetime(df["pdf_date"],errors="coerce"); df=df.dropna(subset=["pdf_date"])
    df["score"]=df["direction"]*df["strength"]
    return df[["pdf_date","ticker","score"]]

def build_old_gp():
    df=pd.read_csv(OUT/"kb_abstract_factors.csv",encoding="utf-8-sig")
    df=df[df["source_folder"]=="美国科技日报(270)"]
    df["pdf_date"]=pd.to_datetime(df["pdf_date"],errors="coerce"); df=df.dropna(subset=["pdf_date"])
    df["score"]=df["n_pos"]-df["n_neg"]+0.5*df["sign"]
    return df[["pdf_date","ticker","score"]]

print("===== forward 20d Rank IC =====")
llm_gp=build_llm_gp(); old_gp=build_old_gp()
ic_rows=[]
for name, gp in [("LLM情绪", llm_gp), ("旧词频", old_gp)]:
    for version in ["raw","surprise","decay"]:
        fm=daily_factor_from_signal(gp, version)
        m=ic_analysis(fm, fwd)
        m["version"]=version; m["name"]=name
        ic_rows.append(m)
        print(f"{name} / {version}: n={m['n']} mean_ic={m.get('mean_ic',float('nan')):.3f} icir={m.get('icir',float('nan')):.2f} hit={m.get('hit',float('nan')):.2f}")
pd.DataFrame(ic_rows).to_csv(OUT/"llm_vs_word_ic.csv",index=False,encoding="utf-8-sig")

# monthly backtest: momentum + LLM factor
ml=px.resample("ME").last(); mom=ml.shift(1)/ml.shift(7)-1
mom=mom.replace([np.inf,-np.inf],np.nan); cols=list(px.columns); daily_ret=px.pct_change().fillna(0.0)
rebal=list(ml.truncate("2025-09-30",px.index[-1]).index)

def llm_score_for(d, window=90):
    s=llm_gp[llm_gp.pdf_date<d]; s=s[s.pdf_date>=d-pd.Timedelta(days=window)]
    if s.empty: return pd.Series(dtype=float)
    g=s.groupby("ticker")["score"].sum()
    return g

def run_bt(mode, lam=0.7, window=90, cost_bps=5):
    weights=pd.DataFrame(0.0,index=px.index,columns=cols); prev=pd.Series(0.0,index=cols); costs=[]
    for i,d in enumerate(rebal):
        if d not in mom.index: continue
        m=mom.loc[d].dropna()
        if m.empty: continue
        if mode=="momentum":
            score=m.rank(pct=True)
        else:
            z=llm_score_for(d,window).reindex(m.index)
            # normalize
            if z.notna().sum()>0:
                z=(z-z.mean())/(z.std() if z.std()>0 else 1)
            else: z=z*0
            score=m.rank(pct=True)+lam*z.fillna(0.0)
        sel=score.sort_values(ascending=False).index[:10]
        if len(sel)==0: continue
        w=pd.Series(0.0,index=cols); w[sel]=1/len(sel)
        end=rebal[i+1] if i+1<len(rebal) else px.index[-1]
        days=px.index[(px.index>d)&(px.index<=end)]
        if len(days):
            weights.loc[days,:]=w.reindex(cols).values
            costs.append((days[0], (w-prev).abs().sum()/2.0*cost_bps/10000.0)); prev=w
    ret=(weights*daily_ret).sum(axis=1)
    for day,c in costs: ret.loc[day]-=c
    ret=ret.fillna(0.0).clip(lower=-0.5)
    nav=(1+ret).cumprod()*20000
    ix=nav.index>=pd.Timestamp("2025-10-01"); nav2=nav[ix]; rr=ret[ix]
    ann=(nav2.iloc[-1]/nav2.iloc[0])**(252/len(rr))-1
    vol=rr.std()*np.sqrt(252); sharpe=ann/vol if vol>0 else np.nan
    dd=(nav2/nav2.cummax()-1).min()
    return {"mode":mode,"final_nav":float(nav2.iloc[-1]),"ann_ret":ann,"sharpe":sharpe,"max_dd":dd,"n_rebal":len(costs)}

bt_rows=[]
for mode in ["momentum","momentum_llm"]:
    r=run_bt(mode); bt_rows.append(r); print(f"BT {mode}: final=${r['final_nav']:,.0f} ann={r['ann_ret']*100:.1f}% sharpe={r['sharpe']:.2f} maxdd={r['max_dd']*100:.1f}%")
pd.DataFrame(bt_rows).to_csv(OUT/"llm_momentum_backtest.csv",index=False,encoding="utf-8-sig")
print("saved llm_vs_word_ic.csv and llm_momentum_backtest.csv")
