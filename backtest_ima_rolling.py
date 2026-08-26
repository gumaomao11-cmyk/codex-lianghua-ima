# -*- coding: utf-8 -*-
"""IMA factor robustness: monthly-return based walk-forward + IS/OOS param selection."""
import sys, json, itertools
from pathlib import Path
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA=Path(r"F:\even-codex\us-stock-data"); OUT=Path(r"F:\even-codex\lianghua+IMA\backtest_output")
px=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce")
recent=px.loc[px.index>=pd.Timestamp("2025-01-01")]; px=px.loc[:,recent.notna().sum()>=150]
ml=px.resample("ME").last(); mom=ml.shift(1)/ml.shift(7)-1.0
mom=mom.replace([np.inf,-np.inf],np.nan)
fac=pd.read_csv(OUT/"kb_abstract_factors.csv",encoding="utf-8-sig"); fac["pdf_date"]=pd.to_datetime(fac["pdf_date"],errors="coerce"); fac=fac.dropna(subset=["pdf_date"])
cols=list(px.columns); daily_ret=px.pct_change().fillna(0.0)
rebal=list(ml.truncate("2025-09-30",px.index[-1]).index)
print("rebalances",len(rebal))

def ima_score_for(d,window=90):
    s=fac[fac.pdf_date<d]; s=s[s.pdf_date>=d-pd.Timedelta(days=window)]
    if s.empty: return pd.Series(dtype=float)
    g=s.groupby("ticker").agg(n=("n_pos","sum"),d=("n_neg","sum"),sig=("sign","sum"))
    sc=g["n"]-g["d"]+g["sig"]*0.5
    return (sc-sc.mean())/(sc.std() if sc.std()>0 else 1.)

def monthly_strategy_returns(mode,window=90,lam=0.7,cost_bps=5):
    rets={}
    for i,d in enumerate(rebal):
        if d not in mom.index: continue
        m=mom.loc[d].dropna()
        if m.empty: continue
        if mode=="momentum":
            score=m.rank(pct=True)
        elif mode=="momentum_ima":
            z=ima_score_for(d,window).reindex(m.index); score=m.rank(pct=True)+lam*z.fillna(0.0)
        elif mode=="ima_only":
            z=ima_score_for(d,window).reindex(m.index)
            if z.notna().sum()==0: continue
            score=z.fillna(-1e9)
        else: raise ValueError(mode)
        sel=score.sort_values(ascending=False).index[:10].tolist()
        if not sel: continue
        w=pd.Series(0.0,index=cols); w[sel]=1/len(sel)
        end=rebal[i+1] if i+1<len(rebal) else px.index[-1]
        days=px.index[(px.index>d)&(px.index<=end)]
        if len(days)==0: continue
        r=(w.reindex(cols).values * daily_ret.loc[days,:].values).sum(axis=1)-((w-pd.Series(0.0,index=cols)).abs().sum()/2.)*cost_bps/10000.
        rets[d]=float(r.sum())
    return pd.Series(rets)

def summary(series):
    if series is None or len(series)==0: return {"n":0,"monthly_ret":np.nan,"vol":np.nan,"sharpe":np.nan}
    r=series.dropna(); n=len(r)
    if n==0: return {"n":0,"monthly_ret":np.nan,"vol":np.nan,"sharpe":np.nan}
    mr=float(r.mean()); vol=float(r.std(ddof=0)); return {"n":n,"monthly_ret":mr*100,"vol":vol*100,"sharpe":float(mr/vol) if vol>0 else np.nan}

print("\n--- default params robustness (window=90 lam=0.7) ---")
rows=[]
for mode in ["momentum","momentum_ima","ima_only"]:
    m=monthly_strategy_returns(mode)
    s=summary(m)
    rows.append({"mode":mode,**s})
    print(mode, s)
pd.DataFrame(rows).to_csv(OUT/"ima_rolling_default_summary.csv",index=False,encoding="utf-8-sig")

print("\n--- rolling / out-of-sample splits ---")
# split at 5th rebalance (IS first 5, OOS last 6)
for mode in ["momentum","momentum_ima","ima_only"]:
    m=monthly_strategy_returns(mode)
    is_=summary(m.iloc[:5]); oos=summary(m.iloc[5:])
    print(f"{mode}: IS n={is_['n']} mean={is_['monthly_ret']:.2f}% sharpe={is_['sharpe']:.2f} | OOS n={oos['n']} mean={oos['monthly_ret']:.2f}% sharpe={oos['sharpe']:.2f}")

print("\n--- param grid IS-select then OOS ---")
mom_ret=monthly_strategy_returns("momentum"); mom_oos=summary(mom_ret.iloc[5:])
best=None; oos_records=[]
for window in [30,60,90]:
    for lam in [0.3,0.7,1.2]:
        m=monthly_strategy_returns("momentum_ima",window=window,lam=lam)
        is_=summary(m.iloc[:5]); oos_=summary(m.iloc[5:])
        if best is None or (is_["sharpe"] is not None and not np.isnan(is_["sharpe"]) and best[1] < is_["sharpe"]):
            best=((window,lam), is_["sharpe"], oos_)
        oos_records.append({"window":window,"lam":lam,"is_sharpe":is_["sharpe"],"oos_sharpe":oos_["sharpe"],"oos_monthly_ret_pct":oos_["monthly_ret"],"oos_vol_pct":oos_["vol"]})
pd.DataFrame(oos_records).to_csv(OUT/"ima_param_oos_grid.csv",index=False,encoding="utf-8-sig")
print("IS best param", best[0], "IS sharpe", best[1], "OOS", best[2])
print("Baseline momentum OOS", mom_oos)
print("saved ima_rolling_default_summary.csv & ima_param_oos_grid.csv")
