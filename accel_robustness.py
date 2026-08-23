# -*- coding: utf-8 -*-
"""稳健性检查：动量+近期加速 的组合参数扫描"""
import os
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data")
DAYS=252; START=20000.0; COST=10; TOP=10
px = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
px = px.loc[:, px.count() >= 2400]
idx=px.index; cols=list(px.columns); idpos={d:i for i,d in enumerate(idx)}; dr=px.pct_change().fillna(0.0)
mom6 = px.shift(21).div(px.shift(147)) - 1.0

def month_last_days():
    s = pd.Series(idx, index=idx.to_period("M"))
    last = s.groupby(level=0).last()
    return [int(idpos[d]) for d in last.tolist() if d in idpos]

def run(acc_lb, w_mom):
    rb=[d for d in month_last_days() if d>147 and d+1<len(idx)]
    acc = px.div(px.shift(acc_lb)) - 1.0
    prev_w=pd.Series(0.0,index=cols); ret=pd.Series(0.0,index=idx); cost_line=pd.Series(0.0,index=idx)
    for k,rdi in enumerate(rb):
        sc = (w_mom*mom6.iloc[rdi] + (1-w_mom)*acc.iloc[rdi]).sort_values(ascending=False).head(TOP)
        nw=pd.Series({t:1.0/len(sc) for t in sc.index}, index=sc.index)
        new_w=pd.Series(0.0,index=cols); new_w[nw.index]=nw.values
        hs=rdi+1; se=rb[k+1] if k+1<len(rb) else len(idx)-1
        if se<=hs: se=hs+1
        cost_line.iloc[hs] += (new_w-prev_w).abs().sum()/2.0*COST/10000.0
        li=[cols.index(c) for c in new_w.index if new_w[c]!=0]
        if li:
            ret.iloc[hs:se+1] += (dr.iloc[hs:se+1,li]*new_w.values[new_w!=0]).sum(axis=1)
        prev_w=new_w.copy()
    return (ret-cost_line).clip(lower=-0.5)

def metrics(r):
    nav=(1+r).cumprod()*START; ann=(nav.iloc[-1]/START)**(DAYS/len(nav))-1
    vol=r.std(ddof=1)*np.sqrt(DAYS); sh=ann/vol if vol>0 else np.nan
    mdd=(nav/nav.cummax()-1).min()
    return sh, ann, mdd, nav.iloc[-1]

oos=idx>=pd.Timestamp("2022-01-01")
rows=[]
for lb in (10,21,42):
    for wm in (0.5,0.6,0.7,0.8,0.9):
        r=run(lb,wm); fs,fa,fm,fe=metrics(r); os,oa,om,oe=metrics(r[oos])
        rows.append(dict(acc_lookback=lb,w_mom=wm,full_sh=round(fs,3),full_ann=round(fa,4),full_mdd=round(fm,4),full_end=round(fe,0),oos_sh=round(os,3),oos_ann=round(oa,4),oos_mdd=round(om,4),oos_end=round(oe,0)))
df=pd.DataFrame(rows).sort_values("full_sh",ascending=False)
print(df.to_string(index=False))
df.to_csv(Path(r"F:\even-codex\lianghua2\backtest_output\accel_robustness_scan.csv"), index=False, encoding="utf-8-sig")
