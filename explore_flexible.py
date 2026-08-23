# -*- coding: utf-8 -*-
"""探索：月频动量策略的“更灵活”变体，对比能不能提高收益/夏普
base(等权) / rank加权 / 前5重仓 / 动量+近期加速 / 动量门槛过滤(空仓一部分)
"""
import os
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data")
OUT = Path(__file__).resolve().parent / "backtest_output"
DAYS=252; START=20000.0; COST=10; TOP=10

px = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
px = px.loc[:, px.count() >= 2400]
idx=px.index; cols=list(px.columns); idpos={d:i for i,d in enumerate(idx)}; dr=px.pct_change().fillna(0.0)
mom6 = px.shift(21).div(px.shift(147)) - 1.0   # 6m-skip1 日近似
acc  = px.div(px.shift(21)) - 1.0                # 近1月加速

def month_last_days():
    s = pd.Series(idx, index=idx.to_period("M"))
    last = s.groupby(level=0).last()
    return [int(idpos[d]) for d in last.tolist() if d in idpos]

def pick_weight(variant, d):
    if variant == "accel":
        sc = (pd.DataFrame({"a": mom6.iloc[d], "b": acc.iloc[d]}).mean(axis=1))
    else:
        sc = mom6.iloc[d]
    sc = sc.sort_values(ascending=False)
    names = sc.index[:TOP].tolist()
    if variant == "mom20":
        names = sc.index[sc > 0.20].tolist()[:TOP]   # 动量不够就空仓
    if variant == "rank_w":
        w = np.arange(len(names),0,-1.0); w = w/w.sum()
        return {t:vv for t,vv in zip(names,w)}
    if variant == "topheavy":
        wd = {}
        for i,t in enumerate(names):
            wd[t] = 0.18 if i < 5 else 0.02
        return wd
    return {t: 1.0/len(names) for t in names}

def run(variant):
    rb = [d for d in month_last_days() if d>147 and d+1<len(idx)]
    prev_w = pd.Series(0.0, index=cols); ret = pd.Series(0.0, index=idx); cost_line = pd.Series(0.0, index=idx)
    for k,rdi in enumerate(rb):
        wd = pick_weight(variant, rdi)
        new_w = pd.Series(0.0, index=cols)
        if wd:
            new_w = pd.Series({t:vv for t,vv in wd.items() if t in cols})
        hold_start = rdi+1; seg_end = rb[k+1] if k+1<len(rb) else len(idx)-1
        if seg_end <= hold_start: seg_end = hold_start+1
        to = (new_w-prev_w).abs().sum()/2.0
        cost_line.iloc[hold_start] += to*COST/10000.0
        li = [cols.index(c) for c in new_w.index if new_w[c]!=0]
        if li:
            ret.iloc[hold_start:seg_end+1] += (dr.iloc[hold_start:seg_end+1,li]*new_w.values[new_w!=0]).sum(axis=1)
        prev_w = new_w.copy()
    return (ret-cost_line).clip(lower=-0.5)

def metrics(r):
    nav=(1+r).cumprod()*START; ann=(nav.iloc[-1]/START)**(DAYS/len(nav))-1
    vol=r.std(ddof=1)*np.sqrt(DAYS); sh=ann/vol if vol>0 else np.nan
    mdd=(nav/nav.cummax()-1).min()
    return dict(ann=ann,vol=vol,sharpe=[round(float(sh),3) if isinstance(sh,float) else sh][0],mdd=float(mdd),final=float(nav.iloc[-1]))

oos = idx >= pd.Timestamp("2022-01-01")
names = dict(base="base: 等权top10", rank_w="rank_w: 按排名线性加权", topheavy="topheavy: 前5各18%/后5各2%",
             accel="accel: 动量+近1月加速", mom20="mom20: 动量>0.2才持仓(常空仓)")
rows=[]
for v,label in names.items():
    r=run(v); f=metrics(r); o=metrics(r[oos])
    rows.append(dict(变体=label, full_sh=f["sharpe"], full_ann=round(f["ann"],4), full_mdd=round(f["mdd"],4), full_end=round(f["final"],0),
                     oos_sh=o["sharpe"], oos_ann=round(o["ann"],4), oos_mdd=round(o["mdd"],4), oos_end=round(o["final"],0)))
df=pd.DataFrame(rows).sort_values("full_sh", ascending=False)
out=OUT/"flexible_variants_scan.csv"; df.to_csv(out,index=False,encoding="utf-8-sig")
print(df.to_string(index=False))
print("saved:", out)
