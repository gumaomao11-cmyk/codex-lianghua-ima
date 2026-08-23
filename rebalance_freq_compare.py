# -*- coding: utf-8 -*-
"""调仓节奏对比：月频 vs 周频 vs 日频（同一套 6m-skip1 动量 top10）
信号在 rebalance 日收盘打分，次个交易日切到新 top10，加换手成本。
"""
import os
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data")
OUT = Path(__file__).resolve().parent / "backtest_output"
DAYS = 252; START = 20000.0

px = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
px = px.loc[:, px.count() >= 2400]
dr = px.pct_change().fillna(0.0)
idx = px.index
cols = list(px.columns)
idpos = {d:i for i,d in enumerate(idx)}

def mom_score(d):
    """6m-skip1 日线近似动量：过去6个月(约126交易日)、跳过最近1个月(约21日)"""
    if d - 147 < 0:
        return None
    a = px.iloc[d-21]; b = px.iloc[d-147]
    return (a/b - 1.0).replace([np.inf, -np.inf], np.nan)

def group_last_days(freq):
    per = idx.to_period("M" if freq == "monthly" else "W")
    s = pd.Series(idx, index=per)
    last = s.groupby(level=0).last()
    return [int(idpos[d]) for d in last.tolist() if d in idpos]

def rebal_days(freq):
    if freq == "monthly":
        return [d for d in group_last_days("monthly") if d > 147 and d+1 < len(idx)]
    if freq == "weekly":
        return [d for d in group_last_days("weekly") if d > 147 and d+1 < len(idx)]
    return list(range(147, len(idx)-1))

def run(freq, top=10, cost_bps=10):
    rb = rebal_days(freq)
    prev_w = pd.Series(0.0, index=cols)
    ret = pd.Series(0.0, index=idx)
    cost_line = pd.Series(0.0, index=idx)
    turnovers = []
    for k, rdi in enumerate(rb):
        if rdi+1 >= len(idx):
            break
        sc = mom_score(rdi)
        if sc is None or sc.dropna().empty:
            continue
        ta = list(sc.sort_values(ascending=False).index[:top])
        w = 1.0/len(ta)
        new_w = pd.Series(0.0, index=cols); new_w[ta] = w
        to = (new_w - prev_w).abs().sum()/2.0
        turnovers.append(to)
        hold_start = rdi + 1
        seg_end = rb[k+1] if k+1 < len(rb) else len(idx)-1   # 下个信号日为止（含，旧仓延持到它收盘）
        if seg_end <= hold_start:
            seg_end = hold_start + 1
        cost_line.iloc[hold_start] += to * cost_bps / 10000.0
        if len([c for c in ta if c in cols]) == 0:
            prev_w = new_w.copy(); continue
        wv = (dr.iloc[hold_start:seg_end+1, [cols.index(c) for c in ta]] * w).sum(axis=1)
        ret.iloc[hold_start:seg_end+1] += wv
        prev_w = new_w.copy()
    net = (ret - cost_line).clip(lower=-0.5)
    ann_turn = float(np.mean(turnovers)) if turnovers else 0.0
    return net, ann_turn

def metrics(ret):
    nav = (1+ret).cumprod()*START
    ann = (nav.iloc[-1]/START)**(DAYS/len(nav))-1 if len(nav) else 0
    vol = ret.std(ddof=1)*np.sqrt(DAYS) if len(ret)>1 else 0
    sh = ann/vol if vol>0 else np.nan
    dd = (nav/nav.cummax()-1).min()
    return dict(ann=ann, vol=vol, sharpe=sh, mdd=float(dd), final=float(nav.iloc[-1]))

oos = idx >= pd.Timestamp("2022-01-01")
rows=[]
for freq in ["monthly","weekly","daily"]:
    for bps in [10, 20]:
        r, turn = run(freq, cost_bps=bps)
        f=metrics(r); o=metrics(r[oos])
        rows.append(dict(freq=freq, bps=bps, full_sh=round(f["sharpe"],3), full_ann=round(f["ann"],4),
                         full_mdd=round(f["mdd"],4), full_end=round(f["final"],0),
                         oos_sh=round(o["sharpe"],3), oos_ann=round(o["ann"],4), oos_mdd=round(o["mdd"],4),
                         turnover=round(turn,4)))
df=pd.DataFrame(rows)
out_csv = OUT/"rebalance_freq_compare.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(df.to_string(index=False))
print("saved:", out_csv)
