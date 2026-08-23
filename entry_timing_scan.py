# -*- coding: utf-8 -*-
"""对比：月底信号后：(1) 次日一次买满 (2) 5日每天一批 (3) 分批+只在回踩买 dip
统一数据源：F:\\even-codex\\us-stock-data\\prices.csv
策略：月频 加速 top10 等权（收益优先，不加 vol25）
"""
import os, sys
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np, pandas as pd
DATA = Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data")
OUT = Path(__file__).resolve().parent / "backtest_output"
DAYS=252; START=20000.0; COST_BPS=10; TOP=10; T=5; WINDOW=10; PULL=-0.015; SPIKE=0.03

px = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
px = px.loc[:, px.count() >= 2400]
idx = px.index; cols = list(px.columns)
dr = px.pct_change().fillna(0.0)

# 月频加速打分
m = px.resample("ME").last()
sc = 0.5*(m.shift(1)/m.shift(7) - 1.0) + 0.5*m.pct_change(1)

# 信号月末 -> 下一个交易日(再平衡日)
me = pd.Series(np.arange(len(idx)), index=idx).groupby(idx.to_period("M")).last().astype(int)
sigs = [int(x) for x in me.tolist() if x > 147 and x+1 < len(idx)]
rbs = []
for s in sigs:
    n = int(np.searchsorted(idx, idx[s], side="right"))
    if n < len(idx): rbs.append((s, n))

def target_at(s):
    pp = pd.Timestamp(idx[s]).to_period("M")
    row = sc.loc[sc.index.to_period("M") == pp]
    if len(row) == 0: return []
    return list(row.iloc[0].dropna().sort_values(ascending=False).index[:TOP])

def run(mode):
    w = {c: 0.0 for c in cols}; cash = 1.0
    rets = []; dts = []; invest = []
    for i, (s, r0) in enumerate(rbs):
        r_next = rbs[i+1][1] if i+1 < len(rbs) else len(idx)
        targ = target_at(s)
        tgt_frac = 1.0/len(targ) if targ else 0.0
        per_t = tgt_frac/T
        filled = {c: 0 for c in targ}
        seg_days = list(range(r0, min(r_next, r0+WINDOW)))
        for k, t in enumerate(range(r0, r_next)):
            # 当日收益（按持仓在当日开盘时的仓位，收盘成交买卖从次日起价）
            rt = sum(w[c]*float(dr.iloc[t][c]) for c in w if w[c] > 1e-12)
            rets.append(rt); dts.append(idx[t]); invest.append(sum(v for v in w.values()) + cash)
            if t == r0:
                sv = sum(v for v in w.values()); cash += sv; cash -= sv*COST_BPS/10000.0
                for c in w: w[c] = 0.0
            buys = []
            if mode == "baseline" and t == r0:
                buys = list(targ)
            elif mode == "spread":
                buys = [c for c in targ if filled[c] < T]
            elif mode == "dip":
                deadline = len(seg_days) - k <= 2
                for c in targ:
                    if filled[c] < T:
                        r1 = float(dr.iloc[t][c])
                        if r1 <= PULL or (deadline and r1 < SPIKE):
                            buys.append(c)
            for cn in buys:
                if filled.get(cn, 0) >= T: continue
                add = tgt_frac if mode == "baseline" else per_t
                cash -= add; cash -= add*COST_BPS/10000.0
                w[cn] += add; filled[cn] += 1
    r = pd.Series(rets, index=pd.DatetimeIndex(dts))
    invested = pd.Series(invest, index=pd.DatetimeIndex(dts))
    return r, invested

def metrics(r, inv):
    nav = (1+r).cumprod()*START
    ann = (nav.iloc[-1]/START)**(DAYS/len(r))-1
    vol = r.std(ddof=1)*np.sqrt(DAYS); sh = ann/vol if vol > 0 else np.nan
    dd = (nav/nav.cummax()-1.0).min(); cal = ann/abs(dd) if dd < 0 else np.nan
    return dict(ann=ann, vol=vol, sh=sh, mdd=dd, calmar=cal, active=inv.mean(), final=nav.iloc[-1])

rows = []
for mode in ["baseline", "spread", "dip"]:
    r, inv = run(mode)
    f = metrics(r, inv)
    o = metrics(r[r.index >= "2022-01-01"], inv[inv.index >= "2022-01-01"])
    rows.append(dict(exec=mode, full_ann=round(f["ann"],4), full_vol=round(f["vol"],4), full_sh=round(f["sh"],3),
                     full_mdd=round(f["mdd"],4), full_calmar=round(f["calmar"],3), full_active=round(f["active"],4),
                     full_end=round(f["final"],0),
                     oos_ann=round(o["ann"],4), oos_sh=round(o["sh"],3), oos_mdd=round(o["mdd"],4), oos_active=round(o["active"],4)))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(OUT/"entry_timing_scan.csv", index=False, encoding="utf-8-sig")
print("saved:", OUT/"entry_timing_scan.csv")
