# -*- coding: utf-8 -*-
"""基准 vs 加止盈止损 回测对比 (6m-skip1 top10, 月度调仓, 日线收盘价)
分段完全对齐项目原 run_fast（slot 映射月末标签）。止盈/止损：收盘触发、次日收盘成交。
"""
import os
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data")
OUT = Path(__file__).resolve().parent / "backtest_output"
DAYS = 252
START = 20000.0

px = pd.read_csv(DATA / "prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
px = px.loc[:, px.count() >= 2400]

def ml(x): return x.resample("ME").last()
def mom(px, p, k):
    m = ml(px); return m.shift(k) / m.shift(p + k) - 1

def backtest(px, scores, top=10, cost_bps=10, tp=None, sl=None):
    cols = list(px.columns); idx = px.index
    dr = px.pct_change().fillna(0.0)
    port = pd.Series(0.0, index=idx)
    cost_line = pd.Series(0.0, index=idx)
    me_arr = np.array(pd.DatetimeIndex(scores.index))
    day_arr = np.array(idx)
    slot0 = np.searchsorted(me_arr, day_arr, side="right") - 1
    exact = (slot0 >= 0) & (day_arr == me_arr[np.clip(slot0, 0, len(me_arr)-1)])
    slot = np.clip(slot0 - exact.astype(int), 0, len(me_arr)-1)
    prev_w = pd.Series(0.0, index=cols)
    for s in range(1, len(me_arr)):
        seg_idx = np.where(slot == s)[0]
        if len(seg_idx) == 0:
            continue
        start_i = int(seg_idx[0]); end_i = int(seg_idx[-1])
        sig = pd.Timestamp(me_arr[s])
        sc = scores.loc[sig].dropna()
        if len(sc) == 0:
            prev_w[:] = 0.0; continue
        ta = list(sc.sort_values(ascending=False).index[:top])
        w = 1.0 / len(ta)
        new_w = pd.Series(0.0, index=cols); new_w[ta] = w
        cost_line.iloc[start_i] += ((new_w - prev_w).abs().sum() / 2.0) * cost_bps / 10000.0
        prev_w = new_w.copy()

        days = idx[start_i:end_i + 1]
        seg = px.loc[days, ta]
        ent = seg.iloc[0].replace(0.0, np.nan)
        names = ent.index[ent.notna()].tolist()
        if not names:
            continue
        seg = seg[names]; e0 = seg.iloc[0]
        cum = seg / e0 - 1.0
        alive = pd.DataFrame(True, index=days, columns=names)
        exit_cost_days = {}
        for nm in names:
            col = cum[nm].values
            trg = None
            for j, c in enumerate(col):
                if np.isnan(c): break
                if tp is not None and c >= tp: trg = j; break
                if sl is not None and c <= -sl: trg = j; break
            if trg is not None and trg + 1 < len(col):
                alive.loc[days[trg + 2:], nm] = False
                exit_cost_days[days[trg + 1]] = exit_cost_days.get(days[trg + 1], 0.0) + w * cost_bps / 10000.0
        contrib = (dr.loc[days, names] * alive * w).sum(axis=1)
        port.loc[days] += contrib
        for d0, c in exit_cost_days.items():
            cost_line.loc[d0] += c
    return (port - cost_line).clip(lower=-0.5)

def metrics(ret):
    nav = (1 + ret).cumprod() * START
    ann = (nav.iloc[-1] / START) ** (DAYS / len(nav)) - 1
    vol = ret.std(ddof=1) * np.sqrt(DAYS)
    sharpe = ann / vol if vol > 0 else np.nan
    mdd = (nav / nav.cummax() - 1).min()
    return dict(ann=ann, vol=vol, sharpe=sharpe, mdd=mdd, final=float(nav.iloc[-1]))

def g(v): return round(v, 4) if isinstance(v, float) else v

scores = mom(px, 6, 1)
cases = {
    "baseline_原版":           dict(tp=None, sl=None),
    "tp20_sl30_当前默认":       dict(tp=0.20, sl=0.30),
    "sl30_only":              dict(tp=None, sl=0.30),
    "tp20_only":              dict(tp=0.20, sl=None),
    "tp15_sl25":              dict(tp=0.15, sl=0.25),
    "sl20_only":              dict(tp=None, sl=0.20),
}
oos = px.index >= pd.Timestamp("2022-01-01")
oos21 = px.index >= pd.Timestamp("2021-01-01")
rows = []
for name, kw in cases.items():
    r = backtest(px, scores, top=10, cost_bps=10, **kw)
    full = metrics(r); oo = metrics(r[oos]); oo21 = metrics(r[oos21])
    rows.append(dict(策略=name,
                     full_sh=g(full["sharpe"]), full_ann=g(full["ann"]), full_mdd=g(full["mdd"]), full_final=round(full["final"],0),
                     oos_sh=g(oo["sharpe"]), oos_ann=g(oo["ann"]), oos_mdd=g(oo["mdd"]), oos_final=round(oo["final"],0),
                     oos21_sh=g(oo21["sharpe"]), oos21_ann=g(oo21["ann"]), oos21_mdd=g(oo21["mdd"]), oos21_final=round(oo21["final"],0)))
df = pd.DataFrame(rows)
df.to_csv(OUT / "tpsl_backtest_compare.csv", index=False, encoding="utf-8-sig")
print(df.to_string(index=False))
print("saved:", OUT / "tpsl_backtest_compare.csv")
