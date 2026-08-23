# -*- coding: utf-8 -*-
"""当前策略(6m-skip1 top10, 月度调仓)回测效果复算与报告生成。
与 optimize_v5 / tpsl_backtest_compare 同口径：单边10bps、日线收盘、月末信号次月生效。
"""
import os
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data")
OUT = Path(__file__).resolve().parent / "backtest_output"
DAYS = 252; START = 20000.0

px = pd.read_csv(DATA / "prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
px = px.loc[:, px.count() >= 2400]

def ml(x): return x.resample("ME").last()
def mom(px, p=6, k=1):
    m = ml(px); return m.shift(k) / m.shift(p + k) - 1

def backtest(px, scores, top=10, cost_bps=10):
    cols = list(px.columns); idx = px.index
    dr = px.pct_change().fillna(0.0)
    port = pd.Series(0.0, index=idx); cost_line = pd.Series(0.0, index=idx)
    me_arr = np.array(pd.DatetimeIndex(scores.index)); day_arr = np.array(idx)
    slot0 = np.searchsorted(me_arr, day_arr, side="right") - 1
    exact = (slot0 >= 0) & (day_arr == me_arr[np.clip(slot0, 0, len(me_arr)-1)])
    slot = np.clip(slot0 - exact.astype(int), 0, len(me_arr)-1)
    prev_w = pd.Series(0.0, index=cols); turnovers = []
    for s in range(1, len(me_arr)):
        seg_idx = np.where(slot == s)[0]
        if len(seg_idx) == 0: continue
        start_i = int(seg_idx[0]); end_i = int(seg_idx[-1])
        sig = pd.Timestamp(me_arr[s]); sc = scores.loc[sig].dropna()
        if len(sc) == 0:
            prev_w[:] = 0.0; continue
        ta = list(sc.sort_values(ascending=False).index[:top]); w = 1.0 / len(ta)
        new_w = pd.Series(0.0, index=cols); new_w[ta] = w
        to = (new_w - prev_w).abs().sum() / 2.0
        turnovers.append(to)
        cost_line.iloc[start_i] += to * cost_bps / 10000.0
        prev_w = new_w.copy()
        days = idx[start_i:end_i + 1]
        seg = px.loc[days, ta]; ent = seg.iloc[0].replace(0.0, np.nan)
        names = ent.index[ent.notna()].tolist()
        if not names: continue
        contrib = (dr.loc[days, names] * w).sum(axis=1)
        port.loc[days] += contrib
    net = (port - cost_line).clip(lower=-0.5)
    return net, (float(np.mean(turnovers)) if turnovers else 0.0)

def metrics(ret):
    nav = (1 + ret).cumprod() * START
    ann = (nav.iloc[-1] / START) ** (DAYS / len(nav)) - 1 if len(nav) else 0
    vol = ret.std(ddof=1) * np.sqrt(DAYS) if len(ret) > 1 else 0
    sharpe = ann / vol if vol > 0 else np.nan
    dd = nav / nav.cummax() - 1
    mdd = dd.min() if len(dd) else 0
    calmar = ann / abs(mdd) if mdd < 0 else np.nan
    return dict(ann=ann, vol=vol, sharpe=sharpe, mdd=float(mdd), calmar=calmar,
                final=float(nav.iloc[-1]), years=len(nav)/DAYS)

scores = mom(px)
ret, turn = backtest(px, scores, top=10, cost_bps=10)

full = metrics(ret)
is_mask = (ret.index >= pd.Timestamp("2017-02-01")) & (ret.index <= pd.Timestamp("2021-12-31"))
oos_mask = ret.index >= pd.Timestamp("2022-01-01")
isr = metrics(ret[is_mask]); oosr = metrics(ret[oos_mask])

yearly = (1 + ret).groupby(ret.index.year).prod() - 1
def pct(x): return f"{x*100:.1f}%"

rows = [dict(区间="全期(2016-08~2026-08)", 年化=pct(full["ann"]), 波动=pct(full["vol"]), 夏普=round(full["sharpe"],2),
             最大回撤=pct(full["mdd"]), Calmar=round(full["calmar"],2), 期末=f"${full['final']:,.0f}"),
        dict(区间="样本内(2017-2021)", 年化=pct(isr["ann"]), 波动=pct(isr["vol"]), 夏普=round(isr["sharpe"],2),
             最大回撤=pct(isr["mdd"]), Calmar=round(isr["calmar"],2), 期末=f"${isr['final']:,.0f}"),
        dict(区间="样本外(2022-2026-08)", 年化=pct(oosr["ann"]), 波动=pct(oosr["vol"]), 夏普=round(oosr["sharpe"],2),
             最大回撤=pct(oosr["mdd"]), Calmar=round(oosr["calmar"],2), 期末=f"${oosr['final']:,.0f}")]
df = pd.DataFrame(rows)
df.to_csv(OUT / "current_backtest_metrics.csv", index=False, encoding="utf-8-sig")
yr = yearly.reset_index(); yr.columns = ["year", "return"]; yr["return"] = yr["return"]*100
yr.to_csv(OUT / "current_backtest_yearly.csv", index=False, encoding="utf-8-sig")

lines = []
lines.append("# 当前策略回测效果复算（6m-skip1 top10 · 月度调仓）\n")
lines.append("> 口径：与项目原回测一致。日线收盘价、单边成本10bps、月末动量信号次月生效、每只等权1/10、起步$20,000。\n")
lines.append(f"> 数据：F:/even-codex/us-stock-data/prices.csv（截止 {px.index[-1].date()}）。\n")
lines.append(f"\n年化换手(月度平均单边换手): {turn*100:.0f}%\n")
lines.append("\n## 收益/风险指标\n")
lines.append(df.to_markdown(index=False))
lines.append("\n## 逐年收益\n")
lines.append("| 年份 | 收益 |")
lines.append("|---:|---:|")
for y, v in yearly.items():
    lines.append(f"| {y} | {v*100:.1f}% |")
lines.append("\n## 结论摘要\n")
lines.append(f"- 全期: 年化 **{full['ann']*100:.1f}%**, 夏普 **{full['sharpe']:.2f}**, 最大回撤 **{full['mdd']*100:.1f}%**, 期末 **${full['final']:,.0f}**")
lines.append(f"- 样本外(2022起): 年化 **{oosr['ann']*100:.1f}%**, 夏普 **{oosr['sharpe']:.2f}**, 最大回撤 **{oosr['mdd']*100:.1f}%**, 期末 **${oosr['final']:,.0f}**")
lines.append("- 口径提醒: 价格不含分红；数据池存在幸存者偏差；实盘执行点位/滑点会打折，建议按 5~8 折参考。")
md = "\n".join(lines) + "\n"
(OUT / "current_backtest_report.md").write_text(md, encoding="utf-8")
print(df.to_string(index=False))
print("\nYearly:")
print(yearly.round(4).to_string())
print("\nsaved:", OUT/"current_backtest_metrics.csv", OUT/"current_backtest_report.md")
