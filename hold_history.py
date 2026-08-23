# -*- coding: utf-8 -*-
"""分析：现在这10只加速因子 top10，最早几月同时确立为模型候选，并测算买入持有后的回调。"""
import os, sys
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np, pandas as pd
DATA = Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data")
OUT = Path(__file__).resolve().parent / "backtest_output"
px = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
px = px.loc[:, px.count() >= 2400]
cur = ["MU","WDC","INTC","STX","MRVL","NBIS","AMD","AMAT","GLW","FLEX"]

# 月频加速打分 = 0.5*6m-skip1 动量 + 0.5*近1月收益
m = px.resample("ME").last()
mom = m.shift(1) / m.shift(7) - 1.0
acc = m.pct_change(1)
sc = 0.5*mom + 0.5*acc
sc = sc.dropna(how="all")

rows = []
for d in sc.index:
    top = list(sc.loc[d].dropna().sort_values(ascending=False).index[:10])
    in_cnt = len([t for t in top if t in cur])
    rows.append((d, in_cnt, top))
df = pd.DataFrame(rows, columns=["month","cur_in_top10","top_list"])
df["all_cur"] = df["cur_in_top10"] >= 10
print("当前10只加入加速 top10 的月份情况：")
print(df[df.month >= '2025-01-01'].to_string(index=False))
print()

# 每只股票第一次入选月份
print("各股票：第一次进入加速 top10 的月份")
first = {}
for t in cur:
    dates = [d for d, r in zip(df.month, df.top_list) if t in r]
    first[t] = (dates[0].strftime("%Y-%m") if dates else None, len(dates), (dates[-1].strftime("%Y-%m") if dates else None))
for t in cur:
    f, cnt, l = first[t]
    print(f"  {t:<5} 首次={f}  入选次数={cnt}  最近入选={l}")
print()

# 最近一次、以及最早一次“10只全部同时在榜”的月份
allruns = df[df["all_cur"]]
print("10只全部同时在榜的月份区间：")
if len(allruns):
    groups = (allruns["month"].astype("period[M]").astype("int64").diff() > 1).cumsum()
    for g, sub in allruns.groupby(groups):
        print(f"  {sub['month'].iloc[0].date()} ~ {sub['month'].iloc[-1].date()}  ({len(sub)}个信号月)")
    print()

# 买入持有测算（以“最早全部同时在榜”月份次月起按收盘买入，10只等权）
if len(allruns):
    signal = allruns["month"].iloc[0]
    month_ends = m.index
    buy_day_idx = None
    for i in range(len(px)):
        if px.index[i] > signal:
            buy_day_idx = i; break
    if buy_day_idx:
        hold = pd.Series(0.0, index=cur)
        all_px = px.iloc[buy_day_idx:][cur].ffill()
        ret = all_px.pct_change().fillna(0.0)
        w = 1.0/len(cur)
        sr = (ret * w).sum(axis=1)
        nav = pd.Series(1.0, index=all_px.index); nav = (1+sr).cumprod()
        # 从下一个可交易日开始投资收益（避免看未来）
        dd = nav/nav.cummax()-1.0
        print(f"信号月={signal.date()}  从 {all_px.index[0].date()} 开始等权买入持有")
        print(f"期末(2026-08-21)净值={nav.iloc[-1]:.4f}  累计={nav.iloc[-1]-1:.1%}")
        print(f"最大回撤={dd.min():.1%}  当前回撤={dd.iloc[-1]:.1%}")
        # 分年度/分段
        for yr in sorted(set(all_px.index.year)):
            seg = sr.loc[str(yr)]
            print(f"  {yr}: 区间收益={seg.sum():+.1%}")
        print()
        print("10只个股自买入日以来的累计涨跌：")
        for t in cur:
            p0 = float(all_px[t].iloc[0]); p1 = float(all_px[t].iloc[-1])
            print(f"  {t:<5} {p0:>10.2f} -> {p1:>10.2f}  累计 {p1/p0-1:+.1%}")
