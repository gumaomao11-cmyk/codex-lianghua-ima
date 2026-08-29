"""扫描并修复价格数据中的异常跳变，输出干净副本（不改动原始数据源）。"""
import pandas as pd, numpy as np, io
from pathlib import Path

LOG=io.open(r"F:\even-codex\lianghua+IMA\logs\clean_px.log","w",encoding="utf-8",buffering=1)
def log(m): LOG.write(str(m)+"\n"); LOG.flush()

SRC = Path(r"F:\even-codex\us-stock-data\prices.csv")
DST = Path(r"F:\even-codex\lianghua+IMA\data\duckdb\prices_clean.csv")
px = pd.read_csv(SRC, parse_dates=["date"]).set_index("date").sort_index()
log(f"原始: {px.shape[0]} 行 x {px.shape[1]} 标的")

ret = px.pct_change()
# 判定：单日 |涨跌| > 50% 且次日出现明显反向（>25%），视为错值而非真实事件
fixed = []
pxc = px.copy()
for c in px.columns:
    s = ret[c]
    for i in np.where(s.abs() > 0.50)[0]:
        if i + 1 >= len(s): continue
        cur, nxt = s.iloc[i], s.iloc[i+1]
        if pd.isna(cur): continue
        # 反向确认：暴涨后暴跌 / 暴跌后暴涨
        if not pd.isna(nxt) and np.sign(cur) != np.sign(nxt) and abs(nxt) > 0.25:
            d = px.index[i]
            old = pxc[c].iloc[i]
            pxc.loc[d, c] = np.nan          # 置空，让下游按缺失处理
            fixed.append((c, d.date(), old, cur*100, nxt*100))

log(f"\n修复 {len(fixed)} 处错值（暴涨/跌后立即反向）:")
for c,d,o,a,b in fixed:
    log(f"  {c:6} {d}  px={o:8.2f}  当日{a:+7.0f}%  次日{b:+7.0f}%  -> 置为缺失")

log(f"\n未修复的大幅波动（视为真实市场事件，保留）:")
kept=0
for c in px.columns:
    s = ret[c]
    for i in np.where(s.abs() > 0.50)[0]:
        d = px.index[i]
        if any(f[0]==c and f[1]==d.date() for f in fixed): continue
        log(f"  {c:6} {d.date()}  {s.iloc[i]*100:+7.0f}%")
        kept+=1
log(f"  共 {kept} 处保留")

pxc.reset_index().to_csv(DST, index=False)
log(f"\n已写出干净副本: {DST}")
log("原始数据未被修改。")
