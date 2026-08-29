"""稳健版异常价格检测：区分"数据错值"与"真实市场事件"。

判定逻辑：数据错值的特征是价格在极短期内"跳出去又跳回来"，
而真实暴涨/暴跌后价格会停留在新水平。
"""
import pandas as pd, numpy as np, io
from pathlib import Path
LOG=io.open(r"F:\even-codex\lianghua+IMA\logs\clean_px2.log","w",encoding="utf-8",buffering=1)
def log(m): LOG.write(str(m)+"\n"); LOG.flush()

SRC=Path(r"F:\even-codex\us-stock-data\prices.csv")
DST=Path(r"F:\even-codex\lianghua+IMA\data\duckdb\prices_clean.csv")
px=pd.read_csv(SRC,parse_dates=["date"]).set_index("date").sort_index()
ret=px.pct_change()
pxc=px.copy()
errs=[]; reals=[]

for c in px.columns:
    s=ret[c]
    for i in np.where(s.abs()>0.45)[0]:
        if i<1 or i+5>=len(s): continue
        p_before=px[c].iloc[i-1]; p_spike=px[c].iloc[i]
        p_after=px[c].iloc[i+1:i+6].median()          # 之后5日中位数
        if pd.isna(p_before) or pd.isna(p_spike) or pd.isna(p_after) or p_before<=0: continue
        jump=p_spike/p_before-1
        # 回归度：spike后价格是否回到spike前水平
        revert=abs(p_after-p_before)/p_before
        stay  =abs(p_after-p_spike)/p_spike
        d=px.index[i]
        if revert < 0.25 and stay > 0.25:            # 回到原位 -> 错值
            errs.append((c,d.date(),p_before,p_spike,p_after,jump*100))
            pxc.loc[d,c]=np.nan
        else:
            reals.append((c,d.date(),p_before,p_spike,p_after,jump*100))

log("=== 判定为数据错值（跳出后回归原水平）-> 置为缺失 ===")
for c,d,a,b,e,j in errs:
    log(f"  {c:6} {d}  {a:8.2f} -> {b:8.2f} ({j:+6.0f}%) -> 后5日中位 {e:8.2f}")
log(f"  共 {len(errs)} 处\n")
log("=== 判定为真实市场事件（价格停留在新水平）-> 保留 ===")
for c,d,a,b,e,j in reals:
    log(f"  {c:6} {d}  {a:8.2f} -> {b:8.2f} ({j:+6.0f}%) -> 后5日中位 {e:8.2f}")
log(f"  共 {len(reals)} 处")

pxc.reset_index().to_csv(DST,index=False)
log(f"\n干净副本: {DST}   (原始数据未改动)")
