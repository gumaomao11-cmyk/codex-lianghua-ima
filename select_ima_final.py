# -*- coding: utf-8 -*-
"""云端自动重选 IMA 目标名单：动量 + IMA 词频（window=60, λ=1.2），月度调仓。
读取仓库内 data/ 的行情和 data/ima/ 的因子表，输出 data/ima/ima_final_top10.csv。
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WS = Path(__file__).resolve().parent
DATA = Path(os.environ.get("STOCK_DATA_DIR") or WS / "data")
FAC = Path(os.environ.get("IMA_FACTORS_FILE") or WS / "data" / "ima" / "kb_abstract_factors.csv")
OUT = WS / "data" / "ima" / "ima_final_top10.csv"

px = pd.read_csv(DATA / "prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
recent = px.loc[px.index >= pd.Timestamp("2025-01-01")]
px = px.loc[:, recent.notna().sum() >= 150]
ml = px.resample("ME").last()
mom = (ml.shift(1) / ml.shift(7) - 1.0).replace([np.inf, -np.inf], np.nan)

fac = pd.read_csv(FAC, encoding="utf-8-sig")
fac["pdf_date"] = pd.to_datetime(fac["pdf_date"], errors="coerce")
fac = fac.dropna(subset=["pdf_date"])

def ima_score_for(d, window=60):
    s = fac[fac.pdf_date < d]
    s = s[s.pdf_date >= d - pd.Timedelta(days=window)]
    if s.empty: return pd.Series(dtype=float)
    g = s.groupby("ticker").agg(n=("n_pos", "sum"), d=("n_neg", "sum"), sig=("sign", "sum"))
    sc = g["n"] - g["d"] + g["sig"] * 0.5
    return (sc - sc.mean()) / (sc.std() if sc.std() > 0 else 1.0)

rebal = list(ml.truncate("2025-09-30", px.index[-1]).index)
if not rebal:
    print("ERROR: 没有可用的调仓日期"); sys.exit(1)
d = rebal[-1]
m = mom.loc[d].dropna()
z = ima_score_for(d, window=60).reindex(m.index)
score = m.rank(pct=True) + 1.2 * z.fillna(0.0)
sel = score.sort_values(ascending=False).index[:10].tolist()

rows = []
last_day = px.index[-1]
for i, t in enumerate(sel):
    px_hist = px.loc[px.index <= last_day, t].dropna()
    price = float(px_hist.iloc[-1]) if len(px_hist) else float("nan")
    rows.append({"rank": i + 1, "ticker": t,
                 "mom_score": round(float(m.loc[t]), 4),
                 "ima_z": round(float(z.loc[t]), 4) if t in z.index and not np.isnan(z.loc[t]) else 0.0,
                 "combined": round(float(score.loc[t]), 4),
                 "last_close": round(price, 2),
                 "date": str(d.date()),
                 "data_last": str(last_day.date())})
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"调仓日期 {d.date()}  window=60 λ=1.2 (momentum_ima)，数据截至 {last_day.date()}")
print(out.to_string(index=False))
print("saved:", OUT)
