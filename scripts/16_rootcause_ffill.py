# -*- coding: utf-8 -*-
"""ROOT CAUSE: unbounded ffill turns the event factor into a near-static per-ticker label."""
import duckdb, pandas as pd, numpy as np
con=duckdb.connect()
df=con.execute("""select date,ticker,factor_clean_alpha,turnover_20d,close
                  from 'data/duckdb/aligned_dataset_a_ortho.parquet'
                  where factor_clean_alpha is not null""").df()
df["date"]=pd.to_datetime(df["date"]); df=df.sort_values(["ticker","date"])
# a value is "FRESH" only if it differs from the previous day's value
df["prev"]=df.groupby("ticker")["factor_clean_alpha"].shift(1)
df["fresh"]=(df.prev.isna())|(np.abs(df.factor_clean_alpha-df.prev)>1e-12)
print(f"total factor obs        : {len(df)}")
print(f"FRESH (new info) obs    : {int(df.fresh.sum())}  ({df.fresh.mean():.1%})")
print(f"FFILLED (stale) obs     : {int((~df.fresh).sum())}  ({(~df.fresh).mean():.1%})")
# how long does one value persist?
runs=df.groupby(["ticker",df.fresh.cumsum()]).size()
print(f"\npersistence of a single value: mean={runs.mean():.1f} days, median={runs.median():.0f}, max={runs.max()}")
print(f"distinct fresh events per month:")
fr=df[df.fresh]
print(fr.groupby(pd.PeriodIndex(fr.date,freq="M")).size().to_string())

print("\n=== is turnover_20d a real turnover? ===")
d2=con.execute("""select date,ticker,close,turnover_20d from 'data/duckdb/aligned_dataset_a_ortho.parquet'
                  where ticker='NVDA' and date>='2026-08-01'""").df()
print(d2.head(8).to_string(index=False))
print("corr(close, turnover_20d) for NVDA:", d2.close.corr(d2.turnover_20d).round(4))
