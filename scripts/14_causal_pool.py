# -*- coding: utf-8 -*-
"""
STRICT causal pool test.
The 'discussed pool' was defined using the WHOLE sample -> lookahead.
Here the pool at date t only contains tickers discussed STRICTLY BEFORE t.
"""
import duckdb, pandas as pd, numpy as np
con=duckdb.connect()
df=con.execute("""select date,ticker,ret_1d,factor_clean_alpha
                  from 'data/duckdb/aligned_dataset_a_ortho.parquet'
                  where date>='2025-01-01' and date<='2026-08-26'""").df()
df["date"]=pd.to_datetime(df["date"])
df["is_cov"]=df.factor_clean_alpha.notna()
df=df.sort_values(["ticker","date"])
# causal: has this ticker EVER been discussed before today (shift 1)
df["seen_before"]=df.groupby("ticker")["is_cov"].transform(lambda s: s.shift(1).cummax()).fillna(0).astype(bool)
# causal rolling: discussed within trailing 60 rows, excluding today
df["seen_60"]=df.groupby("ticker")["is_cov"].transform(
    lambda s: s.shift(1).rolling(60,min_periods=1).max()).fillna(0).astype(bool)

START="2025-07-01"
bt=df[df.date>=START].copy()
def stats(s):
    s=s.dropna(); n=len(s); m,sd=s.mean(),s.std(ddof=1); eq=(1+s).cumprod()
    return n,(1+s).prod()**(252/n)-1,m/sd*np.sqrt(252) if sd>0 else np.nan,(eq/eq.cummax()-1).min()

ever=df.groupby("ticker")["is_cov"].max(); lookahead_pool=set(ever[ever].index)
variants={
 "pool = ever-discussed (LOOKAHEAD)": bt[bt.ticker.isin(lookahead_pool)].groupby("date").ret_1d.mean(),
 "pool = seen strictly before t (causal)": bt[bt.seen_before].groupby("date").ret_1d.mean(),
 "pool = seen in trailing 60 rows (causal)": bt[bt.seen_60].groupby("date").ret_1d.mean(),
 "pool = discussed TODAY (concurrent)": bt[bt.is_cov].groupby("date").ret_1d.mean(),
 "full universe 515 EW": bt.groupby("date").ret_1d.mean(),
}
print(f"{'variant':<44}{'days':>6}{'ann':>10}{'sharpe':>8}{'maxDD':>9}{'avgN':>7}")
sizes={
 "pool = ever-discussed (LOOKAHEAD)": bt[bt.ticker.isin(lookahead_pool)].groupby("date").ticker.nunique().mean(),
 "pool = seen strictly before t (causal)": bt[bt.seen_before].groupby("date").ticker.nunique().mean(),
 "pool = seen in trailing 60 rows (causal)": bt[bt.seen_60].groupby("date").ticker.nunique().mean(),
 "pool = discussed TODAY (concurrent)": bt[bt.is_cov].groupby("date").ticker.nunique().mean(),
 "full universe 515 EW": bt.groupby("date").ticker.nunique().mean(),
}
for k,s in variants.items():
    n,a,sh,dd=stats(s)
    print(f"{k:<44}{n:>6}{a:>9.1%}{sh:>8.2f}{dd:>9.1%}{sizes[k]:>7.0f}")

print("\n=== causal pool spread vs full universe ===")
full=bt.groupby("date").ret_1d.mean()
for key in ["pool = seen strictly before t (causal)","pool = seen in trailing 60 rows (causal)"]:
    s=variants[key]; idx=s.index.intersection(full.index)
    sp=(s.loc[idx]-full.loc[idx]).dropna()
    t=sp.mean()/(sp.std(ddof=1)/np.sqrt(len(sp)))
    print(f"  {key}: ann={sp.mean()*252:+.1%} t={t:+.2f} n={len(sp)}")
