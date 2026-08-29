# -*- coding: utf-8 -*-
"""
Is the 'discussed pool' itself contaminated? Two checks:
 (1) What ARE these 38 stocks? Compare to naive baselines that need NO LLM at all.
 (2) Survivorship: is the pool built from posts that mention stocks *because* they moved?
"""
import duckdb, pandas as pd, numpy as np
con=duckdb.connect()
df=con.execute("""select date,ticker,close,ret_1d,factor_clean_alpha
                  from 'data/duckdb/aligned_dataset_a_ortho.parquet'
                  where date>='2025-01-01' and date<='2026-08-26'""").df()
df["date"]=pd.to_datetime(df["date"])
df["is_cov"]=df.factor_clean_alpha.notna()
START="2025-07-01"

ever=df.groupby("ticker")["is_cov"].max()
pool=sorted(ever[ever].index.tolist())
print(f"discussed pool size = {len(pool)}")
print("pool =", ", ".join(pool))

bt=df[df.date>=START].copy()
def stats(s):
    s=s.dropna(); n=len(s); m,sd=s.mean(),s.std(ddof=1); eq=(1+s).cumprod()
    return n,(1+s).prod()**(252/n)-1, m/sd*np.sqrt(252) if sd>0 else np.nan,(eq/eq.cummax()-1).min()

print(f"\n{'baseline (NO LLM needed)':<44}{'days':>6}{'ann':>10}{'sharpe':>8}{'maxDD':>9}")
cands={}
cands["discussed 38 EW (needs LLM)"]=bt[bt.ticker.isin(pool)].groupby("date").ret_1d.mean()

# naive baseline 1: top-N by dollar volume proxy = highest close * ... we only have close.
# use highest average close as crude 'big name' proxy? better: momentum-based, and mega-cap tech hardcode
megacap=["NVDA","MSFT","AAPL","GOOGL","AMZN","META","AVGO","TSLA","AMD","QCOM",
         "MU","LRCX","AMAT","KLAC","MRVL","INTC","ARM","SMCI","PLTR","NFLX"]
have=[t for t in megacap if t in set(df.ticker)]
cands[f"hardcoded megacap-tech {len(have)} EW"]=bt[bt.ticker.isin(have)].groupby("date").ret_1d.mean()

# naive baseline 2: trailing 6m momentum top 38, rebalanced monthly (pure price, no text)
px=df.pivot_table(index="date",columns="ticker",values="close").sort_index()
mom=px.pct_change(126)
rets=px.pct_change().shift(-1)   # forward 1d, same convention
sel_dates=sorted(set(bt.date))
mo_series={}
cur=None; last_m=None
for d in sel_dates:
    m=(d.year,d.month)
    if m!=last_m:
        row=mom.loc[:d].iloc[-1].dropna()
        cur=row.sort_values(ascending=False).head(38).index.tolist()
        last_m=m
    if cur is not None and d in rets.index:
        mo_series[d]=rets.loc[d,cur].mean()
cands["momentum-6m top38 EW (no text)"]=pd.Series(mo_series)

for k,s in cands.items():
    n,a,sh,dd=stats(s)
    print(f"{k:<44}{n:>6}{a:>9.1%}{sh:>8.2f}{dd:>9.1%}")

print("\n=== overlap between discussed pool and megacap list ===")
print(f"  {len(set(pool)&set(have))} of {len(pool)} discussed stocks are in the hardcoded megacap list "
      f"({len(set(pool)&set(have))/len(pool):.0%})")
missing=set(pool)-set(have)
print("  discussed but not megacap:", ", ".join(sorted(missing)))
