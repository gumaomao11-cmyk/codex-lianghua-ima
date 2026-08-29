# -*- coding: utf-8 -*-
"""Final: does the +42.9%/yr pool effect survive a sector-matched benchmark?
This is the cleanest single question, and it uses the FULL 2025-07~2026-08 window."""
import numpy as np, pandas as pd, duckdb, sys
sys.path.insert(0,".")
con=duckdb.connect()
df=con.execute("""select date,ticker,ret_1d,factor_clean_alpha
                  from 'data/duckdb/aligned_dataset_a_ortho.parquet'
                  where date>='2025-07-01' and date<='2026-08-26'""").df()
df["date"]=pd.to_datetime(df["date"])
ever=df.groupby("ticker").factor_clean_alpha.apply(lambda s: s.notna().any())
pool=set(ever[ever].index)
from risk.industry_map import get_industry_map
im=get_industry_map().set_index("ticker")["sector"].to_dict()
df["sector"]=df.ticker.map(im).fillna("OTHER")

# sector composition of the pool
pm=pd.Series({t:im.get(t,"OTHER") for t in pool}).value_counts()
print("pool sector mix:", pm.to_dict())

def stats(s):
    s=s.dropna(); n=len(s); m,sd=s.mean(),s.std(ddof=1); eq=(1+s).cumprod()
    return n,(1+s).prod()**(252/n)-1, m/sd*np.sqrt(252) if sd>0 else np.nan,(eq/eq.cummax()-1).min()

pool_ew=df[df.ticker.isin(pool)].groupby("date").ret_1d.mean()
full_ew=df.groupby("date").ret_1d.mean()

# sector-matched benchmark: same sector weights as pool, but ALL stocks in each sector
wts=(pm/pm.sum()).to_dict()
sec_ret=df.groupby(["date","sector"]).ret_1d.mean().unstack()
matched=sum(sec_ret[s]*w for s,w in wts.items() if s in sec_ret.columns)
matched=matched/sum(w for s,w in wts.items() if s in sec_ret.columns)

print(f"\n{'series':<40}{'days':>6}{'ann':>10}{'sharpe':>8}{'maxDD':>9}")
for k,s in [("pool EW (discussed 39)",pool_ew),("SECTOR-MATCHED benchmark",matched),("full universe EW",full_ew)]:
    n,a,sh,dd=stats(s); print(f"{k:<40}{n:>6}{a:>9.1%}{sh:>8.2f}{dd:>9.1%}")

def spread(a,b,label):
    i=a.index.intersection(b.index); d=(a.loc[i]-b.loc[i]).dropna()
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
    print(f"  {label}: ann={d.mean()*252:+7.1%}  t={t:+5.2f}  n={len(d)}")
print("\nspreads:")
spread(pool_ew,full_ew,"pool - full universe      ")
spread(pool_ew,matched,"pool - SECTOR-MATCHED     ")
spread(matched,full_ew,"SECTOR-MATCHED - full univ")
