# -*- coding: utf-8 -*-
"""Re-run sector attribution with REAL sector map (100% coverage vs 26% before)."""
import duckdb, pandas as pd, numpy as np, sys
sys.stdout.reconfigure(encoding="utf-8")
con=duckdb.connect()
d=con.execute("""select date,ticker,factor_clean_alpha,ret_1d
                 from 'data/duckdb/aligned_v2_a.parquet' where date>='2025-01-01'""").df()
d["date"]=pd.to_datetime(d["date"])
sm=pd.read_parquet("data/duckdb/industry_map_real.parquet")
S=sm.set_index("ticker")["sector"].to_dict(); I=sm.set_index("ticker")["industry"].to_dict()
d["sector"]=d.ticker.map(S); d["industry"]=d.ticker.map(I)
print("unmapped rows:",d.sector.isna().mean().round(4))
d=d.dropna(subset=["sector"])

pool=set(d.loc[d.factor_clean_alpha.notna(),"ticker"].unique())
print(f"pool={len(pool)}  universe={d.ticker.nunique()}")

def stats(s):
    s=s.dropna(); n=len(s); m,sd=s.mean(),s.std(ddof=1); eq=(1+s).cumprod()
    return n,(1+s).prod()**(252/n)-1,m/sd*np.sqrt(252) if sd>0 else np.nan,(eq/eq.cummax()-1).min()
def sp(a,b,lab):
    i=a.index.intersection(b.index); x=(a.loc[i]-b.loc[i]).dropna()
    t=x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
    v="SIG" if abs(t)>1.96 else ("marg" if abs(t)>1.64 else "ZERO")
    print(f"  {lab:<30} ann={x.mean()*252:+7.1%}  t={t:+5.2f}  n={len(x)}  {v}")

pool_ew=d[d.ticker.isin(pool)].groupby("date").ret_1d.mean()
full_ew=d.groupby("date").ret_1d.mean()

# sector-matched: pool's sector weights x each sector's full return
pm=pd.Series({t:S[t] for t in pool if t in S}).value_counts()
sec=d.groupby(["date","sector"]).ret_1d.mean().unstack()
w={s:v/pm.sum() for s,v in pm.items() if s in sec.columns}
matched=sum(sec[s]*v for s,v in w.items())/sum(w.values())

# INDUSTRY-matched (finer, 100+ industries)
pmi=pd.Series({t:I[t] for t in pool if t in I}).value_counts()
ind=d.groupby(["date","industry"]).ret_1d.mean().unstack()
wi={s:v/pmi.sum() for s,v in pmi.items() if s in ind.columns}
matched_i=sum(ind[s]*v for s,v in wi.items())/sum(wi.values())

print(f"\n{'series':<36}{'days':>6}{'ann':>10}{'sharpe':>8}{'maxDD':>9}")
for k,s in [("pool EW (175)",pool_ew),("SECTOR-matched (real, 10 sec)",matched),
            ("INDUSTRY-matched (real, fine)",matched_i),("full universe EW",full_ew)]:
    n,a,sh,dd=stats(s); print(f"{k:<36}{n:>6}{a:>9.1%}{sh:>8.2f}{dd:>9.1%}")
print("\nspreads:")
sp(pool_ew,full_ew,"pool - full universe")
sp(pool_ew,matched,"pool - SECTOR-matched")
sp(pool_ew,matched_i,"pool - INDUSTRY-matched")
print("\npool sector mix:", {k:int(v) for k,v in pm.items()})
