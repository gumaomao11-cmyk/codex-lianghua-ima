# -*- coding: utf-8 -*-
"""
FIX C: replace the hand-written 60-ticker sector map with the REAL sector data
from us-stock-data/sector/sector_map.csv (514 tickers, sector + industry).
The old map left 74% of the 175-name pool as "OTHER", which invalidated any
sector-neutral attribution.
"""
import sys
from pathlib import Path
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
PROJ=Path(r"F:\even-codex\lianghua+IMA"); DB=PROJ/"data"/"duckdb"
SRC=Path(r"F:\even-codex\us-stock-data\sector\sector_map.csv")

sm=pd.read_csv(SRC)
sm.columns=[c.strip().lower() for c in sm.columns]
sm=sm.rename(columns={"symbol":"ticker"})
sm["ticker"]=sm.ticker.str.strip().str.upper()
sm=sm.drop_duplicates("ticker")
print(f"[real sector map] {len(sm)} tickers")
print("\nsector distribution:")
print(sm.sector.value_counts().to_string())

px=pd.read_parquet(DB/"prices.parquet")
univ=set(c for c in px.columns if c!="date")
cov=sm[sm.ticker.isin(univ)]
print(f"\ncoverage of 515-name universe: {len(cov)}/{len(univ)} = {len(cov)/len(univ):.1%}")
missing=sorted(univ-set(sm.ticker))
print(f"missing ({len(missing)}):", ", ".join(missing[:20]))

# how much does this fix the pool?
import duckdb
con=duckdb.connect()
pool=con.execute("""select distinct ticker from 'data/duckdb/aligned_v2_a.parquet'
                    where factor_clean_alpha is not null""").df().ticker.tolist()
sys.path.insert(0,str(PROJ))
from risk.industry_map import get_industry_map
old=get_industry_map().set_index("ticker")["sector"].to_dict()
new=sm.set_index("ticker")["sector"].to_dict()
o=pd.Series([old.get(t,"OTHER") for t in pool]).value_counts()
n=pd.Series([new.get(t,"OTHER") for t in pool]).value_counts()
print(f"\n=== 175-name pool sector coverage ===")
print(f"OLD hand map: OTHER={o.get('OTHER',0)}/{len(pool)} ({o.get('OTHER',0)/len(pool):.0%} unknown)")
print(f"NEW real map: OTHER={n.get('OTHER',0)}/{len(pool)} ({n.get('OTHER',0)/len(pool):.0%} unknown)")
print("\nNEW pool sector mix:")
print(n.to_string())

out=DB/"industry_map_real.parquet"
sm[["ticker","sector","industry"]].to_parquet(out,index=False)
print(f"\n[saved] {out}")
