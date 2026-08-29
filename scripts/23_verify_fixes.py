# -*- coding: utf-8 -*-
"""Verify both fixes, then re-run the decisive tests on clean data."""
import duckdb, pandas as pd, numpy as np, sys
sys.path.insert(0,"."); sys.stdout.reconfigure(encoding="utf-8")
con=duckdb.connect()

print("="*70); print("VERIFY FIX A: staleness"); print("="*70)
for tag,f,col in [("OLD","aligned_dataset_a_ortho","factor_clean_alpha"),
                  ("NEW","aligned_v2_a","factor_clean_alpha")]:
    d=con.execute(f"select date,ticker,{col} from 'data/duckdb/{f}.parquet' where {col} is not null").df()
    d["date"]=pd.to_datetime(d["date"]); d=d.sort_values(["ticker","date"])
    d["prev"]=d.groupby("ticker")[col].shift(1)
    d["fresh"]=(d.prev.isna())|(np.abs(d[col]-d.prev)>1e-9)
    runs=d.groupby(["ticker",d.fresh.cumsum()]).size()
    print(f"{tag}: obs={len(d):>6}  fresh={d.fresh.mean():>6.1%}  "
          f"persist mean={runs.mean():>5.1f}d median={runs.median():>3.0f}d MAX={runs.max():>4}d  "
          f"range={d.date.min().date()}~{d.date.max().date()}")

print()
print("="*70); print("VERIFY FIX B: turnover is real"); print("="*70)
for tag,f in [("OLD","aligned_dataset_a_ortho"),("NEW","aligned_v2_a")]:
    d=con.execute(f"select close,turnover_20d from 'data/duckdb/{f}.parquet' where ticker='NVDA' and date>='2025-01-01'").df()
    print(f"{tag}: corr(close,turnover_20d)={d.close.corr(d.turnover_20d):+.3f}  "
          f"turnover mean={d.turnover_20d.mean():,.0f}  (real volume ~1e8, mean-price ~200)")

print()
print("="*70); print("RE-RUN: quantile / long-short on CLEAN data (route A)"); print("="*70)
d=con.execute("""select date,ticker,factor_clean_alpha,ret_1d,ret_5d,ret_21d
                 from 'data/duckdb/aligned_v2_a.parquet' where date>='2025-01-01'""").df()
d["date"]=pd.to_datetime(d["date"])
for h in ["ret_1d","ret_5d","ret_21d"]:
    d["x_"+h]=d[h]-d.groupby("date")[h].transform("mean")
f=d[d.factor_clean_alpha.notna()]
print(f"signal obs={len(f)} dates={f.date.nunique()} avg/day={f.groupby('date').size().mean():.1f} tickers={f.ticker.nunique()}")

def qt(s,fac,h,minn=8):
    s=s.dropna(subset=[fac,"x_"+h]).copy()
    cnt=s.groupby("date")[fac].transform("count"); s=s[cnt>=minn]
    if s.date.nunique()<20: return None
    s["q"]=s.groupby("date")[fac].transform(lambda x: pd.qcut(x.rank(method="first"),5,labels=False,duplicates="drop") if x.nunique()>4 else np.nan)
    s=s.dropna(subset=["q"]); s["q"]=s["q"].astype(int)
    piv=s.pivot_table(index="date",columns="q",values="x_"+h,aggfunc="mean")
    if 0 not in piv or 4 not in piv: return None
    ls=(piv[4]-piv[0]).dropna(); step=int(h.replace("ret_","").replace("d",""))
    t=ls.mean()/(ls.std(ddof=1)/np.sqrt(len(ls)))
    lsn=ls.iloc[::step]; tn=lsn.mean()/(lsn.std(ddof=1)/np.sqrt(len(lsn))) if len(lsn)>3 else np.nan
    g=s.groupby("q")["x_"+h].mean()
    mono=pd.Series(g.values).corr(pd.Series(range(len(g))),method="spearman")
    ic=s.groupby("date").apply(lambda x:x[fac].corr(x["x_"+h],method="spearman")).dropna()
    return dict(n=len(ls),ann=ls.mean()*252/step,t=t,tn=tn,mono=mono,ic=ic.mean(),
                ir=ic.mean()/ic.std(ddof=1) if ic.std(ddof=1)>0 else np.nan)
print(f"\n{'h':<9}{'dates':>6}{'LS_ann':>10}{'t_ov':>8}{'t_nov':>8}{'mono':>7}{'IC':>8}{'IR':>7}")
for h in ["ret_1d","ret_5d","ret_21d"]:
    r=qt(f,"factor_clean_alpha",h)
    if r: print(f"{h:<9}{r['n']:>6}{r['ann']:>9.1%}{r['t']:>8.2f}{r['tn']:>8.2f}{r['mono']:>7.2f}{r['ic']:>8.3f}{r['ir']:>7.3f}")
    else: print(f"{h:<9}  insufficient")

print()
print("="*70); print("RE-RUN: route B factors"); print("="*70)
db=con.execute("""select date,ticker,factor_research_20d,factor_event_3d,factor_news_1d,factor_opinion_1d,
                  ret_1d,ret_5d,ret_21d from 'data/duckdb/aligned_v2_b.parquet' where date>='2025-01-01'""").df()
db["date"]=pd.to_datetime(db["date"])
for h in ["ret_1d","ret_5d","ret_21d"]:
    db["x_"+h]=db[h]-db.groupby("date")[h].transform("mean")
print(f"{'factor':<22}{'h':<9}{'dates':>6}{'LS_ann':>10}{'t_ov':>8}{'t_nov':>8}{'mono':>7}{'IC':>8}")
for fac in ["factor_research_20d","factor_event_3d","factor_news_1d","factor_opinion_1d"]:
    sub=db[db[fac].notna()]
    for h in ["ret_5d","ret_21d"]:
        r=qt(sub,fac,h,minn=5)
        if r: print(f"{fac:<22}{h:<9}{r['n']:>6}{r['ann']:>9.1%}{r['t']:>8.2f}{r['tn']:>8.2f}{r['mono']:>7.2f}{r['ic']:>8.3f}")
        else: print(f"{fac:<22}{h:<9}  insufficient dates (obs={len(sub)})")
