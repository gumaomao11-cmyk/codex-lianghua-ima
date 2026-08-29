# -*- coding: utf-8 -*-
"""
ITEM 2: is research COVERAGE just a proxy for LARGE CAP?
Tests:
  A. correlation / rank overlap between coverage and ln_mcap
  B. size-matched benchmark (match pool on mcap deciles, not sector)
  C. size+sector double-matched benchmark
  D. within-size-decile coverage spread
"""
import duckdb, pandas as pd, numpy as np, sys, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
con=duckdb.connect()
d=con.execute("""select date,ticker,factor_clean_alpha,ret_1d,ln_mcap,ln_dvol_20d,mom_126d
                 from 'data/duckdb/aligned_v2_a.parquet' where date>='2025-01-01'""").df()
d["date"]=pd.to_datetime(d["date"])
sm=pd.read_parquet("data/duckdb/industry_map_real.parquet")
d["sector"]=d.ticker.map(sm.set_index("ticker")["sector"].to_dict())
d=d.dropna(subset=["sector"])
d["iscov"]=d.factor_clean_alpha.notna().astype(int)
pool=set(d.loc[d.iscov==1,"ticker"].unique())
d["in_pool"]=d.ticker.isin(pool).astype(int)

print("="*78); print("A. Is coverage the same thing as size?"); print("="*78)
# per-ticker: total coverage days vs median ln_mcap
g=d.groupby("ticker").agg(cov_days=("iscov","sum"), mcap=("ln_mcap","median"),
                          dvol=("ln_dvol_20d","median")).dropna()
print(f"tickers with mcap data: {len(g)}")
print(f"  corr(coverage_days, ln_mcap) = {g.cov_days.corr(g.mcap):+.3f}   spearman={g.cov_days.corr(g.mcap,method='spearman'):+.3f}")
print(f"  corr(coverage_days, ln_dvol) = {g.cov_days.corr(g.dvol):+.3f}   spearman={g.cov_days.corr(g.dvol,method='spearman'):+.3f}")
g["dec"]=pd.qcut(g.mcap,10,labels=False,duplicates="drop")
tab=g.groupby("dec").agg(n=("cov_days","size"), in_pool=("cov_days",lambda s:(s>0).sum()),
                         mean_cov=("cov_days","mean"), mcap=("mcap","median"))
tab["pool_rate"]=tab.in_pool/tab.n
print("\nby market-cap decile (0=smallest, 9=largest):")
print(tab.to_string(float_format=lambda v:f"{v:.2f}"))

print()
print("="*78); print("B/C. size-matched and size+sector-matched benchmarks"); print("="*78)
# assign daily size decile within universe
d["sdec"]=d.groupby("date")["ln_mcap"].transform(lambda s: pd.qcut(s.rank(method="first"),10,labels=False,duplicates="drop"))
dd=d.dropna(subset=["sdec"]).copy(); dd["sdec"]=dd.sdec.astype(int)
def stats(s):
    s=s.dropna(); n=len(s); m,sd=s.mean(),s.std(ddof=1); eq=(1+s).cumprod()
    return n,(1+s).prod()**(252/n)-1,m/sd*np.sqrt(252) if sd>0 else np.nan,(eq/eq.cummax()-1).min()
def sp(a,b,lab):
    i=a.index.intersection(b.index); x=(a.loc[i]-b.loc[i]).dropna()
    t=x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
    v="SIG" if abs(t)>1.96 else ("marg" if abs(t)>1.64 else "ZERO")
    print(f"  {lab:<34} ann={x.mean()*252:+7.1%} t={t:+5.2f} n={len(x)} {v}")

pool_ew=dd[dd.in_pool==1].groupby("date").ret_1d.mean()
full_ew=dd.groupby("date").ret_1d.mean()
# size weights of the pool
pw=dd[dd.in_pool==1].groupby("sdec").size(); pw=pw/pw.sum()
szret=dd.groupby(["date","sdec"]).ret_1d.mean().unstack()
size_matched=sum(szret[k]*v for k,v in pw.items() if k in szret.columns)/sum(v for k,v in pw.items() if k in szret.columns)
# size x sector cell weights
dd["szg"]=(dd.sdec//2).astype(str)
dd["cell"]=dd["szg"]+"|"+dd.sector
cw=dd[dd.in_pool==1].groupby("cell").size(); cw=cw/cw.sum()
cellret=dd.groupby(["date","cell"]).ret_1d.mean().unstack()
ok={k:v for k,v in cw.items() if k in cellret.columns}
both_matched=sum(cellret[k]*v for k,v in ok.items())/sum(ok.values())

print(f"\n{'series':<38}{'days':>6}{'ann':>10}{'sharpe':>8}{'maxDD':>9}")
for k,s in [("pool EW",pool_ew),("SIZE-matched",size_matched),
            ("SIZE x SECTOR-matched",both_matched),("full universe EW",full_ew)]:
    n,a,sh,dd2=stats(s); print(f"{k:<38}{n:>6}{a:>9.1%}{sh:>8.2f}{dd2:>9.1%}")
print("\nspreads:")
sp(pool_ew,full_ew,"pool - full universe")
sp(pool_ew,size_matched,"pool - SIZE-matched")
sp(pool_ew,both_matched,"pool - SIZE x SECTOR-matched")
print("\npool size-decile weights:", {int(k):round(v,3) for k,v in pw.items()})

print()
print("="*78); print("D. within each size decile: covered vs not covered"); print("="*78)
print(f"{'decile':<8}{'n_cov':>8}{'n_unc':>8}{'cov_ann':>10}{'unc_ann':>10}{'spread':>10}{'t':>7}")
for k in sorted(dd.sdec.unique()):
    s=dd[dd.sdec==k]
    a=s[s.in_pool==1].groupby("date").ret_1d.mean(); b=s[s.in_pool==0].groupby("date").ret_1d.mean()
    i=a.index.intersection(b.index)
    if len(i)<50: print(f"{k:<8} too few dates"); continue
    x=(a.loc[i]-b.loc[i]).dropna(); t=x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
    print(f"{k:<8}{int(s[s.in_pool==1].ticker.nunique()):>8}{int(s[s.in_pool==0].ticker.nunique()):>8}"
          f"{a.mean()*252:>9.1%}{b.mean()*252:>9.1%}{x.mean()*252:>9.1%}{t:>7.2f}")
