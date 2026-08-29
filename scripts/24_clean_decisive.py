# -*- coding: utf-8 -*-
"""Clean-data version of the DECISIVE tests: sector-matched + within-pool + causal decomposition."""
import duckdb, pandas as pd, numpy as np, sys
sys.path.insert(0,"."); sys.stdout.reconfigure(encoding="utf-8")
con=duckdb.connect()
d=con.execute("""select date,ticker,close,factor_clean_alpha,ret_1d,ret_21d
                 from 'data/duckdb/aligned_v2_a.parquet' where date>='2025-01-01'""").df()
d["date"]=pd.to_datetime(d["date"])
b=con.execute("""select date,ticker,factor_research_20d,ret_1d,ret_21d
                 from 'data/duckdb/aligned_v2_b.parquet' where date>='2025-01-01'""").df()
b["date"]=pd.to_datetime(b["date"])
from risk.industry_map import get_industry_map
im=get_industry_map().set_index("ticker")["sector"].to_dict()
d["sector"]=d.ticker.map(im).fillna("OTHER")

def stats(s):
    s=s.dropna(); n=len(s); m,sd=s.mean(),s.std(ddof=1); eq=(1+s).cumprod()
    return n,(1+s).prod()**(252/n)-1,m/sd*np.sqrt(252) if sd>0 else np.nan,(eq/eq.cummax()-1).min()
def sp(a,bb,lab):
    i=a.index.intersection(bb.index); x=(a.loc[i]-bb.loc[i]).dropna()
    t=x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
    print(f"  {lab}: ann={x.mean()*252:+7.1%} t={t:+5.2f} n={len(x)}"); return t

print("="*74); print("1) POOL now = 175 tickers (was 39). Is it still just sector beta?"); print("="*74)
pool=set(d.loc[d.factor_clean_alpha.notna(),"ticker"].unique())
print(f"pool size={len(pool)}")
pm=pd.Series({t:im.get(t,"OTHER") for t in pool}).value_counts()
print("sector mix:",{k:int(v) for k,v in pm.items()})
pool_ew=d[d.ticker.isin(pool)].groupby("date").ret_1d.mean()
full_ew=d.groupby("date").ret_1d.mean()
sec=d.groupby(["date","sector"]).ret_1d.mean().unstack()
w={s:v/pm.sum() for s,v in pm.items() if s in sec.columns}
matched=sum(sec[s]*v for s,v in w.items())/sum(w.values())
print(f"\n{'series':<34}{'days':>6}{'ann':>10}{'sharpe':>8}{'maxDD':>9}")
for k,s in [("pool EW (175)",pool_ew),("SECTOR-MATCHED",matched),("full universe EW",full_ew)]:
    n,a,sh,dd=stats(s); print(f"{k:<34}{n:>6}{a:>9.1%}{sh:>8.2f}{dd:>9.1%}")
print("spreads:"); sp(pool_ew,full_ew,"pool - full      "); sp(pool_ew,matched,"pool - SECTOR-MTC")

print()
print("="*74); print("2) WITHIN-POOL: does the score pick winners among discussed stocks?"); print("="*74)
for tag,df2,fac in [("clean_alpha",d,"factor_clean_alpha"),("research_20d",b,"factor_research_20d")]:
    s=df2[df2[fac].notna()].copy()
    for h in ["ret_1d","ret_21d"]:
        s["w"]=s[h]-s.groupby("date")[h].transform("mean")   # demean WITHIN pool
        cnt=s.groupby("date")[fac].transform("count"); ss=s[cnt>=8].dropna(subset=["w",fac])
        if ss.date.nunique()<20: print(f"  {tag}/{h}: insufficient"); continue
        ss=ss.copy()
        ss["q"]=ss.groupby("date")[fac].transform(lambda x: pd.qcut(x.rank(method="first"),5,labels=False,duplicates="drop") if x.nunique()>4 else np.nan)
        ss=ss.dropna(subset=["q"]); ss["q"]=ss["q"].astype(int)
        piv=ss.pivot_table(index="date",columns="q",values="w",aggfunc="mean")
        ls=(piv[4]-piv[0]).dropna(); step=int(h.replace("ret_","").replace("d",""))
        t=ls.mean()/(ls.std(ddof=1)/np.sqrt(len(ls))); lsn=ls.iloc[::step]
        tn=lsn.mean()/(lsn.std(ddof=1)/np.sqrt(len(lsn))) if len(lsn)>3 else np.nan
        g=ss.groupby("q")["w"].mean(); mono=pd.Series(g.values).corr(pd.Series(range(len(g))),method="spearman")
        v="SIG" if abs(tn)>1.96 else ("marg" if abs(tn)>1.64 else "ZERO")
        print(f"  {tag:<13}{h:<8} LS_ann={ls.mean()*252/step:+7.1%} t_ov={t:+5.2f} t_nonov={tn:+5.2f} mono={mono:+.2f} n={len(ls)}  {v}")

print()
print("="*74); print("3) CAUSAL static-vs-dynamic (is it a fixed ticker list again?)"); print("="*74)
for tag,df2,fac in [("clean_alpha",d,"factor_clean_alpha"),("research_20d",b,"factor_research_20d")]:
    f=df2[df2[fac].notna()].sort_values("date").copy()
    tot=f[fac].var(); tm=f.groupby("ticker")[fac].transform("mean")
    print(f"  {tag}: between-ticker var share = {tm.var()/tot:.1%}")
    f["cs"]=f.groupby("ticker")[fac].transform(lambda s:s.shift(1).expanding().mean())
    f["cd"]=f[fac]-f["cs"]
    base=df2.copy()
    for h in ["ret_21d"]:
        base["x"]=base[h]-base.groupby("date")[h].transform("mean")
        m=f.merge(base[["date","ticker","x"]],on=["date","ticker"],how="left")
        for v,nm in [(fac,"original"),("cs","CAUSAL-STATIC"),("cd","CAUSAL-DYN")]:
            s=m.dropna(subset=[v,"x"]).copy()
            cnt=s.groupby("date")[v].transform("count"); s=s[cnt>=8]
            if s.date.nunique()<20: print(f"    {nm}: insufficient"); continue
            s["q"]=s.groupby("date")[v].transform(lambda x: pd.qcut(x.rank(method="first"),5,labels=False,duplicates="drop") if x.nunique()>4 else np.nan)
            s=s.dropna(subset=["q"]); s["q"]=s["q"].astype(int)
            piv=s.pivot_table(index="date",columns="q",values="x",aggfunc="mean")
            if 0 not in piv or 4 not in piv: continue
            ls=(piv[4]-piv[0]).dropna(); t=ls.mean()/(ls.std(ddof=1)/np.sqrt(len(ls)))
            lsn=ls.iloc[::21]; tn=lsn.mean()/(lsn.std(ddof=1)/np.sqrt(len(lsn))) if len(lsn)>3 else np.nan
            print(f"    {nm:<15} ret_21d LS_ann={ls.mean()*12:+7.1%} t_ov={t:+5.2f} t_nonov={tn:+5.2f} n={len(ls)}")
