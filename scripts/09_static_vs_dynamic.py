# -*- coding: utf-8 -*-
"""Decisive test: is the opinion factor a TIME-VARYING signal, or just a STATIC ticker tilt?"""
import duckdb, pandas as pd, numpy as np
con=duckdb.connect()
df=con.execute("""select date,ticker,factor_opinion_1d,ret_21d,ret_5d
                  from 'data/duckdb/aligned_dataset_b_ortho.parquet'
                  where date>='2025-01-01' and factor_opinion_1d is not null""").df()
df["date"]=pd.to_datetime(df["date"])

# variance decomposition
tm=df.groupby("ticker").factor_opinion_1d.transform("mean")
between=tm.var(); within=(df.factor_opinion_1d-tm).var(); tot=df.factor_opinion_1d.var()
print(f"factor variance: total={tot:.4f}  between-ticker={between:.4f} ({between/tot:.1%})  within-ticker={within:.4f} ({within/tot:.1%})")

full=con.execute("""select date,ticker,ret_21d,ret_5d from 'data/duckdb/aligned_dataset_b_ortho.parquet'
                    where date>='2025-01-01'""").df()
full["date"]=pd.to_datetime(full["date"])
for h in ["ret_5d","ret_21d"]:
    full["x_"+h]=full[h]-full.groupby("date")[h].transform("mean")
df=df.merge(full[["date","ticker","x_ret_5d","x_ret_21d"]],on=["date","ticker"],how="left")

# STATIC version: each ticker gets its FULL-SAMPLE mean factor (uses future info, on purpose)
df["static"]=df.groupby("ticker").factor_opinion_1d.transform("mean")
# DEMEANED version: pure time variation, ticker effect removed
df["dyn"]=df.factor_opinion_1d - df["static"]

def ls(fac,h):
    s=df.dropna(subset=[fac,"x_"+h]).copy()
    cnt=s.groupby("date")[fac].transform("count"); s=s[cnt>=10]
    s["q"]=s.groupby("date")[fac].transform(lambda x: pd.qcut(x.rank(method="first"),5,labels=False) if x.nunique()>4 else np.nan)
    s=s.dropna(subset=["q"]); s["q"]=s["q"].astype(int)
    piv=s.pivot_table(index="date",columns="q",values="x_"+h,aggfunc="mean")
    r=(piv[4]-piv[0]).dropna()
    step=int(h.replace("ret_","").replace("d",""))
    t=r.mean()/(r.std(ddof=1)/np.sqrt(len(r)))
    g=s.groupby("q")["x_"+h].mean()
    mono=pd.Series(g.values).corr(pd.Series(range(5)),method="spearman")
    return r.mean()*252/step, t, mono, len(r)

print()
print(f"{'variant':<10}{'horizon':<9}{'LS_ann':>10}{'t':>8}{'mono':>7}{'dates':>7}")
for fac,name in [("factor_opinion_1d","original"),("static","STATIC"),("dyn","DYNAMIC")]:
    for h in ["ret_5d","ret_21d"]:
        a,t,m,n=ls(fac,h)
        print(f"{name:<10}{h:<9}{a:>9.1%}{t:>8.2f}{m:>7.2f}{n:>7}")
