# -*- coding: utf-8 -*-
"""Causal (no-lookahead) decomposition: expanding-window ticker mean."""
import duckdb, pandas as pd, numpy as np
con=duckdb.connect()
f=con.execute("""select date,ticker,factor_opinion_1d from 'data/duckdb/aligned_dataset_b_ortho.parquet'
                 where date>='2025-01-01' and factor_opinion_1d is not null""").df()
full=con.execute("""select date,ticker,ret_5d,ret_21d from 'data/duckdb/aligned_dataset_b_ortho.parquet'
                    where date>='2025-01-01'""").df()
for d in (f,full): d["date"]=pd.to_datetime(d["date"])
for h in ["ret_5d","ret_21d"]:
    full["x_"+h]=full[h]-full.groupby("date")[h].transform("mean")
f=f.sort_values("date")
# expanding mean per ticker, SHIFTED so today is excluded -> causal
g=f.groupby("ticker").factor_opinion_1d
f["causal_static"]=g.transform(lambda s: s.shift(1).expanding().mean())
f["causal_dyn"]=f.factor_opinion_1d-f["causal_static"]
df=f.merge(full[["date","ticker","x_ret_5d","x_ret_21d"]],on=["date","ticker"],how="left")

def ls(fac,h):
    s=df.dropna(subset=[fac,"x_"+h]).copy()
    cnt=s.groupby("date")[fac].transform("count"); s=s[cnt>=10]
    if s.empty: return (np.nan,)*4
    s["q"]=s.groupby("date")[fac].transform(lambda x: pd.qcut(x.rank(method="first"),5,labels=False) if x.nunique()>4 else np.nan)
    s=s.dropna(subset=["q"]); s["q"]=s["q"].astype(int)
    piv=s.pivot_table(index="date",columns="q",values="x_"+h,aggfunc="mean")
    r=(piv[4]-piv[0]).dropna(); step=int(h.replace("ret_","").replace("d",""))
    t=r.mean()/(r.std(ddof=1)/np.sqrt(len(r)))
    gg=s.groupby("q")["x_"+h].mean()
    mono=pd.Series(gg.values).corr(pd.Series(range(5)),method="spearman")
    return r.mean()*252/step,t,mono,len(r)

print(f"{'variant':<16}{'horizon':<9}{'LS_ann':>10}{'t':>8}{'mono':>7}{'dates':>7}")
for fac,name in [("factor_opinion_1d","original"),("causal_static","CAUSAL-STATIC"),("causal_dyn","CAUSAL-DYN")]:
    for h in ["ret_5d","ret_21d"]:
        a,t,m,n=ls(fac,h)
        print(f"{name:<16}{h:<9}{a:>9.1%}{t:>8.2f}{m:>7.2f}{n:>7}")
