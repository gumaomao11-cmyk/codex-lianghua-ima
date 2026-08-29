# -*- coding: utf-8 -*-
"""Coverage tilt test: is the alpha from the SCORE, or just from being TALKED ABOUT?"""
import duckdb, pandas as pd, numpy as np
con=duckdb.connect()
df=con.execute("""select date,ticker,factor_clean_alpha,ret_21d,ret_5d,ret_1d
                  from 'data/duckdb/aligned_dataset_a_ortho.parquet' where date>='2025-01-01'""").df()
df["date"]=pd.to_datetime(df["date"])
for h in ["ret_1d","ret_5d","ret_21d"]:
    df["x_"+h]=df[h]-df.groupby("date")[h].transform("mean")
df["covered"]=df.factor_clean_alpha.notna()

print("=== A. COVERAGE TILT (excess vs full 515-univ mean) ===")
for h in ["ret_1d","ret_5d","ret_21d"]:
    t=df.groupby("covered")["x_"+h].agg(["mean","count"])
    # daily series of covered-minus-universe
    d=df[df.covered].groupby("date")["x_"+h].mean()
    step=int(h.replace("ret_","").replace("d",""))
    tt=d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
    dn=d.iloc[::step]; ttn=dn.mean()/(dn.std(ddof=1)/np.sqrt(len(dn)))
    print(f"{h}: covered_excess={t.loc[True,'mean']:.4f} uncovered={t.loc[False,'mean']:.4f} "
          f"ann={d.mean()*252/step:.2%} t_overlap={tt:.2f} t_nonoverlap={ttn:.2f} n={len(dn)}")

print()
print("=== B. within-covered: does SCORE add anything beyond coverage? ===")
sub=df[df.covered].copy()
# demean again WITHIN covered universe -> pure score effect
for h in ["ret_1d","ret_5d","ret_21d"]:
    sub["w_"+h]=sub[h]-sub.groupby("date")[h].transform("mean")
cnt=sub.groupby("date").ticker.transform("count"); sub=sub[cnt>=10]
for h in ["ret_1d","ret_5d","ret_21d"]:
    sub["q"]=sub.groupby("date").factor_clean_alpha.transform(
        lambda s: pd.qcut(s.rank(method="first"),5,labels=False) if s.nunique()>4 else np.nan)
    s2=sub.dropna(subset=["q"])
    g=s2.groupby("q")["w_"+h].mean()
    piv=s2.pivot_table(index="date",columns="q",values="w_"+h,aggfunc="mean")
    ls=(piv[4]-piv[0]).dropna()
    step=int(h.replace("ret_","").replace("d",""))
    tt=ls.mean()/(ls.std(ddof=1)/np.sqrt(len(ls)))
    lsn=ls.iloc[::step]; ttn=lsn.mean()/(lsn.std(ddof=1)/np.sqrt(len(lsn)))
    mono=pd.Series(g.values).corr(pd.Series(range(5)),method="spearman")
    print(f"{h}: Q1..Q5 = "+" ".join(f"{v:+.4f}" for v in g.values)+
          f" | mono={mono:+.2f} LS_ann={ls.mean()*252/step:+.2%} t_ov={tt:+.2f} t_nonov={ttn:+.2f}")
