# -*- coding: utf-8 -*-
import duckdb, pandas as pd, numpy as np
con=duckdb.connect()
df=con.execute("""select date,ticker,factor_opinion_1d,factor_opinion_1d_ortho,raw_personal_opinion,ret_21d,ret_5d
                  from 'data/duckdb/aligned_dataset_b_ortho.parquet' where date>='2025-01-01'""").df()
df["date"]=pd.to_datetime(df["date"])
for h in ["ret_5d","ret_21d"]:
    df["x_"+h]=df[h]-df.groupby("date")[h].transform("mean")

for fac in ["factor_opinion_1d","factor_opinion_1d_ortho"]:
  for h in ["ret_5d","ret_21d"]:
    step=int(h.replace("ret_","").replace("d",""))
    s=df.dropna(subset=[fac,h]).copy()
    cnt=s.groupby("date")[fac].transform("count"); s=s[cnt>=10]
    s["q"]=s.groupby("date")[fac].transform(lambda x: pd.qcut(x.rank(method="first"),5,labels=False) if x.nunique()>4 else np.nan)
    s=s.dropna(subset=["q"]); s["q"]=s["q"].astype(int)
    piv=s.pivot_table(index="date",columns="q",values="x_"+h,aggfunc="mean")
    ls=(piv[4]-piv[0]).dropna()
    print(f"--- {fac} / {h}  n_dates={len(ls)}")
    for drop in [0,1,3,5,10]:
        v=ls.sort_values(ascending=False).iloc[drop:] if drop else ls
        v=v.reindex(ls.index).dropna() if drop==0 else v
        tt=v.mean()/(v.std(ddof=1)/np.sqrt(len(v)))
        print(f"    drop top{drop:>2}: LS_ann={v.mean()*252/step:+8.2%} t_ov={tt:+.2f}")
    print(f"    Q5-only ann={piv[4].mean()*252/step:+.2%}   Q1-only ann={piv[0].mean()*252/step:+.2%}")
    top=ls.sort_values(ascending=False).head(5)
    print("    best dates:", ", ".join(f"{d.date()}:{v:+.2%}" for d,v in top.items()))
    # what is in Q5 vs Q1
    print("    Q1 mean factor=%.3f  Q5 mean factor=%.3f"%(s[s.q==0][fac].mean(), s[s.q==4][fac].mean()))
    tk=s[s.q==4].ticker.value_counts().head(8)
    print("    Q5 top tickers:", dict(tk))
    tk1=s[s.q==0].ticker.value_counts().head(8)
    print("    Q1 top tickers:", dict(tk1))
