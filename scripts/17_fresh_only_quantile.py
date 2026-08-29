# -*- coding: utf-8 -*-
"""Re-run the quantile test using ONLY FRESH factor observations (no ffill contamination)."""
import duckdb, pandas as pd, numpy as np
con=duckdb.connect()
df=con.execute("""select date,ticker,factor_clean_alpha,ret_1d,ret_5d,ret_21d
                  from 'data/duckdb/aligned_dataset_a_ortho.parquet' where date>='2025-01-01'""").df()
df["date"]=pd.to_datetime(df["date"])
for h in ["ret_1d","ret_5d","ret_21d"]:
    df["x_"+h]=df[h]-df.groupby("date")[h].transform("mean")
f=df[df.factor_clean_alpha.notna()].sort_values(["ticker","date"]).copy()
f["prev"]=f.groupby("ticker")["factor_clean_alpha"].shift(1)
f["fresh"]=(f.prev.isna())|(np.abs(f.factor_clean_alpha-f.prev)>1e-12)

def qtest(s,fac,h,minn):
    s=s.dropna(subset=[fac,"x_"+h]).copy()
    cnt=s.groupby("date")[fac].transform("count"); s=s[cnt>=minn]
    if s.empty or s.date.nunique()<20: return None
    s["q"]=s.groupby("date")[fac].transform(lambda x: pd.qcut(x.rank(method="first"),5,labels=False,duplicates="drop") if x.nunique()>4 else np.nan)
    s=s.dropna(subset=["q"]); s["q"]=s["q"].astype(int)
    piv=s.pivot_table(index="date",columns="q",values="x_"+h,aggfunc="mean")
    if 0 not in piv or 4 not in piv: return None
    ls=(piv[4]-piv[0]).dropna(); step=int(h.replace("ret_","").replace("d",""))
    t=ls.mean()/(ls.std(ddof=1)/np.sqrt(len(ls))) if ls.std(ddof=1)>0 else np.nan
    lsn=ls.iloc[::step]; tn=lsn.mean()/(lsn.std(ddof=1)/np.sqrt(len(lsn))) if len(lsn)>3 else np.nan
    g=s.groupby("q")["x_"+h].mean()
    mono=pd.Series(g.values).corr(pd.Series(range(len(g))),method="spearman")
    ic=s.groupby("date").apply(lambda x: x[fac].corr(x["x_"+h],method="spearman")).dropna()
    return dict(dates=len(ls),ann=ls.mean()*252/step,t=t,tn=tn,mono=mono,ic=ic.mean(),
                q=[g.get(i,np.nan) for i in range(5)])

print("=== ALL obs (with ffill, as previously reported) ===")
print(f"{'h':<8}{'dates':>6}{'LS_ann':>10}{'t_ov':>8}{'t_nov':>8}{'mono':>7}{'IC':>8}")
for h in ["ret_1d","ret_5d","ret_21d"]:
    r=qtest(f,"factor_clean_alpha",h,10)
    if r: print(f"{h:<8}{r['dates']:>6}{r['ann']:>9.1%}{r['t']:>8.2f}{r['tn']:>8.2f}{r['mono']:>7.2f}{r['ic']:>8.3f}")

print("\n=== FRESH obs only (real new information) ===")
fr=f[f.fresh]
print("fresh obs:",len(fr),"dates:",fr.date.nunique(),"avg per day:",round(fr.groupby('date').size().mean(),1))
print(f"{'h':<8}{'dates':>6}{'LS_ann':>10}{'t_ov':>8}{'t_nov':>8}{'mono':>7}{'IC':>8}")
for h in ["ret_1d","ret_5d","ret_21d"]:
    for mn in [5,3]:
        r=qtest(fr,"factor_clean_alpha",h,mn)
        if r:
            print(f"{h:<8}{r['dates']:>6}{r['ann']:>9.1%}{r['t']:>8.2f}{r['tn']:>8.2f}{r['mono']:>7.2f}{r['ic']:>8.3f}  (minN={mn})")
            break

print("\n=== FRESH: simple long/short on sign of sentiment, no quantiles ===")
for h in ["ret_1d","ret_5d","ret_21d"]:
    step=int(h.replace("ret_","").replace("d",""))
    s=fr.dropna(subset=["x_"+h])
    hi=s[s.factor_clean_alpha>0.5].groupby("date")["x_"+h].mean()
    lo=s[s.factor_clean_alpha<0.2].groupby("date")["x_"+h].mean()
    idx=hi.index.intersection(lo.index)
    if len(idx)<20: 
        print(f"{h}: too few overlapping dates ({len(idx)})"); continue
    d=(hi.loc[idx]-lo.loc[idx]).dropna()
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
    print(f"{h}: hi(>0.5) ann={hi.mean()*252/step:+.1%}  lo(<0.2) ann={lo.mean()*252/step:+.1%}  "
          f"spread ann={d.mean()*252/step:+.1%} t={t:+.2f} n={len(d)}")
