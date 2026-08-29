# -*- coding: utf-8 -*-
"""FINAL VERDICT on factor_research_20d using the now-REAL controls."""
import duckdb, pandas as pd, numpy as np, sys, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,"."); sys.stdout.reconfigure(encoding="utf-8")
con=duckdb.connect()
df=con.execute("""select date,ticker,factor_research_20d,factor_clean_alpha_x,ret_1d,ret_5d,ret_21d,
                  mom_20d,mom_126d,vol_20d,rsi_14,turnover_20d,ln_dvol_20d,ln_mcap
                  from (select b.*, a.factor_clean_alpha as factor_clean_alpha_x
                        from 'data/duckdb/aligned_v2_b.parquet' b
                        left join 'data/duckdb/aligned_v2_a.parquet' a
                        on b.date=a.date and b.ticker=a.ticker)
                  where date>='2025-01-01'""").df()
df["date"]=pd.to_datetime(df["date"])
from risk.industry_map import get_industry_map
df["sector"]=df.ticker.map(get_industry_map().set_index("ticker")["sector"].to_dict()).fillna("OTHER")
CTRL=["ln_mcap","ln_dvol_20d","mom_20d","mom_126d","vol_20d","rsi_14"]
print("ctrl missing:",{c:f"{df[c].isna().mean():.1%}" for c in CTRL})

def nw_t(x,lag=5):
    x=np.asarray(x,float); x=x[~np.isnan(x)]; n=len(x); m=x.mean(); e=x-m
    s=(e@e)/n
    for l in range(1,min(lag,n-1)+1): s+=2*(1-l/(lag+1))*((e[l:]@e[:-l])/n)
    return m,m/np.sqrt(max(s,1e-18)/n),n

def fm(fac,h,ctrl,sector=True,minn=40):
    gam=[];r2=[]
    for dt,g in df.groupby("date"):
        g=g.dropna(subset=ctrl+[fac,h])
        if len(g)<minn or g[fac].std()==0: continue
        g=g.copy()
        for c in ctrl+[h]:
            lo,hi=g[c].quantile([0.01,0.99]); g[c]=g[c].clip(lo,hi)
        g[fac]=g[fac].rank(pct=True)-0.5     # rank-normalize the factor
        X=[np.ones(len(g))]+[g[c].values for c in ctrl]
        if sector:
            dm=pd.get_dummies(g["sector"],drop_first=True).astype(float)
            for c in dm.columns: X.append(dm[c].values)
        X.append(g[fac].values); X=np.column_stack(X); y=g[h].values
        try: bb,*_=np.linalg.lstsq(X,y,rcond=None)
        except Exception: continue
        yh=X@bb; ss=((y-y.mean())**2).sum()
        r2.append(1-((y-yh)**2).sum()/ss if ss>0 else np.nan); gam.append(bb[-1])
    if len(gam)<20: return None
    step=int(h.replace("ret_","").replace("d",""))
    m,t,n=nw_t(gam)
    return m,m*252/step,t,np.nanmean(r2),n

print("\n"+"="*92)
print("FAMA-MACBETH with REAL controls (ln_mcap, ln_dvol, mom20, mom126, vol20, rsi14) + sector")
print("="*92)
print(f"{'factor':<22}{'h':<8}{'gamma':>10}{'ann':>10}{'t(NW5)':>9}{'R2':>8}{'dates':>7}  verdict")
for fac in ["factor_research_20d","factor_clean_alpha_x"]:
    for h in ["ret_1d","ret_5d","ret_21d"]:
        r=fm(fac,h,CTRL)
        if not r: print(f"{fac:<22}{h:<8}  insufficient"); continue
        v="SIG" if abs(r[2])>1.96 else ("marg" if abs(r[2])>1.64 else "ZERO")
        print(f"{fac:<22}{h:<8}{r[0]:>10.5f}{r[1]:>9.1%}{r[2]:>9.2f}{r[3]:>8.3f}{r[4]:>7}  {v}")

print("\n--- without sector dummies ---")
for fac in ["factor_research_20d"]:
    for h in ["ret_1d","ret_5d","ret_21d"]:
        r=fm(fac,h,CTRL,sector=False)
        if r: print(f"{fac:<22}{h:<8}{r[0]:>10.5f}{r[1]:>9.1%}{r[2]:>9.2f}{r[3]:>8.3f}{r[4]:>7}")

print("\n--- subperiod stability, factor_research_20d, ret_21d ---")
print(f"{'period':<12}{'LS_ann':>10}{'t_nonov':>9}{'mono':>7}{'dates':>7}")
for nm,(s,e) in {"2025H1":("2025-01-01","2025-06-30"),"2025H2":("2025-07-01","2025-12-31"),
                 "2026H1":("2026-01-01","2026-06-30"),"2026Q3":("2026-07-01","2026-12-31")}.items():
    sub=df[(df.date>=s)&(df.date<=e)].copy()
    sub["x"]=sub.ret_21d-sub.groupby("date").ret_21d.transform("mean")
    ss=sub.dropna(subset=["factor_research_20d","x"])
    cnt=ss.groupby("date").factor_research_20d.transform("count"); ss=ss[cnt>=5]
    if ss.date.nunique()<15: print(f"{nm:<12}  insufficient"); continue
    ss=ss.copy()
    ss["q"]=ss.groupby("date").factor_research_20d.transform(lambda x: pd.qcut(x.rank(method="first"),5,labels=False,duplicates="drop") if x.nunique()>4 else np.nan)
    ss=ss.dropna(subset=["q"]); ss["q"]=ss["q"].astype(int)
    piv=ss.pivot_table(index="date",columns="q",values="x",aggfunc="mean")
    if 0 not in piv or 4 not in piv: print(f"{nm:<12}  no q"); continue
    ls=(piv[4]-piv[0]).dropna(); lsn=ls.iloc[::21]
    tn=lsn.mean()/(lsn.std(ddof=1)/np.sqrt(len(lsn))) if len(lsn)>2 else np.nan
    g=ss.groupby("q")["x"].mean(); mono=pd.Series(g.values).corr(pd.Series(range(len(g))),method="spearman")
    print(f"{nm:<12}{ls.mean()*12:>9.1%}{tn:>9.2f}{mono:>7.2f}{len(ls):>7}")
