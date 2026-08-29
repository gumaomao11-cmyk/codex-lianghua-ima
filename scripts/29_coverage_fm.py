# -*- coding: utf-8 -*-
"""
DECISIVE: coverage effect after controlling size/liquidity/momentum/sector,
via Fama-MacBeth on the FULL 512-name universe (coverage dummy, not sentiment).
Plus non-overlapping t-stats and subperiod stability.
"""
import duckdb, pandas as pd, numpy as np, sys, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
con=duckdb.connect()
d=con.execute("""select date,ticker,factor_clean_alpha,ret_1d,ret_5d,ret_21d,
                 ln_mcap,ln_dvol_20d,mom_20d,mom_126d,vol_20d,rsi_14
                 from 'data/duckdb/aligned_v2_a.parquet' where date>='2025-01-01'""").df()
d["date"]=pd.to_datetime(d["date"])
sm=pd.read_parquet("data/duckdb/industry_map_real.parquet")
d["sector"]=d.ticker.map(sm.set_index("ticker")["sector"].to_dict())
d=d.dropna(subset=["sector"])
d["iscov"]=d.factor_clean_alpha.notna().astype(float)
# rolling causal coverage: covered at least once in trailing 60 trading rows, excl today
d=d.sort_values(["ticker","date"])
d["cov_roll"]=d.groupby("ticker")["iscov"].transform(lambda s:s.shift(1).rolling(60,min_periods=1).max()).fillna(0)
CTRL=["ln_mcap","ln_dvol_20d","mom_20d","mom_126d","vol_20d","rsi_14"]

def nw_t(x,lag=5):
    x=np.asarray(x,float); x=x[~np.isnan(x)]; n=len(x); m=x.mean(); e=x-m
    s=(e@e)/n
    for l in range(1,min(lag,n-1)+1): s+=2*(1-l/(lag+1))*((e[l:]@e[:-l])/n)
    return m,m/np.sqrt(max(s,1e-18)/n),n

def fm(fac,h,ctrl,sector=True,sub=None,minn=60):
    dat=d if sub is None else sub
    gam=[];r2=[];dts=[]
    for dt,g in dat.groupby("date"):
        g=g.dropna(subset=ctrl+[fac,h])
        if len(g)<minn or g[fac].std()==0: continue
        g=g.copy()
        for c in ctrl+[h]:
            lo,hi=g[c].quantile([0.01,0.99]); g[c]=g[c].clip(lo,hi)
        X=[np.ones(len(g))]+[g[c].values for c in ctrl]
        if sector:
            dm=pd.get_dummies(g["sector"],drop_first=True).astype(float)
            for c in dm.columns: X.append(dm[c].values)
        X.append(g[fac].values); X=np.column_stack(X); y=g[h].values
        try: bb,*_=np.linalg.lstsq(X,y,rcond=None)
        except Exception: continue
        yh=X@bb; ss=((y-y.mean())**2).sum()
        r2.append(1-((y-yh)**2).sum()/ss if ss>0 else np.nan); gam.append(bb[-1]); dts.append(dt)
    if len(gam)<20: return None
    step=int(h.replace("ret_","").replace("d",""))
    m,t,n=nw_t(gam)
    gs=pd.Series(gam,index=dts)
    gn=gs.iloc[::step]
    tn=gn.mean()/(gn.std(ddof=1)/np.sqrt(len(gn))) if len(gn)>3 else np.nan
    return dict(g=m,ann=m*252/step,t=t,tn=tn,r2=np.nanmean(r2),n=n)

print("="*96)
print("FAMA-MACBETH: COVERAGE dummy, full 512-name universe, real controls + sector")
print("="*96)
print(f"{'variable':<14}{'h':<9}{'gamma':>10}{'ann':>10}{'t(NW5)':>9}{'t_nonov':>9}{'R2':>7}{'dates':>7}  verdict")
for fac in ["iscov","cov_roll"]:
    for h in ["ret_1d","ret_5d","ret_21d"]:
        r=fm(fac,h,CTRL)
        if not r: print(f"{fac:<14}{h:<9} insufficient"); continue
        v="SIG" if abs(r["tn"])>1.96 else ("marg" if abs(r["tn"])>1.64 else "ZERO")
        print(f"{fac:<14}{h:<9}{r['g']:>10.5f}{r['ann']:>9.1%}{r['t']:>9.2f}{r['tn']:>9.2f}{r['r2']:>7.3f}{r['n']:>7}  {v}")

print("\n--- drop ln_mcap (22% missing) to keep all dates ---")
C2=["ln_dvol_20d","mom_20d","mom_126d","vol_20d","rsi_14"]
print(f"{'variable':<14}{'h':<9}{'gamma':>10}{'ann':>10}{'t(NW5)':>9}{'t_nonov':>9}{'R2':>7}{'dates':>7}  verdict")
for fac in ["iscov","cov_roll"]:
    for h in ["ret_1d","ret_5d","ret_21d"]:
        r=fm(fac,h,C2)
        if r:
            v="SIG" if abs(r["tn"])>1.96 else ("marg" if abs(r["tn"])>1.64 else "ZERO")
            print(f"{fac:<14}{h:<9}{r['g']:>10.5f}{r['ann']:>9.1%}{r['t']:>9.2f}{r['tn']:>9.2f}{r['r2']:>7.3f}{r['n']:>7}  {v}")

print("\n--- subperiod stability: iscov, ret_1d, full controls ---")
print(f"{'period':<10}{'gamma_ann':>11}{'t(NW5)':>9}{'dates':>7}")
for nm,(s,e) in {"2025H1":("2025-01-01","2025-06-30"),"2025H2":("2025-07-01","2025-12-31"),
                 "2026H1":("2026-01-01","2026-06-30"),"2026Q3":("2026-07-01","2026-12-31")}.items():
    sub=d[(d.date>=s)&(d.date<=e)]
    r=fm("iscov","ret_1d",CTRL,sub=sub)
    if r: print(f"{nm:<10}{r['ann']:>10.1%}{r['t']:>9.2f}{r['n']:>7}")
    else: print(f"{nm:<10}  insufficient")
