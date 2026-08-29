# -*- coding: utf-8 -*-
"""Robustness: (a) drop ln_mcap (use ln_dvol as size proxy) to recover dates,
(b) pooled OLS with date fixed effects + clustered SE, (c) matched-pair test."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
df=pd.read_parquet("data/duckdb/heat_panel.parquet"); df["date"]=pd.to_datetime(df["date"])
df=df[df.date>="2026-04-22"].copy()

def nw_t(x,lag=5):
    x=np.asarray(x,float); x=x[~np.isnan(x)]; n=len(x); m=x.mean(); e=x-m
    s=(e@e)/n
    for l in range(1,min(lag,n-1)+1): s+=2*(1-l/(lag+1))*((e[l:]@e[:-l])/n)
    return m, m/np.sqrt(max(s,1e-18)/n), n

def fm(heat,h,ctrl):
    gam=[]; r2=[]
    for d,g in df.groupby("date"):
        g=g.dropna(subset=ctrl+[heat,h])
        if len(g)<80 or g[heat].std()==0: continue
        g=g.copy()
        for c in ctrl+[h]:
            lo,hi=g[c].quantile([0.01,0.99]); g[c]=g[c].clip(lo,hi)
        X=[np.ones(len(g))]+[g[c].values for c in ctrl]
        dm=pd.get_dummies(g["sector"],drop_first=True).astype(float)
        for c in dm.columns: X.append(dm[c].values)
        X.append(g[heat].values); X=np.column_stack(X); y=g[h].values
        b,*_=np.linalg.lstsq(X,y,rcond=None)
        yh=X@b; ss=((y-y.mean())**2).sum()
        r2.append(1-((y-yh)**2).sum()/ss if ss>0 else np.nan); gam.append(b[-1])
    if len(gam)<20: return None
    step=int(h.replace("ret_","").replace("d",""))
    m,t,n=nw_t(gam)
    return m,m*252/step,t,np.nanmean(r2),n

C1=["ln_mcap","mom_20d","mom_126d","ln_dvol","vol_20d"]
C2=["mom_20d","mom_126d","ln_dvol","vol_20d"]          # ln_dvol proxies size, keeps all dates
print("(a) controls WITHOUT ln_mcap (liquidity as size proxy) -> more usable dates")
print(f"{'HEAT':<15}{'h':<9}{'gamma':>11}{'ann':>10}{'t(NW5)':>9}{'R2':>8}{'dates':>7}  verdict")
for heat in ["heat_dummy","heat_ln","heat_surprise"]:
    for h in ["ret_1d","ret_5d","ret_21d"]:
        r=fm(heat,h,C2)
        if r:
            v="SIG" if abs(r[2])>1.96 else ("marg" if abs(r[2])>1.64 else "ZERO")
            print(f"{heat:<15}{h:<9}{r[0]:>11.5f}{r[1]:>9.1%}{r[2]:>9.2f}{r[3]:>8.3f}{r[4]:>7}  {v}")

print("\n(b) pooled OLS, date fixed effects, SE clustered by date")
for heat in ["heat_dummy","heat_ln"]:
    for h in ["ret_1d","ret_5d","ret_21d"]:
        d=df.dropna(subset=C2+[heat,h]).copy()
        # demean everything within date == date FE
        for c in C2+[heat,h]:
            d[c]=d[c]-d.groupby("date")[c].transform("mean")
        dm=pd.get_dummies(d["sector"],drop_first=True).astype(float)
        for c in dm.columns:
            d["s_"+str(c)]=dm[c]-dm[c].groupby(d["date"]).transform("mean")
        cols=C2+[heat]+["s_"+str(c) for c in dm.columns]
        X=d[cols].values; y=d[h].values
        b,*_=np.linalg.lstsq(X,y,rcond=None)
        res=y-X@b
        k=cols.index(heat)
        XtX_inv=np.linalg.pinv(X.T@X)
        # cluster by date
        meat=np.zeros((X.shape[1],X.shape[1]))
        for dt,idx in d.groupby("date").indices.items():
            Xg=X[idx]; ug=res[idx]; s=Xg.T@ug; meat+=np.outer(s,s)
        V=XtX_inv@meat@XtX_inv
        se=np.sqrt(max(V[k,k],1e-24)); t=b[k]/se
        step=int(h.replace("ret_","").replace("d",""))
        v="SIG" if abs(t)>1.96 else ("marg" if abs(t)>1.64 else "ZERO")
        print(f"  {heat:<13}{h:<9} gamma={b[k]:+.5f} ann={b[k]*252/step:+7.1%} t_cluster={t:+6.2f} n={len(d)} ndates={d.date.nunique()}  {v}")

print("\n(c) matched-pair: for each discussed stock, match nearest non-discussed by size+mom+sector")
for h in ["ret_1d","ret_5d","ret_21d"]:
    diffs=[]
    for dt,g in df.groupby("date"):
        g=g.dropna(subset=["ln_dvol","mom_126d",h])
        A=g[g.heat_dummy==1]; B=g[g.heat_dummy==0]
        if len(A)<3 or len(B)<20: continue
        for _,r in A.iterrows():
            cand=B[B.sector==r.sector]
            if len(cand)<3: cand=B
            dist=((cand.ln_dvol-r.ln_dvol)/max(g.ln_dvol.std(),1e-9))**2 + \
                 ((cand.mom_126d-r.mom_126d)/max(g.mom_126d.std(),1e-9))**2
            k=dist.nsmallest(3).index
            diffs.append(r[h]-cand.loc[k,h].mean())
    d=pd.Series(diffs).dropna(); step=int(h.replace("ret_","").replace("d",""))
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
    v="SIG" if abs(t)>1.96 else ("marg" if abs(t)>1.64 else "ZERO")
    print(f"  {h}: mean_diff={d.mean():+.5f} ann={d.mean()*252/step:+7.1%} t={t:+6.2f} n_pairs={len(d)}  {v}")
