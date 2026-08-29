# -*- coding: utf-8 -*-
"""
ITEM 2 FINAL: Fama-MacBeth regression.
  ret_fwd[i,t] = a + b*ln_mcap + b*mom_20d + b*mom_126d + b*ln_dvol + b*vol_20d
                 + SECTOR dummies + gamma*HEAT[i,t] + e
Cross-sectional OLS each date -> t-stat on the time series of gamma (Newey-West lag=5).
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet("data/duckdb/heat_panel.parquet")
df["date"] = pd.to_datetime(df["date"])
# restrict to the window where the LLM cache actually has coverage
df = df[df.date >= "2026-04-22"].copy()
CTRL = ["ln_mcap","mom_20d","mom_126d","ln_dvol","vol_20d"]
print("window:", df.date.min().date(), "~", df.date.max().date(), " dates:", df.date.nunique())
print("avg discussed/day:", round(df.groupby("date").heat_dummy.sum().mean(),1))

def nw_t(x, lag=5):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x); m = x.mean(); e = x - m
    g0 = (e@e)/n; s = g0
    for l in range(1, min(lag, n-1)+1):
        gl = (e[l:]@e[:-l])/n
        s += 2*(1-l/(lag+1))*gl
    se = np.sqrt(max(s,1e-18)/n)
    return m, m/se, n

def fm(heat, h, add_sector=True, winsor=True):
    gam, a_int, r2 = [], [], []
    need = CTRL+[heat,h]
    for d, g in df.groupby("date"):
        g = g.dropna(subset=need)
        if len(g) < 60 or g[heat].std() == 0: continue
        g = g.copy()
        if winsor:
            for c in CTRL+[h]:
                lo,hi = g[c].quantile([0.01,0.99]); g[c]=g[c].clip(lo,hi)
        X = [np.ones(len(g))] + [g[c].values for c in CTRL]
        names = ["const"]+CTRL
        if add_sector:
            dm = pd.get_dummies(g["sector"], drop_first=True).astype(float)
            for c in dm.columns: X.append(dm[c].values); names.append("sec_"+str(c))
        X.append(g[heat].values); names.append("HEAT")
        X = np.column_stack(X); y = g[h].values
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception: continue
        yh = X@beta; ss = ((y-y.mean())**2).sum()
        r2.append(1-((y-yh)**2).sum()/ss if ss>0 else np.nan)
        gam.append(beta[-1]); a_int.append(beta[0])
    if len(gam) < 20: return None
    step = int(h.replace("ret_","").replace("d",""))
    gm, gt, n = nw_t(gam)
    return dict(h=h, heat=heat, n=n, gamma=gm, gamma_ann=gm*252/step, t=gt,
                r2=np.nanmean(r2), sector=add_sector)

print("\n" + "="*96)
print("FAMA-MACBETH: gamma on HEAT, controlling size/mom20/mom126/liquidity/vol + SECTOR dummies")
print("="*96)
print(f"{'HEAT var':<16}{'horizon':<9}{'gamma':>11}{'gamma_ann':>11}{'t(NW5)':>9}{'R2':>8}{'dates':>7}  verdict")
rows=[]
for heat in ["heat_dummy","heat_count","heat_ln","heat_surprise"]:
    for h in ["ret_1d","ret_5d","ret_21d"]:
        r = fm(heat, h, add_sector=True)
        if not r: 
            print(f"{heat:<16}{h:<9}  insufficient dates"); continue
        v = "SIGNIFICANT" if abs(r["t"])>1.96 else ("marginal" if abs(r["t"])>1.64 else "ZERO")
        print(f"{heat:<16}{h:<9}{r['gamma']:>11.5f}{r['gamma_ann']:>10.1%}{r['t']:>9.2f}{r['r2']:>8.3f}{r['n']:>7}  {v}")
        rows.append(r)

print("\n--- same, WITHOUT sector dummies (to see how much sector explains) ---")
print(f"{'HEAT var':<16}{'horizon':<9}{'gamma':>11}{'gamma_ann':>11}{'t(NW5)':>9}{'R2':>8}{'dates':>7}")
for heat in ["heat_dummy","heat_ln"]:
    for h in ["ret_1d","ret_5d","ret_21d"]:
        r = fm(heat, h, add_sector=False)
        if r: print(f"{heat:<16}{h:<9}{r['gamma']:>11.5f}{r['gamma_ann']:>10.1%}{r['t']:>9.2f}{r['r2']:>8.3f}{r['n']:>7}")

print("\n--- RAW (no controls at all): the +42.9%/yr number, for reference ---")
print(f"{'HEAT var':<16}{'horizon':<9}{'gamma':>11}{'gamma_ann':>11}{'t(NW5)':>9}{'dates':>7}")
for heat in ["heat_dummy"]:
    for h in ["ret_1d","ret_5d","ret_21d"]:
        gam=[]
        for d,g in df.groupby("date"):
            g=g.dropna(subset=[heat,h])
            if len(g)<60 or g[heat].std()==0: continue
            X=np.column_stack([np.ones(len(g)),g[heat].values])
            b,*_=np.linalg.lstsq(X,g[h].values,rcond=None); gam.append(b[-1])
        step=int(h.replace("ret_","").replace("d",""))
        gm,gt,n=nw_t(gam)
        print(f"{heat:<16}{h:<9}{gm:>11.5f}{gm*252/step:>10.1%}{gt:>9.2f}{n:>7}")
pd.DataFrame(rows).to_csv("backtest_output/heat_fm_regression.csv",index=False,encoding="utf-8-sig")
