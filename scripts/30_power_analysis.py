# -*- coding: utf-8 -*-
"""
CRITICAL CHECK before archiving: is the binding constraint really "20 independent
monthly observations"? For the 1-DAY horizon, non-overlapping = 411 independent
daily observations, NOT 20. If t is still 1.57 with n=411, the limit is EFFECT SIZE
relative to noise, not degrees of freedom. That changes what "more data" would buy.
Also: how much of the raw 56,532 posts were actually extracted?
"""
import json, duckdb, pandas as pd, numpy as np, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")

print("="*80); print("1. EXTRACTION COVERAGE: raw posts vs LLM-processed"); print("="*80)
raw=json.load(open("backtest_output/zsxq_group_48418411254128_web.json",encoding="utf-8"))
if isinstance(raw,dict):
    for k in ("topics","data","items"):
        if k in raw: raw=raw[k]; break
rawm=Counter(str(t.get("create_time"))[:7] for t in raw if t.get("create_time"))
done=Counter()
for f in ["zsxq_19_26_granular_cache.jsonl","zsxq_v3_clean_sample.jsonl","zsxq_19_26_events_cache.jsonl"]:
    try:
        for line in open("backtest_output/"+f,encoding="utf-8",errors="replace"):
            line=line.strip()
            if not line: continue
            try: o=json.loads(line)
            except: continue
            ct=o.get("create_time")
            if ct: done[str(ct)[:7]]+=1
    except FileNotFoundError: pass
months=sorted(set(rawm)|set(done))
print(f"{'month':<10}{'raw':>8}{'extracted':>11}{'pct':>8}")
tr=te=0
for m in months:
    r,e=rawm.get(m,0),done.get(m,0); tr+=r; te+=e
    print(f"{m:<10}{r:>8}{e:>11}{(e/r if r else 0):>7.0%}")
print(f"{'TOTAL':<10}{tr:>8}{te:>11}{te/tr:>7.0%}")
print(f"\n=> {tr-te} posts ({1-te/tr:.0%}) never sent to LLM, ALL within the same 20 months.")
print("=> Processing them adds CROSS-SECTIONAL density, NOT time-series degrees of freedom.")

print()
print("="*80); print("2. What is the ACTUAL number of independent observations?"); print("="*80)
print("For horizon h, non-overlapping obs = n_dates / h:")
for h,nd in [("ret_1d",411),("ret_5d",407),("ret_21d",391)]:
    step=int(h.split("_")[1].replace("d",""))
    print(f"  {h:<9} n_dates={nd:>4}  independent obs = {nd//step:>4}")
print("\n=> The '20 independent observations' limit applies ONLY to ret_21d.")
print("=> For ret_1d we already have 411 independent daily obs, and t is still 1.57.")

print()
print("="*80); print("3. POWER ANALYSIS: what would it take to reach t=1.96?"); print("="*80)
con=duckdb.connect()
d=con.execute("""select date,ticker,factor_clean_alpha,ret_1d,ln_mcap
                 from 'data/duckdb/aligned_v2_a.parquet' where date>='2025-01-01'""").df()
d["date"]=pd.to_datetime(d["date"]); d["iscov"]=d.factor_clean_alpha.notna().astype(float)
d=d.sort_values(["ticker","date"])
d["cov_roll"]=d.groupby("ticker")["iscov"].transform(lambda s:s.shift(1).rolling(60,min_periods=1).max()).fillna(0)
# simple daily spread: covered minus uncovered, equal weight
a=d[d.cov_roll==1].groupby("date").ret_1d.mean(); b=d[d.cov_roll==0].groupby("date").ret_1d.mean()
i=a.index.intersection(b.index); x=(a.loc[i]-b.loc[i]).dropna()
m,s,n=x.mean(),x.std(ddof=1),len(x)
t=m/(s/np.sqrt(n))
print(f"daily coverage spread: mean={m*1e4:.2f}bps  sd={s*1e4:.1f}bps  n={n}  t={t:.2f}")
print(f"  annualized: {m*252:+.1%}   info ratio: {m/s*np.sqrt(252):+.2f}")
need=(1.96/t)**2
print(f"\nTo reach t=1.96 at the SAME effect size, need n = {n} x {need:.2f} = {int(n*need)} days")
print(f"  = {n*need/252:.1f} years of data  (currently {n/252:.1f} years)")
print(f"  => need ~{(n*need-n)/252:.1f} MORE years. Source data does not exist (2025-01 is the start).")
print(f"\nALTERNATIVE: raise the effect size instead of n.")
print(f"  Required effect at n={n} for t=1.96: {1.96*s/np.sqrt(n)*252:+.1%}/yr vs current {m*252:+.1%}/yr")
print(f"  => need effect x{1.96/t:.2f}. Denser extraction could plausibly reduce factor noise,")
print(f"     but there is no evidence it would multiply the effect by {1.96/t:.2f}.")
