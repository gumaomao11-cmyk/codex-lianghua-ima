# -*- coding: utf-8 -*-
"""Attribution against the SAME pool: strategy vs discussed-pool EW vs full universe EW."""
import pandas as pd, numpy as np, duckdb

perf = pd.read_csv("backtest_output/walkforward_v4_dynamic_results.csv")
perf["date"] = pd.to_datetime(perf["date"])
pools = pd.read_csv("backtest_output/pool_benchmarks.csv", index_col=0)
pools.index = pd.to_datetime(pools.index)

def stats(s, rf=0.0):
    s = s.dropna(); n = len(s)
    if n < 5: return dict(days=n)
    m, sd = s.mean(), s.std(ddof=1)
    eq = (1+s).cumprod()
    return dict(days=n, ann=(1+s).prod()**(252/n)-1, sharpe=m/sd*np.sqrt(252) if sd>0 else np.nan,
                cum=eq.iloc[-1]-1, mdd=(eq/eq.cummax()-1).min())

bench = pools["discussed_pool_EW_rolling60d"].dropna()
full  = pools["full_universe_EW"].dropna()
qqq   = pools["QQQ"].dropna() if "QQQ" in pools else None

print(f"{'series':<34}{'days':>6}{'ann':>10}{'sharpe':>8}{'cum':>10}{'maxDD':>9}")
rows = {}
for route in ["A","B"]:
    s = perf[perf.route==route].set_index("date")["xgb_dynamic"].dropna()
    rows[f"strategy_{route}"] = s
rows["POOL_EW (discussed 38)"] = bench
rows["FULL_EW (515)"] = full
if qqq is not None: rows["QQQ"] = qqq
for k,s in rows.items():
    st = stats(s)
    print(f"{k:<34}{st['days']:>6}{st.get('ann',np.nan):>9.1%}{st.get('sharpe',np.nan):>8.2f}{st.get('cum',np.nan):>9.1%}{st.get('mdd',np.nan):>9.1%}")

print("\n=== decomposition: where does the return come from? ===")
for route in ["A","B"]:
    s = perf[perf.route==route].set_index("date")["xgb_dynamic"].dropna()
    idx = s.index.intersection(bench.index)
    st, bm, fu = s.loc[idx], bench.loc[idx], full.loc[idx]
    # stock selection = strategy - pool
    sel = (st - bm)
    pool = (bm - fu)
    tsel = sel.mean()/(sel.std(ddof=1)/np.sqrt(len(sel)))
    tpool = pool.mean()/(pool.std(ddof=1)/np.sqrt(len(pool)))
    # beta vs pool
    beta = np.cov(st, bm)[0,1]/np.var(bm, ddof=1)
    corr = np.corrcoef(st, bm)[0,1]
    alpha_d = st.mean() - beta*bm.mean()
    ir = sel.mean()/sel.std(ddof=1)*np.sqrt(252)
    print(f"\n[route {route}]  n={len(idx)}")
    print(f"  strategy ann         = {(1+st).prod()**(252/len(st))-1:+.1%}")
    print(f"  POOL effect ann      = {pool.mean()*252:+.1%}   t={tpool:+.2f}   <- 'which stocks are discussed'")
    print(f"  STOCK-SEL ann        = {sel.mean()*252:+.1%}   t={tsel:+.2f}   IR={ir:+.2f}  <- 'LLM picking within pool'")
    print(f"  vs POOL: beta={beta:.2f} corr={corr:.2f}  ann_alpha_vs_pool={alpha_d*252:+.1%}")
