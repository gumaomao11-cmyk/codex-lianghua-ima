# -*- coding: utf-8 -*-
"""严格事件因子 vs 原始事件因子 vs 纯动量 回测对比
读取 strict_cache 和原始 cache，自动过滤非美股 ticker。
"""
import json, sys, collections
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA = Path(r"F:\even-codex\us-stock-data")
OUT  = Path(r"F:\even-codex\lianghua+IMA\backtest_output")

px = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
recent = px.loc[px.index >= pd.Timestamp("2025-01-01")]
px = px.loc[:, recent.notna().sum() >= 150]
valid_tickers = set(px.columns)
ml  = px.resample("ME").last()
mom = (ml.shift(1)/ml.shift(7)-1.0).replace([np.inf,-np.inf], np.nan)
daily_ret = px.pct_change().fillna(0.0)
cols = list(px.columns)
rebal = list(ml.truncate("2025-09-30", px.index[-1]).index)

def load_cache(name):
    events = []
    if name:
        p = OUT / f"zsxq_events_浑水调研Plus_{name}_cache.jsonl"
    else:
        p = OUT / "zsxq_events_浑水调研Plus_cache.jsonl"
    if not p.exists(): return pd.DataFrame()
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            for s in json.loads(line).get("sig", []): events.append(s)
        except: pass
    if not events: return pd.DataFrame()
    df = pd.DataFrame(events)
    df["pdf_date"] = pd.to_datetime(df["pdf_date"], errors="coerce")
    df["direction"] = pd.to_numeric(df["direction"], errors="coerce").fillna(0)
    df["strength"]  = pd.to_numeric(df["strength"], errors="coerce").fillna(0.5)
    df["pt_delta_pct"] = pd.to_numeric(df.get("pt_delta_pct",0), errors="coerce").fillna(0)
    df = df.dropna(subset=["pdf_date"])
    # filter to valid US tickers only
    before = len(df)
    df = df[df["ticker"].str.strip().str.upper().isin(valid_tickers)].copy()
    df["ticker"] = df["ticker"].str.strip().str.upper()
    print(f"  {name}: {len(events)} events -> {len(df)} after US-filter (dropped {before-len(df)})")
    return df

print("loading caches...")
df_loose  = load_cache("")        # original loose cache
df_strict = load_cache("strict")  # strict cache
STRONG = {"upgrade","downgrade","reinitiate","price_up","price_down"}

def fac_score(fac, d, window, mode="default"):
    if fac.empty: return pd.Series(dtype=float)
    cur0 = d - pd.Timedelta(days=window)
    cur = fac[(fac.pdf_date < d) & (fac.pdf_date >= cur0)]
    if cur.empty: return pd.Series(dtype=float)
    if mode == "default":
        cw = cur.copy()
        cw["w"] = cw["direction"] * cw["strength"] * (1 + cw["pt_delta_pct"].abs()/100.0)
        sc = cw.groupby("ticker")["w"].sum()
    elif mode == "count":
        g = cur.groupby("ticker").agg(n_pos=("direction", lambda x: (x>0).sum()),
                                      n_neg=("direction", lambda x: (x<0).sum()))
        sc = g["n_pos"] - g["n_neg"]
    sc = sc.dropna()
    if sc.empty: return pd.Series(dtype=float)
    if sc.std() and sc.std() > 0: return (sc - sc.mean()) / sc.std()
    return sc - sc.mean()

def monthly(window, lam, fac, mode="default", cost_bps=5):
    rets = {}
    for i, d in enumerate(rebal):
        if d not in mom.index: continue
        m = mom.loc[d].dropna()
        if m.empty: continue
        score = m.rank(pct=True)
        fs = fac_score(fac, d, window, mode)
        if len(fs): score = score + lam * fs.reindex(m.index).fillna(0.0)
        sel = score.sort_values(ascending=False).index[:10]
        if len(sel)==0: continue
        w = pd.Series(0.0, index=cols); w[sel] = 1/len(sel)
        end = rebal[i+1] if i+1<len(rebal) else px.index[-1]
        days = px.index[(px.index>d) & (px.index<=end)]
        if len(days)==0: continue
        r = (w.reindex(cols).values * daily_ret.loc[days,:].values).sum(axis=1)
        r = r - ((w-pd.Series(0.0,index=cols)).abs().sum()/2.0)*cost_bps/10000.0
        rets[d] = float(r.sum())
    return pd.Series(rets)

def summ(s):
    s = s.dropna()
    if len(s)==0: return {"n":0,"mean":np.nan,"sharpe":np.nan}
    if len(s)==1: return {"n":1,"mean":float(s.mean()),"sharpe":np.nan}
    mr=float(s.mean()); v=float(s.std(ddof=0))
    return {"n":len(s),"mean":mr*100,"sharpe":float(mr/v) if v>0 else np.nan}

print(f"\n{'='*60}")
print("严格 vs 原始事件因子对比")
print(f"{'='*60}")
if not df_loose.empty:
    print(f"原始(loose): {len(df_loose)} events, 强信号 {len(df_loose[df_loose.action.isin(STRONG)])}")
    print(f"  action分布: {dict(df_loose.action.value_counts())}")
if not df_strict.empty:
    print(f"严格(strict): {len(df_strict)} events, 全部是强信号")
    print(f"  action分布: {dict(df_strict.action.value_counts())}")

# pure momentum
mom_ret = monthly(60, 0, pd.DataFrame(), "default")
sm = summ(mom_ret)
print(f"\n[0] 纯动量: 月均{sm['mean']:.2f}% 夏普{sm['sharpe']:.2f} n={sm['n']}")

# original loose factor
if not df_loose.empty:
    print(f"\n[1] 原始事件因子(loose)")
    for w in [30,60]:
        for lam in [0.3,0.7,1.2]:
            r = monthly(w, lam, df_loose, "default")
            s = summ(r)
            print(f"  w={w} lam={lam}: 月均{s['mean']:.2f}% 夏普{s['sharpe']:.2f}")

# strict factor
if not df_strict.empty:
    print(f"\n[2] 严格事件因子(strict)")
    for w in [30,60]:
        for lam in [0.3,0.7,1.2]:
            r = monthly(w, lam, df_strict, "default")
            s = summ(r)
            print(f"  w={w} lam={lam}: 月均{s['mean']:.2f}% 夏普{s['sharpe']:.2f}")

# strict + count mode
if not df_strict.empty:
    print(f"\n[3] 严格事件(count模式)")
    for w in [30,60]:
        for lam in [0.3,0.7,1.2]:
            r = monthly(w, lam, df_strict, "count")
            s = summ(r)
            print(f"  w={w} lam={lam}: 月均{s['mean']:.2f}% 夏普{s['sharpe']:.2f}")

# selection comparison
print(f"\n{'='*60}")
print("07-31 选股对比")
d = pd.Timestamp("2026-07-31")
if d in mom.index:
    m = mom.loc[d].dropna()
    base = m.rank(pct=True).sort_values(ascending=False).index[:10]
    print(f"纯动量: {', '.join(base)}")
    for fac, label in [(df_loose,"loose"), (df_strict,"strict")]:
        if fac.empty: continue
        for mode in ["default","count"]:
            fs = fac_score(fac, d, 60, mode)
            if len(fs) == 0:
                print(f"{label}({mode}): (无因子覆盖)")
                continue
            sc = m.rank(pct=True) + 0.7 * fs.reindex(m.index).fillna(0)
            sel = sc.sort_values(ascending=False).index[:10]
            diff = sorted(set(sel) - set(base))
            print(f"{label}({mode}): {', '.join(sel)}  [换入: {', '.join(diff) if diff else '无'}]")

print(f"\n{'='*60}")
print("结论:")
print(f"  纯动量: {sm['sharpe']:.2f}")
if not df_strict.empty:
    best_strict = None
    for w in [30,60]:
        for lam in [0.3,0.7,1.2]:
            r = monthly(w, lam, df_strict, "default")
            s = summ(r)
            if best_strict is None or (s['sharpe'] is not np.nan and (best_strict['sharpe'] is np.nan or s['sharpe'] > best_strict['sharpe'])):
                best_strict = s
    if best_strict:
        print(f"  严格因子最优: {best_strict['sharpe']:.2f}")
        diff = best_strict['sharpe'] - sm['sharpe'] if (best_strict['sharpe'] is not np.nan and sm['sharpe'] is not np.nan) else None
        if diff is not None:
            print(f"  差异: {diff:+.2f} ({'改善' if diff > 0 else '无改善' if diff == 0 else '变差'})")
else:
    print("  严格因子: 无数据")

