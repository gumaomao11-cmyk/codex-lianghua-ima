# -*- coding: utf-8 -*-
"""星球事件因子优化对比：纯动量 vs 原始事件 vs 优化事件(异常关注度+强信号+目标价加权)"""
import json, sys, collections
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA = Path(r"F:\even-codex\us-stock-data")
OUT  = Path(r"F:\even-codex\lianghua+IMA\backtest_output")

# ---- load prices ----
px = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
recent = px.loc[px.index >= pd.Timestamp("2025-01-01")]
px = px.loc[:, recent.notna().sum() >= 150]
ml  = px.resample("ME").last()
mom = (ml.shift(1)/ml.shift(7)-1.0).replace([np.inf,-np.inf], np.nan)
daily_ret = px.pct_change().fillna(0.0)
cols = list(px.columns)
rebal = list(ml.truncate("2025-09-30", px.index[-1]).index)

# ---- load events from cache ----
events = []
for line in (OUT/"zsxq_events_浑水调研Plus_cache.jsonl").read_text(encoding="utf-8", errors="ignore").splitlines():
    try:
        obj = json.loads(line)
        for s in obj.get("sig", []):
            events.append(s)
    except: pass
df = pd.DataFrame(events)
df["pdf_date"]   = pd.to_datetime(df["pdf_date"], errors="coerce")
df["direction"]  = pd.to_numeric(df["direction"], errors="coerce").fillna(0)
df["strength"]   = pd.to_numeric(df["strength"], errors="coerce").fillna(0.5)
df["pt_delta_pct"] = pd.to_numeric(df.get("pt_delta_pct",0), errors="coerce").fillna(0)
df = df.dropna(subset=["pdf_date"])
print(f"total events: {len(df)}, date range: {df.pdf_date.min().date()} ~ {df.pdf_date.max().date()}")

STRONG = {"upgrade","downgrade","reinitiate","price_up","price_down"}

def fac_score(fac, d, window, mode="original"):
    cur0 = d - pd.Timedelta(days=window)
    cur  = fac[(fac.pdf_date < d) & (fac.pdf_date >= cur0)]
    if mode == "original":
        if cur.empty: return pd.Series(dtype=float)
        g = cur.groupby("ticker").agg(n_pos=("direction", lambda x: (x>0).sum()),
                                       n_neg=("direction", lambda x: (x<0).sum()))
        sc = g["n_pos"] - g["n_neg"]
        if sc.std() and sc.std() > 0: return (sc-sc.mean())/sc.std()
        return sc - sc.mean()

    elif mode == "optimized":
        # 1) strong signal filter
        cur_s = cur[cur.action.isin(STRONG)] if "action" in cur.columns else cur
        if cur_s.empty: return pd.Series(dtype=float)
        # 2) abnormal attention: current mentions vs baseline (previous window)
        prev = fac[(fac.pdf_date < cur0) & (fac.pdf_date >= d - pd.Timedelta(days=2*window))]
        prev_s = prev[prev.action.isin(STRONG)] if "action" in prev.columns else prev
        cur_cnt  = cur_s.groupby("ticker").size()
        prev_cnt = prev_s.groupby("ticker").size()
        all_tk = cur_cnt.index.union(prev_cnt.index)
        prev_bl = prev_cnt.reindex(all_tk, fill_value=0)
        abnormal = (cur_cnt.reindex(all_tk, fill_value=0) - prev_bl) / (prev_bl + 1.0)
        # 3) weighted signal: direction * strength * (1 + |pt_delta|/100)
        cw = cur_s.copy()
        cw["weight"] = cw["direction"] * cw["strength"] * (1 + cw["pt_delta_pct"].abs()/100.0)
        sig = cw.groupby("ticker")["weight"].sum()
        combined = abnormal + sig.reindex(all_tk, fill_value=0)
        if combined.std() and combined.std() > 0: return (combined-combined.mean())/combined.std()
        return combined - combined.mean()

def monthly(window, lam, mode="original", cost_bps=5):
    rets = {}
    for i, d in enumerate(rebal):
        if d not in mom.index: continue
        m = mom.loc[d].dropna()
        if m.empty: continue
        score = m.rank(pct=True)
        fs = fac_score(df, d, window, mode)
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
    if len(s)==0: return {"n":0,"mean":np.nan,"vol":np.nan,"sharpe":np.nan}
    if len(s)==1: return {"n":1,"mean":float(s.mean()),"vol":np.nan,"sharpe":np.nan}
    mr=float(s.mean()); v=float(s.std(ddof=0))
    return {"n":len(s),"mean":mr*100,"vol":v*100,"sharpe":float(mr/v) if v>0 else np.nan}

print("\n" + "="*60)
print("星球事件因子优化对比")
print("="*60)

# pure momentum
mom_ret = monthly(60, 0, "original")
sm = summ(mom_ret)
print(f"\n[1] 纯动量: 月均 {sm['mean']:.2f}% 夏普 {sm['sharpe']:.2f} n={sm['n']}")

# original event
print(f"\n[2] 原始事件因子 (n_pos - n_neg)")
best_orig = None
for w in [30,60]:
    for lam in [0.7,1.2]:
        r = monthly(w, lam, "original")
        s = summ(r)
        print(f"    w={w} lam={lam}: 月均 {s['mean']:.2f}% 夏普 {s['sharpe']:.2f}")
        if best_orig is None or (s['sharpe'] and (best_orig[1]['sharpe'] is np.nan or s['sharpe'] > best_orig[1]['sharpe'])):
            best_orig = ((w,lam), s)

# optimized event
print(f"\n[3] 优化事件因子 (异常关注度 + 强信号 + 目标价加权)")
best_opt = None
for w in [30,60]:
    for lam in [0.7,1.2]:
        r = monthly(w, lam, "optimized")
        s = summ(r)
        print(f"    w={w} lam={lam}: 月均 {s['mean']:.2f}% 夏普 {s['sharpe']:.2f}")
        if best_opt is None or (s['sharpe'] and (best_opt[1]['sharpe'] is np.nan or s['sharpe'] > best_opt[1]['sharpe'])):
            best_opt = ((w,lam), s)

# strong signal stats
df_strong = df[df.action.isin(STRONG)]
print(f"\n--- 强信号统计 ---")
print(f"总事件: {len(df)}, 强信号: {len(df_strong)} ({100*len(df_strong)/max(len(df),1):.0f}%)")
print(f"动作分布:")
for a, c in df_strong.action.value_counts().items():
    print(f"  {a}: {c}")

# recent period selection comparison
print(f"\n--- 07-31 调仓选股对比 ---")
d = pd.Timestamp("2026-07-31")
if d in mom.index:
    m = mom.loc[d].dropna()
    base_sel = m.rank(pct=True).sort_values(ascending=False).index[:10]
    orig_sc = m.rank(pct=True) + 1.2 * fac_score(df, d, 60, "original").reindex(m.index).fillna(0)
    opt_sc   = m.rank(pct=True) + 1.2 * fac_score(df, d, 60, "optimized").reindex(m.index).fillna(0)
    orig_sel = orig_sc.sort_values(ascending=False).index[:10]
    opt_sel   = opt_sc.sort_values(ascending=False).index[:10]
    print(f"纯动量:   {', '.join(base_sel)}")
    print(f"原始事件: {', '.join(orig_sel)}")
    print(f"优化事件: {', '.join(opt_sel)}")
    print(f"  原始 vs 动量 换入: {', '.join(sorted(set(orig_sel)-set(base_sel))) or '无'}")
    print(f"  优化 vs 动量 换入: {', '.join(sorted(set(opt_sel)-set(base_sel))) or '无'}")

# also check 06-30
d2 = pd.Timestamp("2026-06-30")
if d2 in mom.index:
    m2 = mom.loc[d2].dropna()
    base2 = m2.rank(pct=True).sort_values(ascending=False).index[:10]
    opt2_sc = m2.rank(pct=True) + 1.2 * fac_score(df, d2, 60, "optimized").reindex(m2.index).fillna(0)
    opt2_sel = opt2_sc.sort_values(ascending=False).index[:10]
    print(f"\n--- 06-30 调仓 ---")
    print(f"纯动量:   {', '.join(base2)}")
    print(f"优化事件: {', '.join(opt2_sel)}")
    print(f"  优化 vs 动量 换入: {', '.join(sorted(set(opt2_sel)-set(base2))) or '无'}")

print("\n="*30)
print("结论:")
if best_opt and best_orig:
    print(f"  纯动量夏普:       {sm['sharpe']:.2f}")
    print(f"  原始事件最优:     {best_orig[1]['sharpe']:.2f} (w={best_orig[0][0]},lam={best_orig[0][1]})")
    print(f"  优化事件最优:     {best_opt[1]['sharpe']:.2f} (w={best_opt[0][0]},lam={best_opt[0][1]})")
    if best_opt[1]['sharpe'] is not np.nan and best_orig[1]['sharpe'] is not np.nan:
        if best_opt[1]['sharpe'] > best_orig[1]['sharpe']:
            print(f"  -> 优化版优于原始版 (+{best_opt[1]['sharpe']-best_orig[1]['sharpe']:.2f})")
        else:
            print(f"  -> 优化版未优于原始版 ({best_opt[1]['sharpe']-best_orig[1]['sharpe']:.2f})")
