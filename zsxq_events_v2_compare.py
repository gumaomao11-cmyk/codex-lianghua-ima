# -*- coding: utf-8 -*-
"""星球事件因子 v2 优化对比：纯动量 vs 原始 vs 异常关注度 vs 平衡加权(log+方向加权)"""
import json, sys, collections, math
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA = Path(r"F:\even-codex\us-stock-data")
OUT  = Path(r"F:\even-codex\lianghua+IMA\backtest_output")

px = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
recent = px.loc[px.index >= pd.Timestamp("2025-01-01")]
px = px.loc[:, recent.notna().sum() >= 150]
ml  = px.resample("ME").last()
mom = (ml.shift(1)/ml.shift(7)-1.0).replace([np.inf,-np.inf], np.nan)
daily_ret = px.pct_change().fillna(0.0)
cols = list(px.columns)
rebal = list(ml.truncate("2025-09-30", px.index[-1]).index)

# load events
events = []
for line in (OUT/"zsxq_events_浑水调研Plus_cache.jsonl").read_text(encoding="utf-8", errors="ignore").splitlines():
    try:
        for s in json.loads(line).get("sig", []): events.append(s)
    except: pass
df = pd.DataFrame(events)
df["pdf_date"]   = pd.to_datetime(df["pdf_date"], errors="coerce")
df["direction"]  = pd.to_numeric(df["direction"], errors="coerce").fillna(0)
df["strength"]   = pd.to_numeric(df["strength"], errors="coerce").fillna(0.5)
df["pt_delta_pct"] = pd.to_numeric(df.get("pt_delta_pct",0), errors="coerce").fillna(0)
df = df.dropna(subset=["pdf_date"])

STRONG = {"upgrade","downgrade","reinitiate","price_up","price_down"}

def fac_score(fac, d, window, mode):
    cur0 = d - pd.Timedelta(days=window)
    cur = fac[(fac.pdf_date < d) & (fac.pdf_date >= cur0)]
    if cur.empty: return pd.Series(dtype=float)

    if mode == "original":
        g = cur.groupby("ticker").agg(n_pos=("direction", lambda x: (x>0).sum()),
                                      n_neg=("direction", lambda x: (x<0).sum()))
        sc = g["n_pos"] - g["n_neg"]
    elif mode == "abnormal":
        prev = fac[(fac.pdf_date < cur0) & (fac.pdf_date >= d - pd.Timedelta(days=2*window))]
        cur_cnt = cur.groupby("ticker").size()
        prev_cnt = prev.groupby("ticker").size()
        all_tk = cur_cnt.index.union(prev_cnt.index)
        prev_bl = prev_cnt.reindex(all_tk, fill_value=0)
        abnormal = (cur_cnt.reindex(all_tk, fill_value=0) - prev_bl) / (prev_bl + 1.0)
        cw = cur.copy()
        cw["w"] = cw["direction"] * cw["strength"]
        sig = cw.groupby("ticker")["w"].sum()
        sc = abnormal + sig.reindex(all_tk, fill_value=0)
    elif mode == "balanced":
        # all events, weighted by direction*strength, log-transformed mention count
        cw = cur.copy()
        cw["w"] = cw["direction"] * cw["strength"] * (1 + cw["pt_delta_pct"].abs()/100.0)
        g = cw.groupby("ticker").agg(mention_count=("ticker","size"), weighted_sig=("w","sum"))
        # log transform mention count to reduce NVDA dominance
        log_mentions = np.log1p(g["mention_count"])
        sc = g["weighted_sig"] + 0.3 * log_mentions
    elif mode == "balanced_strong":
        # only strong signals + direction weighted + abnormal attention
        cur_s = cur[cur.action.isin(STRONG)]
        if cur_s.empty: return pd.Series(dtype=float)
        prev = fac[(fac.pdf_date < cur0) & (fac.pdf_date >= d - pd.Timedelta(days=2*window))]
        prev_s = prev[prev.action.isin(STRONG)] if "action" in prev.columns else prev
        cur_cnt = cur_s.groupby("ticker").size()
        prev_cnt = prev_s.groupby("ticker").size()
        all_tk = cur_cnt.index.union(prev_cnt.index)
        prev_bl = prev_cnt.reindex(all_tk, fill_value=0)
        abnormal = (cur_cnt.reindex(all_tk, fill_value=0) - prev_bl) / (prev_bl + 1.0)
        cw = cur_s.copy()
        cw["w"] = cw["direction"] * cw["strength"] * (1 + cw["pt_delta_pct"].abs()/100.0)
        sig = cw.groupby("ticker")["w"].sum()
        sc = abnormal + sig.reindex(all_tk, fill_value=0)
    else:
        return pd.Series(dtype=float)
    sc = sc.dropna()
    if sc.empty: return pd.Series(dtype=float)
    if sc.std() and sc.std() > 0: return (sc - sc.mean()) / sc.std()
    return sc - sc.mean()

def monthly(window, lam, mode, cost_bps=5):
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
    if len(s)==0: return {"n":0,"mean":np.nan,"sharpe":np.nan}
    if len(s)==1: return {"n":1,"mean":float(s.mean()),"sharpe":np.nan}
    mr=float(s.mean()); v=float(s.std(ddof=0))
    return {"n":len(s),"mean":mr*100,"sharpe":float(mr/v) if v>0 else np.nan}

print("="*60)
print("星球事件因子 v2 优化对比")
print(f"事件数: {len(df)}, 日期: {df.pdf_date.min().date()}~{df.pdf_date.max().date()}")
print(f"强信号: {len(df[df.action.isin(STRONG)])} / {len(df)} ({100*len(df[df.action.isin(STRONG)])/len(df):.1f}%)")
print("="*60)

modes = [("original","原始(n_pos-n_neg)"), ("abnormal","异常关注度+方向加权"),
         ("balanced","平衡(log+方向加权+pt_delta)"), ("balanced_strong","强信号+异常+pt_delta")]

mom_ret = monthly(60, 0, "original")
sm = summ(mom_ret)
print(f"\n[0] 纯动量: 月均{sm['mean']:.2f}% 夏普{sm['sharpe']:.2f}")

results = {}
for mode, label in modes:
    print(f"\n--- {label} ---")
    best = None
    for w in [30, 60]:
        for lam in [0.3, 0.7, 1.2]:
            r = monthly(w, lam, mode)
            s = summ(r)
            print(f"  w={w} lam={lam}: 月均{s['mean']:.2f}% 夏普{s['sharpe']:.2f}")
            if best is None or (s['sharpe'] is not np.nan and (best[1]['sharpe'] is np.nan or s['sharpe'] > best[1]['sharpe'])):
                best = ((w,lam), s)
    results[mode] = best
    print(f"  -> 最优: w={best[0][0]} lam={best[0][1]} 夏普{best[1]['sharpe']:.2f}")

# selection comparison
print(f"\n{'='*60}")
print("07-31 选股对比")
d = pd.Timestamp("2026-07-31")
if d in mom.index:
    m = mom.loc[d].dropna()
    base = m.rank(pct=True).sort_values(ascending=False).index[:10]
    print(f"纯动量: {', '.join(base)}")
    for mode, label in modes:
        fs = fac_score(df, d, 60, mode)
        if len(fs) == 0:
            print(f"{label}: (无因子覆盖)")
            continue
        sc = m.rank(pct=True) + 0.7 * fs.reindex(m.index).fillna(0)
        sel = sc.sort_values(ascending=False).index[:10]
        diff = sorted(set(sel) - set(base))
        print(f"{label}: {', '.join(sel)}  [换入: {', '.join(diff) if diff else '无'}]")

print(f"\n{'='*60}")
print("结论:")
print(f"  纯动量:            夏普 {sm['sharpe']:.2f}")
for mode, label in modes:
    print(f"  {label}: 夏普 {results[mode][1]['sharpe']:.2f}")
