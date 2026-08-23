# -*- coding: utf-8 -*-
"""生成日报里的影子策略对比文本（供 paper_tracker 调用）。
与 shadow_compare 同一套口径：从 paper 起始日到最新数据，逐影子候选算区间表现。
"""
import os
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data")
IDX = Path(os.environ.get("ETFS_REF_FILE") or r"F:\even-codex\panda\backtest\prices_2016.csv")
DAYS = 252; START = 20000.0

_stk = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
_stk = _stk.loc[:, _stk.count() >= 2400]
_spy = pd.read_csv(IDX, index_col=0, parse_dates=True)["SPY"]

def ml(x): return x.resample("ME").last()
def mom(px, p, k):
    m = ml(px); return m.shift(k) / m.shift(p + k) - 1
def mvol(px, l=60):
    d = px.pct_change(); return (d.rolling(l).std() * np.sqrt(DAYS)).resample("ME").last()
def weekly_scores(px, p=6, k=1):
    span_p = p*21; span_k = k*21
    mm = px.div(px.shift(span_p + span_k)).sub(1.0)
    s = pd.Series(np.arange(len(px)), index=px.index)
    last = s.groupby(px.index.to_period("W")).last().values.astype(int)
    return mm.iloc[last]

def run_fast(px, scores, top, cost_bps, vol_m=None, vol_target=None):
    cols = list(px.columns); M = len(cols); dr = px.pct_change().fillna(0.0)
    me = np.array(pd.DatetimeIndex(scores.index)); day = np.array(px.index)
    slot = np.searchsorted(me, day, side="right")-1
    exact = (slot>=0) & (day==me[np.clip(slot,0,len(me)-1)])
    slot = np.clip(slot-exact.astype(int),0,len(me)-1); T = len(me)
    Wmat = np.zeros((T,M))
    for s in range(1, T):
        d = pd.Timestamp(me[s]); w = np.zeros(M); sc = scores.loc[d].dropna()
        if len(sc) > 0:
            ta = list(sc.sort_values(ascending=False).index[:top]); ix = [cols.index(c) for c in ta]
            w[ix] = 1.0/len(ta)
        Wmat[s] = w
    Wdf = pd.DataFrame(Wmat[slot], index=px.index, columns=cols)
    g = (Wdf*dr).sum(axis=1).fillna(0.0)
    if vol_target:
        scale = np.ones(len(day))
        for s in range(1, T):
            ix2 = np.searchsorted(day, me[s], side="right")
            if ix2 >= len(day): continue
            look = g.loc[:pd.Timestamp(day[ix2])].iloc[-(61):-1].dropna()
            if len(look)>=21 and look.std(ddof=1)>0:
                scale[ix2] = min(1.0, vol_target/(look.std(ddof=1)*np.sqrt(DAYS)))
        for s in range(1, T):
            ix3 = np.searchsorted(day, me[s], side="right")
            if ix3 >= len(day): continue
            lo = day[ix3]; up = me[s+1] if s+1 < T else day[-1]
            sel = (day>=lo)&(day<=up)
            if np.any(sel): scale[sel] = scale[ix3]
        Wdf = Wdf.mul(scale, axis=0); g = (Wdf*dr).sum(axis=1).fillna(0.0)
    cost = pd.Series(0.0, index=g.index); prev = np.zeros(M)
    for s in range(1,T):
        ix = np.searchsorted(day, me[s], side="right")
        if ix >= len(day): continue
        wn = Wmat[slot[ix]]; to = np.abs(wn-prev).sum()/2.0
        cost.iloc[ix] = to*cost_bps/10000.0; prev = wn.copy()
    return (g - cost).clip(lower=-0.5)

def _summarize(seg):
    if len(seg)==0: return (0.0, 0.0, np.nan, 0.0, START)
    nav = (1+seg).cumprod()*START
    ret = nav.iloc[-1]/START-1
    vol = seg.std(ddof=1)*np.sqrt(DAYS)
    sh = ret/vol if vol>0 else np.nan
    dd = (nav/nav.cummax()-1).min()
    return (ret, vol, sh, float(dd), float(nav.iloc[-1]))

def build(start="2026-08-18"):
    try:
        start = pd.Timestamp(start)
    except Exception:
        start = pd.Timestamp("2026-08-18")
    lines = []
    lines.append("【影子策略对比 · 自 {} 至今】(仅供参考, 非实际持仓)".format(start.date()))
    spy_seg = _spy.pct_change().loc[start:].fillna(0.0)
    sr, sv, ss, sd, sf = _summarize(spy_seg)
    vm = mvol(_stk, 60)
    configs = [
        ("月频base(现paper)", mom(_stk,6,1), 10, 10, None),
        ("月频+vol25", mom(_stk,6,1), 10, 10, 0.25),
        ("周频base", weekly_scores(_stk,6,1), 10, 10, None),
        ("周频+vol25", weekly_scores(_stk,6,1), 10, 10, 0.25),
        ("top15(月)", mom(_stk,6,1), 15, 10, None),
        ("top20稳健(月)", mom(_stk,6,1), 20, 10, None),
        ("3m top10", mom(_stk,3,1), 10, 10, None),
        ("9m top10", mom(_stk,9,0), 10, 10, None),
    ]
    header = f"  {'候选':<16}{'收益':>8}{'波动':>7}{'夏普':>6}{'回撤':>7}{'期末':>10}"
    lines.append(header)
    for name, sc, top, cost, vt in configs:
        r = run_fast(_stk, sc, top, cost, vol_m=vm, vol_target=vt).loc[start:]
        a, v, s, d, f = _summarize(r)
        sh_txt = "  n/a" if (isinstance(s,float) and np.isnan(s)) else f"{s:5.2f}"
        lines.append(f"  {name:<16}{a*100:+7.2f}%{v*100:6.1f}%{sh_txt:>6}{d*100:6.1f}%{f:>10,.0f}")
    lines.append(f"  {'SPY':<16}{sr*100:+7.2f}%{sv*100:6.1f}%{'':>6}{sd*100:6.1f}%{sf:>10,.0f}")
    return "\n".join(lines)

if __name__ == "__main__":
    print(build())
