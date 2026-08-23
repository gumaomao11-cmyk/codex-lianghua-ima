# -*- coding: utf-8 -*-
"""扫描：灵活调仓时机 + 灵活仓位，能否提高夏普 / 降低回撤。
统一数据源：F:\\even-codex\\us-stock-data
基线：月频 accel top10 等权（当前主策略）
事件驱动：周频评估，个股跌出阈值才卖/升入阈值才买（不机械每月全换）
灵活仓位：等权 / 排名加权 / 分数加权 / 波动率倒数 / vol25 缩放
"""
import os
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data")
OUT  = Path(__file__).resolve().parent / "backtest_output"
DAYS=252; START=20000.0; COST_BPS=10; TOP=10

px = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
px = px.loc[:, px.count() >= 2400]
idx=px.index; cols=list(px.columns); idpos={d:i for i,d in enumerate(idx)}
dr = px.pct_change().fillna(0.0)
vol60 = px.pct_change().rolling(60, min_periods=30).std().mul(np.sqrt(DAYS))
mom = px.div(px.shift(147)).sub(px.shift(21).div(px.shift(147)))  # placeholder, replaced below
mom = px.shift(21).div(px.shift(147)) - 1.0       # 6m-skip1 日线近似
acc = px.div(px.shift(21)) - 1.0                  # 近1月
score = 0.5*mom + 0.5*acc

def eval_days(period):
    s = pd.Series(np.arange(len(idx)), index=idx)
    g = "M" if period=="M" else "W"
    last = s.groupby(idx.to_period(g)).last().astype(int)
    return last.tolist()

def pick_sizing(held, sz_kind):
    if sz_kind == "equal":
        w = np.full(len(held), 1.0/len(held))
    elif sz_kind == "rankw":
        rk = pd.Series(score.iloc[day_idx].reindex(held).rank(ascending=False)) if False else None
        # 用传入时的排名计算较复杂，这里用等权的 rank 加权简化：按当前 held 列表顺序（已按分数降序）
        pos = np.arange(len(held), 0, -1.0); w = pos/pos.sum()
    elif sz_kind == "scorew":
        v = score.iloc[day_idx].reindex(held)
        v = v.fillna(0.0); w = v.clip(lower=0.05)
        if w.sum() <= 0: w = np.ones(len(held))
        w = w/w.sum()
    elif sz_kind == "volpar":
        v = vol60.iloc[day_idx].reindex(held)
        inv = (1.0/v).fillna(1.0)
        w = inv/inv.sum()
    else:
        w = np.full(len(held), 1.0/len(held))
    return w

def run(period, mode, sz="equal", min_change=2, exit_rank=15, entry_rank=8, vol_target=None, acc_floor=None):
    days = [d for d in eval_days(period) if d > 147 and d+1 < len(idx)]
    held = []; w = np.array([])
    prev_w = pd.Series(0.0, index=cols)
    ret = pd.Series(0.0, index=idx); cost = pd.Series(0.0, index=idx)
    global day_idx
    for k, d in enumerate(days):
        day_idx = d
        sc = score.iloc[d]
        ideal = list(sc.sort_values(ascending=False).index[:TOP])
        if mode == "rotate":
            if not held:
                new_held = ideal
            else:
                ranks = sc.rank(ascending=False)
                flt = [(t, ranks.get(t,1e9)) for t in held]
                keep = [t for t,r in flt if float(r) <= exit_rank and (acc_floor is None or sc.get(t,float("-inf")) > acc_floor)]
                n_remove = TOP - len(keep)
                if n_remove > 0:
                    cand = [t for t in ideal if t not in keep][:n_remove]
                    new_held = keep + cand
                else:
                    new_held = keep
        else:  # change
            if not held or len(set(ideal)-set(held)) >= min_change:
                new_held = ideal
            else:
                new_held = held
        if new_held:
            new_w_vals = pick_sizing(new_held, sz)
        else:
            new_w_vals = np.array([])
        # vol 缩放（简化：按当前持仓平均波动率把仓位压到目标）
        if vol_target and len(new_held):
            vs = vol60.iloc[d].reindex(new_held).replace([np.inf,-np.inf], np.nan).fillna(np.nan)
            mv = vs.dropna()
            if len(mv) > 0:
                fac = min(1.0, vol_target / mv.mean())
                new_w_vals = new_w_vals * fac
        new_w = pd.Series(0.0, index=cols)
        if len(new_held):
            new_w.loc[new_held] = new_w_vals
        hs = d + 1; se = days[k+1] if k+1 < len(days) else len(idx)-1
        if se <= hs: se = hs + 1
        cost.iloc[hs] += (new_w - prev_w).abs().sum()/2.0 * COST_BPS/10000.0
        if len(new_held):
            seg = dr.iloc[hs:se+1, [cols.index(t) for t in new_held]]
            ret.iloc[hs:se+1] += (seg * new_w_vals).sum(axis=1)
        prev_w = new_w.copy(); held = new_held; w = new_w_vals
    return (ret - cost).clip(lower=-0.5)

def metrics(r):
    nav=(1+r).cumprod()*START; ann=(nav.iloc[-1]/START)**(DAYS/len(nav))-1
    vol=r.std(ddof=1)*np.sqrt(DAYS); sh=ann/vol if vol>0 else np.nan
    mdd=(nav/nav.cummax()-1).min(); calmar = ann/abs(mdd) if mdd<0 else np.nan
    active=(r.abs()>1e-6).mean()
    return ann, vol, sh, mdd, calmar, active, nav.iloc[-1]

oos = idx >= pd.Timestamp("2022-01-01")
rows=[]
configs = [
    ("月频等权 top10 (基线)", "M", "change", "equal", dict(min_change=1)),
    ("月频 仅名单变动>=2才换", "M", "change", "equal", dict(min_change=2)),
    ("月频 仅名单变动>=3才换", "M", "change", "equal", dict(min_change=3)),
    ("周频等权 top10", "W", "change", "equal", dict(min_change=1)),
    ("周频 变动>=2才换", "W", "change", "equal", dict(min_change=2)),
    ("事件:跌出前15卖/进前8买", "W", "rotate", "equal", dict(exit_rank=15, entry_rank=8)),
    ("事件:加速为负不持有", "W", "rotate", "equal", dict(exit_rank=15, entry_rank=8, acc_floor=0.0)),
    ("周频 排名加权", "W", "change", "rankw", dict(min_change=1)),
    ("周频 分数加权", "W", "change", "scorew", dict(min_change=1)),
    ("周频 波动率倒数加权", "W", "change", "volpar", dict(min_change=1)),
    ("事件+vol25", "W", "rotate", "equal", dict(exit_rank=15, entry_rank=8, vol_target=0.25)),
    ("周频变动>=2+vol25", "W", "change", "equal", dict(min_change=2, vol_target=0.25)),
]
for name, period, mode, sz, kw in configs:
    r = run(period, mode, sz, **kw)
    fa, fv, fsh, fm, fcal, fact, fe = metrics(r)
    oa, ov, osh, om, ocal, oact, oe = metrics(r[oos])
    rows.append(dict(策略=name, full_ann=round(fa,4), full_vol=round(fv,4), full_sh=round(fsh,3),
                     full_mdd=round(fm,4), full_calmar=round(fcal,3), full_active=round(fact,3), full_end=round(fe,0),
                     oos_ann=round(oa,4), oos_vol=round(ov,4), oos_sh=round(osh,3), oos_mdd=round(om,4),
                     oos_calmar=round(ocal,3), oos_active=round(oact,3), oos_end=round(oe,0)))
df = pd.DataFrame(rows).sort_values("full_sh", ascending=False)
print(df.to_string(index=False))
df.to_csv(OUT/"event_timing_scan.csv", index=False, encoding="utf-8-sig")
print("saved:", OUT/"event_timing_scan.csv")
