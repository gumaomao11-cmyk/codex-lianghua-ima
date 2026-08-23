# -*- coding: utf-8 -*-
"""周频策略优化扫描：在周频 6m-skip1 top10 基础上试多种增强
变体：base / top15 / 板块分散(max3/簇) / vol25 / 多周期动量 / SPY>200ma才满仓 / 风险平价权重
统一成本10bps，输出全期+样本外 夏普/年化/回撤/期末，并给出推荐。
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import AgglomerativeClustering

DATA = Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data")
IDX  = Path(os.environ.get("ETFS_REF_FILE") or r"F:\even-codex\panda\backtest\prices_2016.csv")
OUT = Path(__file__).resolve().parent / "backtest_output"
DAYS=252; START=20000.0; COST=10

px = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
px = px.loc[:, px.count() >= 2400]
idx = px.index; cols = list(px.columns); idpos = {d:i for i,d in enumerate(idx)}
dr = px.pct_change().fillna(0.0)
spy = pd.read_csv(IDX, index_col=0, parse_dates=True)["SPY"].reindex(idx).ffill()

# --- 动量(日线近似) ---
mom3  = px.div(px.shift(63)) - 1.0
mom6  = px.shift(21).div(px.shift(147)) - 1.0
mom9  = px.div(px.shift(189)) - 1.0
VOL60 = dr.rolling(60).std().fillna(0.0)

# --- 板块聚类代理 ---
rt = dr.tail(504).dropna(how="all")
corr = rt.corr().clip(-1,1).fillna(0.0)
lab_series = AgglomerativeClustering(n_clusters=12, metric="precomputed", linkage="average").fit_predict((1-corr).values)
sector_lab = dict(zip(corr.columns, lab_series))

def weekly_last_days():
    s = pd.Series(idx, index=idx.to_period("W"))
    last = s.groupby(level=0).last()
    return [int(idpos[d]) for d in last.tolist() if d in idpos]

def pick(variant, d, top, maxk):
    if variant in ("multi",):
        sc = -(pd.DataFrame({"a": mom3.iloc[d].rank(), "b": mom6.iloc[d].rank(), "c": mom9.iloc[d].rank()}).mean(axis=1))
    else:
        sc = mom6.iloc[d]
    sc = sc.sort_values(ascending=False)
    if variant == "sector_cap":
        out=[]; cnt={}
        for t, v in sc.items():
            if t not in sector_lab: continue
            l=sector_lab[t]
            if cnt.get(l,0) >= maxk: continue
            out.append(t); cnt[l]=cnt.get(l,0)+1
            if len(out)>=top: break
        return out
    return sc.index[:top].tolist()

def weights(variant, d, ta):
    if variant == "risk_parity":
        v = VOL60.iloc[d].reindex(ta).replace(0, np.nan)
        w = (1.0/v).fillna(1.0)
        return (w/w.sum()).to_dict() if w.sum()>0 else {t:1.0/len(ta) for t in ta}
    return {t: 1.0/len(ta) for t in ta}

def run(variant, top, maxk=3, vol_target=None, spy_ma=False):
    rb = [d for d in weekly_last_days() if d > 189 and d+1 < len(idx)]
    prev_w = pd.Series(0.0, index=cols); ret = pd.Series(0.0, index=idx); cost_line = pd.Series(0.0, index=idx)
    blocks = []
    for k, rdi in enumerate(rb):
        ta = pick(variant, rdi, top, maxk)
        wd = weights(variant, rdi, ta)
        new_w = pd.Series(0.0, index=cols)
        if ta:
            new_w = pd.Series({t: wd.get(t,0.0) for t in ta if t in cols})
        # 趋势过滤
        if spy_ma:
            ma = spy.rolling(200).mean()[idx[rdi]]
            if ma == ma and spy.iloc[rdi] < ma:
                new_w[:] = 0.0
        hold_start = rdi + 1
        seg_end = rb[k+1] if k+1 < len(rb) else len(idx)-1
        if seg_end <= hold_start: seg_end = hold_start + 1
        to = (new_w - prev_w).abs().sum()/2.0
        cost_line.iloc[hold_start] += to * COST / 10000.0
        li = [cols.index(c) for c in new_w.index if new_w[c] != 0]
        if li:
            wv = (dr.iloc[hold_start:seg_end+1, li] * new_w.values[new_w != 0]).sum(axis=1)
            ret.iloc[hold_start:seg_end+1] += wv
            block_ret = pd.Series(0.0, index=ret.index)
            block_ret.iloc[hold_start:seg_end+1] = wv
            blocks.append((hold_start, seg_end, block_ret))
        prev_w = new_w.copy()
    if vol_target:
        scale_series = pd.Series(1.0, index=ret.index)
        for k, rdi in enumerate(rb):
            hold_start = rdi+1
            if hold_start >= len(idx): continue
            look = ret.loc[:idx[hold_start]].iloc[-(61):-1].dropna()
            if len(look) >= 21 and look.std(ddof=1) > 0:
                s = min(1.0, vol_target/(look.std(ddof=1)*np.sqrt(DAYS)))
                seg_end = rb[k+1] if k+1 < len(rb) else len(idx)-1
                scale_series.iloc[hold_start:seg_end+1] = s
        ret = ret * scale_series
    return (ret - cost_line).clip(lower=-0.5)

def metrics(r):
    nav=(1+r).cumprod()*START
    ann=(nav.iloc[-1]/START)**(DAYS/len(nav))-1
    vol=r.std(ddof=1)*np.sqrt(DAYS)
    sh=ann/vol if vol>0 else np.nan
    mdd=(nav/nav.cummax()-1).min()
    return dict(ann=ann,vol=vol,sharpe=sh,mdd=float(mdd),final=float(nav.iloc[-1]))

oos = idx >= pd.Timestamp("2022-01-01")
variants = [
    ("base（周频6m-skip1 top10）", "base", 10, None, False),
    ("top15 更分散", "base", 15, None, False),
    ("板块分散 top10(max3/簇)", "sector_cap", 10, None, False),
    ("vol25 波动率目标", "base", 10, 0.25, False),
    ("多周期动量(3/6/9合成)", "multi", 10, None, False),
    ("SPY>200ma才满仓", "base", 10, None, True),
    ("风险平价权重 top10", "risk_parity", 10, None, False),
]
rows=[]
for name,variant,top,vt,ma in variants:
    r = run(variant, top, vol_target=vt, spy_ma=ma)
    f=metrics(r); o=metrics(r[oos])
    rows.append(dict(变体=name, full_sh=round(f["sharpe"],3), full_ann=round(f["ann"],4), full_mdd=round(f["mdd"],4), full_end=round(f["final"],0),
                     oos_sh=round(o["sharpe"],3), oos_ann=round(o["ann"],4), oos_mdd=round(o["mdd"],4), oos_end=round(o["final"],0)))
df=pd.DataFrame(rows).sort_values("full_sh", ascending=False)
out=OUT/"weekly_optimize_scan.csv"; df.to_csv(out, index=False, encoding="utf-8-sig")
print(df.to_string(index=False))
print("saved:", out)
