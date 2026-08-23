# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path(r"F:\even-codex\lianghua2\backtest_output")
DATA = Path(r"F:\even-codex\us-stock-data")
IDX = Path(r"F:\even-codex\panda\backtest\prices_2016.csv")

etf = pd.read_csv(IDX, index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce").ffill()
stk = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
stk = stk.loc[:, stk.count() >= 2400]
START = 20000.0

def ml(px): return px.resample("ME").last()
def mom(px, period, skip):
    m=ml(px); return m.shift(skip)/m.shift(skip+period)-1
def score(px, period, skip): return mom(px, period, skip)

# ---- filters (boolean over month-end index of score)
score_idx = ml(stk).index
spy_m = ml(etf[['SPY']])['SPY']
spy_daily = etf['SPY']

f_none  = pd.Series(True, index=score_idx)
f_12_1  = mom(etf[['SPY']],12,1)['SPY'] > 0
f_6_1   = mom(etf[['SPY']],6,1)['SPY'] > 0
f_200d  = (spy_daily > spy_daily.rolling(200,min_periods=200).mean()).reindex(score_idx).fillna(False)
f_10m   = spy_m >= spy_m.rolling(10,min_periods=10).mean()
filters = {"none":f_none, "SPY12-1>0":f_12_1, "SPY6-1>0":f_6_1, "SPY>200dSMA":f_200d, "SPY10mSMA":f_10m}

mret_stk = ml(stk).pct_change().fillna(0.0)

def run_period(px, mret, sc, filt, top, cost_bps, start_date, end_date, start_cap=START):
    idx = list(px.index)
    dates = list(mret.index)
    in_range = [d for d in dates if start_date <= d.to_timestamp() <= end_date] if hasattr(dates[0], 'to_timestamp') else dates
    # actually dates are Timestamps already (DatetimeIndex Element is Timestamp)
    in_range = [d for d in dates if start_date <= d <= end_date]
    nav = pd.Series(dtype=float)
    w_prev = pd.Series(0.0, index=px.columns)
    value = start_cap
    rets = []
    jour = []
    rng = []
    for i, d in enumerate(in_range):
        if i == 0:
            continue
        # signal at previous month end
        prev_d = in_range[i-1]
        if prev_d not in sc.index:
            continue
        s = sc.loc[prev_d].dropna()
        inv = bool(filt.loc[prev_d]) if prev_d in filt.index else True
        if inv and len(s) > 0:
            ta = s.sort_values(ascending=False).index[:top].tolist()
            w = pd.Series(0.0, index=px.columns); w[ta] = 1.0/len(ta)
        else:
            w = pd.Series(0.0, index=px.columns)
        turnover = (w - w_prev).abs().sum() / 2.0
        cost = turnover * cost_bps / 10000.0
        # asset return for this month
        r = float((w * mret.loc[d]).sum()) - cost
        r = max(r, -0.5)
        value *= (1 + r)
        rets.append(r)
        rng.append(d)
        w_prev = w.copy()
        jour.append((w.abs().sum() > 1e-6))
    ret = pd.Series(rets, index=pd.DatetimeIndex(rng))
    nav = (1+ret).cumprod() * start_cap
    if len(ret) == 0:
        return None
    ann = (nav.iloc[-1]/start_cap)**(12/len(ret)) - 1
    vol = ret.std(ddof=1)*np.sqrt(12)
    sharpe = ann/vol if vol>0 else np.nan
    dd = (nav/nav.cummax()-1).min()
    calmar = ann/abs(dd) if dd<0 else np.nan
    active = float(np.mean(jour))
    return dict(ann=ann, vol=vol, sharpe=sharpe, mdd=dd, calmar=calmar,
                final=nav.iloc[-1], active=active, ret=ret, nav=nav, positive=(ret>0).mean())

looks = [("Mom6", 6, 0), ("Mom9-1", 9, 1), ("Mom12-1", 12, 1)]
tops = [10, 20]
IS = (pd.Timestamp("2017-01-31"), pd.Timestamp("2021-12-31"))
rows = []
for lname, period, skip in looks:
    sc = score(stk, period, skip)
    for top in tops:
        for fname, filt in filters.items():
            r = run_period(stk, mret_stk, sc, filt, top, 10, IS[0], IS[1])
            if r:
                rows.append(dict(strategy=f"{lname}_top{top}", filter=fname, sharpe=r['sharpe'],
                                 ann=r['ann']*100, mdd=r['mdd']*100, vol=r['vol']*100,
                                 calmar=r['calmar'], active=r['active']*100,
                                 final=f"${r['final']:,.0f}"))
grid = pd.DataFrame(rows)
grid.to_csv(OUT/"is_grid_filters.csv", index=False, encoding="utf-8-sig")
print("IS 2017-01..2021-12 grid (stock momentum, cost 10bps/side):")
print(grid.sort_values('sharpe', ascending=False).to_string(index=False))
