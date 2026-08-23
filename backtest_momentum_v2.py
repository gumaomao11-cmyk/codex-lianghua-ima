# -*- coding: utf-8 -*-
"""
Refined non-intraday backtest with volatility-target overlay.
Long-only, monthly rebalance, $20k account, costs included.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(r"F:\even-codex\us-stock-data")
IDX_FILE = Path(r"F:\even-codex\panda\backtest\prices_2016.csv")
OUT_DIR  = Path(r"F:\even-codex\lianghua2\backtest_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = 20000.0
RF = 0.0
DAYS = 252

etf_px = pd.read_csv(IDX_FILE, index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce").ffill()
stock_px = pd.read_csv(DATA_DIR / "prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
full = stock_px.count() >= 2400
stock_px = stock_px.loc[:, full]

def monthly_last(px):
    return px.resample("ME").last()

def momentum_score(px, period, skip):
    ml = monthly_last(px)
    return ml.shift(skip) / ml.shift(skip + period) - 1.0

def momentum_accel_score(px, period, skip, accel_months=1, w_mom=0.5):
    """复合打分 = w_mom*动量 + (1-w_mom)*近1月加速"""
    ml = monthly_last(px)
    m = ml.shift(skip) / ml.shift(skip + period) - 1.0
    a = ml / ml.shift(accel_months) - 1.0
    return w_mom * m + (1.0 - w_mom) * a

def sma10_filter(px, month_last):
    ma = month_last.rolling(10, min_periods=10).mean()
    return (month_last >= ma).fillna(False)

def run(px, score, top, market_filter=None, cash_asset=None, cost_bps=5, vol_target=None, roll=60):
    px = px.dropna(how="all")
    cols = list(px.columns)
    daily_ret = px.pct_change().fillna(0.0)
    month_ends = pd.DatetimeIndex(score.index)

    # ---- pass 1: universe weights (position in "names", subject to trend filter)
    unit_weights = pd.DataFrame(0.0, index=px.index, columns=cols)
    rebal_days = []
    for i in range(1, len(month_ends)):
        d = month_ends[i]
        w = pd.Series(0.0, index=cols)
        s = score.loc[d].dropna()
        if len(s) > 0:
            ta = s.sort_values(ascending=False).index[:top].tolist()
            w[ta] = 1.0 / len(ta)
        invest = True
        if market_filter is not None:
            invest = bool(market_filter.loc[d]) if d in market_filter.index else True
        if not invest:
            w[:] = 0.0
            if cash_asset is not None and cash_asset in cols:
                w[cash_asset] = 1.0
        end = month_ends[i + 1] if i + 1 < len(month_ends) else px.index[-1]
        days = px.index[(px.index > d) & (px.index <= end)]
        if len(days):
            for c in cols:
                unit_weights.loc[days, c] = w[c]
            rebal_days.append(days[0])

    unit_gross = (unit_weights * daily_ret).sum(axis=1).fillna(0.0)

    # ---- pass 2: volatility scaling
    scale = pd.Series(1.0, index=unit_gross.index)
    if vol_target is not None and vol_target > 0:
        for rd in rebal_days:
            look = unit_gross.loc[:rd].iloc[-(roll+1):-1]
            look = look.dropna()
            if len(look) >= 21:
                ann_vol = look.std(ddof=1) * np.sqrt(DAYS)
                s = min(1.0, vol_target / ann_vol) if ann_vol > 0 else 1.0
            else:
                s = 1.0
            d2 = rd + pd.Timedelta(days=0)
            nxt = unit_gross.index[(unit_gross.index >= d2)]
            if len(nxt):
                scale.loc[nxt[:1]] = s
        scale = scale.replace(0.0, np.nan).ffill().fillna(1.0)
        # forward-fill per month block
        for i in range(1, len(month_ends)):
            d = month_ends[i]
            end = month_ends[i + 1] if i + 1 < len(month_ends) else px.index[-1]
            days = px.index[(px.index > d) & (px.index <= end)]
            if len(days):
                sval = float(scale.loc[days[0]]) if days[0] in scale.index else 1.0
                scale.loc[days] = sval
    else:
        scale[:] = 1.0

    weights = unit_weights.mul(scale, axis=0)

    gross = (weights * daily_ret).sum(axis=1).fillna(0.0)
    # transaction cost on USD turnover
    idx_map = pd.Series(np.arange(len(gross)), index=gross.index)
    cost = pd.Series(0.0, index=gross.index)
    prev = pd.Series(0.0, index=cols)
    for rd in rebal_days:
        d0 = rd - pd.Timedelta(days=1)
        if d0 in weights.index:
            w_new = weights.loc[d0]
        else:
            w_new = weights.loc[rd]
        turnover = (w_new - prev).abs().sum() / 2.0
        cost.loc[rd] = turnover * cost_bps / 10000.0
        prev = w_new.copy()
    strat = (gross - cost).clip(lower=-0.5)
    nav = (1 + strat).cumprod() * START

    ann_ret = (nav.iloc[-1] / START) ** (DAYS / len(strat)) - 1
    vol = strat.std(ddof=1) * np.sqrt(DAYS)
    sharpe = (ann_ret - RF) / vol if vol > 0 else np.nan
    dd = nav / nav.cummax() - 1
    mdd = dd.min()
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan
    active = (weights.abs().sum(axis=1) > 1e-6).mean()
    yearly = (1 + strat).groupby(strat.index.year).prod() - 1
    # annualized turnover
    trns = []
    prev = pd.Series(0.0, index=cols)
    for rd in rebal_days:
        d0 = rd - pd.Timedelta(days=1)
        w_new = weights.loc[d0] if d0 in weights.index else weights.loc[rd]
        trns.append((w_new - prev).abs().sum() / 2)
        prev = w_new.copy()
    ann_turnover = float(pd.Series(trns).mean()) if trns else 0.0
    return dict(nav=nav, strat=strat, weights=weights, dd=dd, yearly=yearly,
                ann_ret=ann_ret, vol=vol, sharpe=sharpe, mdd=mdd, calmar=calmar,
                active=float(active), ann_turnover=ann_turnover, final=nav.iloc[-1],
                scale_avg=float(scale.mean()))

def buyhold(px, ticker):
    ser = px[ticker].dropna()
    nav = ser / ser.iloc[0] * START
    r = nav.pct_change().fillna(0)
    dd = nav / nav.cummax() - 1
    ann_ret = (nav.iloc[-1] / START) ** (DAYS / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(DAYS)
    sharpe = (ann_ret - RF) / vol if vol > 0 else np.nan
    mdd = dd.min()
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    return dict(nav=nav, strat=r, dd=dd, yearly=yearly, ann_ret=ann_ret, vol=vol,
                sharpe=sharpe, mdd=mdd, calmar=ann_ret/abs(mdd) if mdd<0 else np.nan,
                active=1.0, ann_turnover=0.0, final=nav.iloc[-1], scale_avg=1.0)

etf12 = momentum_score(etf_px, 12, 1)
etf6  = momentum_score(etf_px, 6, 0)
spy_m = monthly_last(etf_px[['SPY']])['SPY']
spy10 = sma10_filter(etf_px['SPY'], spy_m)
stk6  = momentum_score(stock_px, 6, 0)
stk12 = momentum_score(stock_px, 12, 1)
stk6a  = momentum_accel_score(stock_px, 6, 1, accel_months=1, w_mom=0.5)
stk12a = momentum_accel_score(stock_px, 12, 1, accel_months=1, w_mom=0.5)

S = {}
S['StockMom6_top20']             = run(stock_px, stk6, 20, cost_bps=10)
S['StockMom6_top20_vol25']       = run(stock_px, stk6, 20, cost_bps=10, vol_target=0.25)
S['StockMom12-1_top10']          = run(stock_px, stk12, 10, cost_bps=10)
S['StockMom6-1_accel_top10']     = run(stock_px, stk6a, 10, cost_bps=10)
S['StockMom6-1_accel_top10_vol25'] = run(stock_px, stk6a, 10, cost_bps=10, vol_target=0.25)
S['StockMom12-1_accel_top10']    = run(stock_px, stk12a, 10, cost_bps=10)
S['ETF_Mom6_top2']               = run(etf_px, etf6, 2, cost_bps=5)
S['ETF_Mom6_top2_vol15']         = run(etf_px, etf6, 2, cost_bps=5, vol_target=0.15)
S['ETF_Mom6_top2_trend_cash']    = run(etf_px, etf6, 2, market_filter=spy10, cash_asset=None, cost_bps=5)
S['ETF_Mom6_top2_trend_cash_vol12'] = run(etf_px, etf6, 2, market_filter=spy10, cash_asset=None, cost_bps=5, vol_target=0.12)
S['ETF_Mom12-1_top2']            = run(etf_px, etf12, 2, cost_bps=5)
S['ETF_Mom12-1_top2_trend_cash'] = run(etf_px, etf12, 2, market_filter=spy10, cash_asset=None, cost_bps=5)

B = {}
for t in ['SPY','QQQ','IWM','SPMO','QUAL','USMV','VLUE']:
    B[t] = buyhold(etf_px, t)

def fmt_pct(x):
    return f"{x*100:.1f}%"

rows = []
for name, r in S.items():
    rows.append(dict(strategy=name, ann=f"{r['ann_ret']*100:.1f}%", vol=f"{r['vol']*100:.1f}%",
                     sharpe=f"{r['sharpe']:.2f}", mdd=f"{r['mdd']*100:.1f}%", calmar=f"{r['calmar']:.2f}",
                     final=f"${r['final']:,.0f}", active=f"{r['active']*100:.0f}%",
                     turn=f"{r['ann_turnover']*100:.0f}%", scale=f"{r['scale_avg']*100:.0f}%"))
for name, r in B.items():
    rows.append(dict(strategy="BH "+name, ann=f"{r['ann_ret']*100:.1f}%", vol=f"{r['vol']*100:.1f}%",
                     sharpe=f"{r['sharpe']:.2f}", mdd=f"{r['mdd']*100:.1f}%", calmar=f"{r['calmar']:.2f}",
                     final=f"${r['final']:,.0f}", active="100%", turn="0%", scale="100%"))
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "results_table.csv", index=False, encoding="utf-8-sig")
print(df.to_string(index=False))

# ---- charts
sel_strats = ['StockMom6-1_accel_top10','StockMom6-1_accel_top10_vol25','StockMom6_top20_vol25','ETF_Mom6_top2','ETF_Mom6_top2_vol15']
sel_bh = ['SPY','QQQ','SPMO','USMV']
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
ax = axes[0,0]
for t in sel_bh:
    ax.plot(B[t]['nav'].index, B[t]['nav'].values/START, lw=1.4, alpha=.75, label=f"{t} (BH)")
for k in sel_strats:
    ax.plot(S[k]['nav'].index, S[k]['nav'].values/START, lw=2, label=k.replace('_',' '))
ax.set_yscale('log'); ax.set_title('Growth of $20,000 (log scale, net of costs)')
ax.legend(fontsize=7, loc='upper left'); ax.grid(alpha=.3)
ax = axes[0,1]
for t in sel_bh:
    ax.plot(B[t]['dd'].index, B[t]['dd'].values*100, lw=1.2, alpha=.6)
for k in sel_strats:
    ax.plot(S[k]['dd'].index, S[k]['dd'].values*100, lw=1.7)
ax.set_title('Drawdown (%)'); ax.grid(alpha=.3)
ax.legend(sel_bh+sel_strats, fontsize=6, loc='lower left')
ax = axes[1,0]
for k in sel_strats:
    y = S[k]['yearly']*100
    ax.plot(y.index.astype(str), y.values, marker='o', lw=1, label=k.replace('_',' '))
ax.axhline(0, color='k', lw=.8); ax.set_title('Strategy yearly returns (%)'); ax.grid(alpha=.3)
ax.legend(fontsize=7)
ax = axes[1,1]
ax.axis('off')
sub = df[~df.strategy.str.startswith('BH')].copy().sort_values('sharpe', ascending=False)
tab = ax.table(cellText=sub[['strategy','ann','sharpe','mdd','calmar','final']].values,
               colLabels=['Strategy','AnnRet','Sharpe','MaxDD','Calmar','Final'],
               loc='center', cellLoc='left', colLoc='left')
tab.auto_set_font_size(False); tab.set_fontsize(7.5); tab.scale(1.05, 1.3)
ax.set_title('Strategy summary', fontsize=11)
plt.savefig(OUT_DIR / 'backtest_chart_v2.png', dpi=150)
plt.close(fig)
print("chart saved ->", OUT_DIR / "backtest_chart_v2.png")
