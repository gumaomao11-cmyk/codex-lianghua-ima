# -*- coding: utf-8 -*-
"""
Non-intraday strategy backtest for a ~$20k account.
Data sources (per AGENTS.md):
 - US stock daily prices: F:/even-codex/us-stock-data (prices.csv)
 - Index/ETF reference:   F:/even-codex/panda/backtest/prices_2016.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR  = Path(r"F:\even-codex\us-stock-data")
IDX_FILE  = Path(r"F:\even-codex\panda\backtest\prices_2016.csv")
OUT_DIR   = Path(r"F:\even-codex\lianghua2\backtest_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RF = 0.0
START_CAPITAL = 20000.0
TRADING_DAYS = 252

# ---------------------------------------------------------------- data load
etf_px = pd.read_csv(IDX_FILE, index_col=0, parse_dates=True).sort_index()
etf_px = etf_px.apply(pd.to_numeric, errors="coerce").ffill()
stock_px = pd.read_csv(DATA_DIR / "prices.csv", index_col=0, parse_dates=True).sort_index()

# clean stock universe: only names with nearly the full 10-year history
full = stock_px.count() >= 2400
stock_px = stock_px.loc[:, full]
stock_px = stock_px.apply(pd.to_numeric, errors="coerce")

# ---------------------------------------------------------------- helpers
def monthly_last(px):
    return px.resample("ME").last()

def momentum_score(px, period, skip):
    """Momentum score at month-end: ret over `period` months, skipping `skip` most recent months."""
    ml = monthly_last(px)
    return ml.shift(skip) / ml.shift(skip + period) - 1.0

def sma10_filter(px, month_last):
    """True where main index month-end close >= its 10-month SMA."""
    ma = month_last.rolling(10, min_periods=10).mean()
    out = (month_last >= ma)
    return out.fillna(False)

def run_monthly(px, score, top, market_filter=None, cash_asset=None,
                cost_bps=5, cash_carry=0.0, start=START_CAPITAL):
    """Monthly rebalance long-only on top-N ranked assets. Returns dict with nav & stats."""
    px = px.dropna(how="all")
    daily_ret = px.pct_change().fillna(0.0)
    cols = list(px.columns)
    weights = pd.DataFrame(0.0, index=px.index, columns=cols)
    month_ends = pd.DatetimeIndex(score.index)
    rebal_days = []
    prev_w = pd.Series(0.0, index=cols)

    for i in range(1, len(month_ends)):
        d = month_ends[i]
        w = pd.Series(0.0, index=cols)
        s = score.loc[d].dropna()
        if len(s) > 0:
            top_assets = s.sort_values(ascending=False).index[:top].tolist()
            w[top_assets] = 1.0 / len(top_assets)
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
                weights.loc[days, c] = w[c]
            rebal_days.append(days[0])
        prev_w = w.copy()

    # gross daily strategy return
    gross = (weights * daily_ret).sum(axis=1)

    # transaction costs only on rebalance days
    idx = pd.Series(range(len(gross)), index=gross.index)
    cost = pd.Series(0.0, index=gross.index)
    # reuse iteration to compute turnover exactly
    prev = pd.Series(0.0, index=cols)
    for rd in rebal_days:
        d = rd - pd.Timedelta(days=1)
        if d not in weights.index:
            # fallback: use first day of pos as the rebalance effective state
            w = weights.loc[rd]
        else:
            w = weights.loc[d]
        turnover = (w - prev).abs().sum() / 2.0
        cost.loc[rd] = turnover * cost_bps / 10000.0
        prev = w.copy()

    gross = gross.fillna(0.0)
    strat_ret = gross - cost
    strat_ret = strat_ret.clip(lower=-0.50)

    nav = (1.0 + strat_ret).cumprod() * start

    # apply tiny cash carry (0 by default)
    # stats
    r = strat_ret
    ann_ret  = (nav.iloc[-1] / start) ** (TRADING_DAYS / max(len(r), 1)) - 1.0
    vol      = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe   = (ann_ret - RF) / vol if vol > 0 else np.nan
    roll_max = nav.cummax()
    dd       = (nav / roll_max - 1.0)
    max_dd   = dd.min()
    calmar   = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    active   = (weights.abs().sum(axis=1) > 1e-6).mean()
    # turnover stats
    turnover_days = [d for d in rebal_days if d in cost.index and cost.loc[d] > 0]
    ann_turnover = float(pd.Series([(weights.loc[d] - weights.loc[d - pd.Timedelta(days=1)]).abs().sum()/2
                                    for d in turnover_days if d - pd.Timedelta(days=1) in weights.index]).mean()) if turnover_days else 0.0
    yearly = (1 + strat_ret).groupby(strat_ret.index.year).prod() - 1.0
    return dict(nav=nav, ret=strat_ret, weights=weights, dd=dd, ann_ret=ann_ret,
                vol=vol, sharpe=sharpe, max_dd=max_dd, calmar=calmar,
                active=active, rebal_days=rebal_days, yearly=yearly,
                ann_turnover=ann_turnover, final=nav.iloc[-1])

def buyhold(px, ticker, start=START_CAPITAL):
    ser = px[ticker].dropna()
    nav = ser / ser.iloc[0] * start
    r = nav.pct_change().fillna(0.0)
    roll_max = nav.cummax()
    dd = nav / roll_max - 1.0
    ann_ret = (nav.iloc[-1] / start) ** (TRADING_DAYS / len(r)) - 1.0
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RF) / vol if vol > 0 else np.nan
    mdd = dd.min()
    yearly = (1 + r).groupby(r.index.year).prod() - 1.0
    return dict(nav=nav, ret=r, dd=dd, ann_ret=ann_ret, vol=vol, sharpe=sharpe,
                max_dd=mdd, calmar=ann_ret/abs(mdd) if mdd < 0 else np.nan,
                active=1.0, rebal_days=[], yearly=yearly, ann_turnover=0.0, final=nav.iloc[-1])

def summarize(name, res):
    return (name, res['ann_ret'], res['vol'], res['sharpe'], res['max_dd'],
            res['calmar'], res['final'], res['active'], res['ann_turnover'])

# ---------------------------------------------------------------- ETF strategies
etf_score_12_1 = momentum_score(etf_px, period=12, skip=1)
etf_score_6_0  = momentum_score(etf_px, period=6,  skip=0)
etf_score_3_0  = momentum_score(etf_px, period=3,  skip=0)
spy_m10        = sma10_filter(etf_px['SPY'].dropna(), monthly_last(etf_px[['SPY']])['SPY'])

strategies = {}
strategies['ETF_12-1_top2']   = run_monthly(etf_px, etf_score_12_1, top=2, cost_bps=5)
strategies['ETF_6_top2']      = run_monthly(etf_px, etf_score_6_0,  top=2, cost_bps=5)
strategies['ETF_12-1_top3_trend10m_cash'] = run_monthly(etf_px, etf_score_12_1, top=3,
                                        market_filter=spy_m10, cash_asset=None, cost_bps=5)
strategies['ETF_12-1_top3_trend10m_TLT']  = run_monthly(etf_px, etf_score_12_1, top=3,
                                        market_filter=spy_m10, cash_asset='TLT', cost_bps=5)
strategies['ETF_6_top3']      = run_monthly(etf_px, etf_score_6_0,  top=3, cost_bps=5)

benchmarks = {
    'SPY buy&hold': buyhold(etf_px, 'SPY'),
    'QQQ buy&hold': buyhold(etf_px, 'QQQ'),
    'IWM buy&hold': buyhold(etf_px, 'IWM'),
    'SPMO buy&hold': buyhold(etf_px, 'SPMO'),
}
benchmarks['USMV buy&hold']  = buyhold(etf_px, 'USMV')
benchmarks['QUAL buy&hold']  = buyhold(etf_px, 'QUAL')

# ---------------------------------------------------------------- stock strategy
mom6_stock = momentum_score(stock_px, period=6, skip=0)
mom12_stock = momentum_score(stock_px, period=12, skip=1)
strategies['STOCK_6_top10']   = run_monthly(stock_px, mom6_stock,  top=10, cost_bps=10)
strategies['STOCK_6_top20']   = run_monthly(stock_px, mom6_stock,  top=20, cost_bps=10)
strategies['STOCK_12-1_top10'] = run_monthly(stock_px, mom12_stock, top=10, cost_bps=10)

# ---------------------------------------------------------------- metrics table
rows = []
for name, r in strategies.items():
    rows.append(summarize("STRAT|"+name, r))
for name, r in benchmarks.items():
    rows.append(summarize("BH|"+name, r))

cols = ['name','ann_ret','vol','sharpe','mdd','calmar','final_value','time_in_mkt','ann_turnover']
df = pd.DataFrame(rows, columns=cols).sort_values('sharpe', ascending=False)
df[['ann_ret','vol','sharpe','mdd','calmar','time_in_mkt','ann_turnover']] = \
    df[['ann_ret','vol','sharpe','mdd','calmar','time_in_mkt','ann_turnover']].apply(pd.to_numeric)

def pct(x, d=1):
    return f"{x*100:,.{d}f}%"
df['ann_ret_f'] = df['ann_ret'].apply(lambda x: pct(x))
df['vol_f']     = df['vol'].apply(lambda x: pct(x))
df['sharpe_f']  = df['sharpe'].apply(lambda x: f"{x:.2f}")
df['mdd_f']     = df['mdd'].apply(lambda x: pct(x,1))
df['calmar_f']  = df['calmar'].apply(lambda x: f"{x:.2f}")

out = pd.DataFrame({
 '指标': ['年化收益','年化波动','夏普(无风险0)','最大回撤','Calmar','期末资产(元)','在场时间','年均换手'],
})
for _, row in df.iterrows():
    tag = row['name'].replace('STRAT|','').replace('BH|','')
    out[tag] = [
        row['ann_ret_f'], row['vol_f'], row['sharpe_f'], row['mdd_f'], row['calmar_f'],
        f"${row['final_value']:,.0f}", pct(row['time_in_mkt'],0), f"{row['ann_turnover']*100:,.0f}%"
    ]
print(out.to_string(index=False))

# summary CSV
df.to_csv(OUT_DIR / 'results_summary.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- charts
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
plot_strats = {k: strategies[k] for k in ['ETF_12-1_top2','ETF_6_top2','ETF_12-1_top3_trend10m_cash','STOCK_6_top20']}
plot_bhs = {k: benchmarks[k] for k in ['SPY buy&hold','QQQ buy&hold','SPMO buy&hold','USMV buy&hold']}

ax = axes[0,0]
for k, r in plot_bhs.items():
    ax.plot(r['nav'].index, r['nav'].values/START_CAPITAL, lw=1.5, alpha=.85, label=f"{k} (BH)")
for k, r in plot_strats.items():
    ax.plot(r['nav'].index, r['nav'].values/START_CAPITAL, lw=2.0, label=k.replace('_',' '))
ax.set_yscale('log'); ax.set_title('Growth of $20,000 (log scale, price returns)')
ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=.3)

ax = axes[0,1]
for k, r in plot_bhs.items():
    ax.plot(r['dd'].index, r['dd'].values*100, lw=1.4, alpha=.7)
for k, r in plot_strats.items():
    ax.plot(r['dd'].index, r['dd'].values*100, lw=1.8)
ax.set_title('Drawdown (%)'); ax.grid(alpha=.3); ax.legend(list(plot_bhs.keys())+list(plot_strats.keys()), fontsize=6, loc='lower left')

ax = axes[1,0]
for k, r in plot_strats.items():
    ax.plot(r['yearly'].index.astype(str), r['yearly'].values*100, marker='o', label=k.replace('_',' '))
ax.axhline(0, color='k', lw=.8)
ax.set_title('Strategy yearly returns (%)'); ax.grid(alpha=.3); ax.legend(fontsize=7)

ax = axes[1,1]
ax.axis('off')
ax.text(0.02, 0.98, "Strategy summary (net of costs)", fontsize=12, va='top', weight='bold')
tbl = df[df['name'].str.startswith('STRAT')].copy()
tbl['name'] = tbl['name'].str.replace('STRAT|','')
tbl2 = tbl[['name','ann_ret_f','sharpe_f','mdd_f','calmar_f','final_value']].copy()
tbl2['final_value'] = tbl2['final_value'].apply(lambda x: f"${x:,.0f}")
tab = ax.table(cellText=tbl2.values, colLabels=['Strategy','AnnRet','Sharpe','MaxDD','Calmar','Final'],
               loc='center', cellLoc='left', colLoc='left')
tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1.1, 1.25)
plt.savefig(OUT_DIR / 'backtest_chart.png', dpi=150)
plt.close(fig)
print("Saved:", OUT_DIR / 'backtest_chart.png')
print("\nWarnings: prices are split-adjusted price only (no dividends); costs are slippage/spread estimates.\n")
