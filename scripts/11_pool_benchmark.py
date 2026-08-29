# -*- coding: utf-8 -*-
"""
Same-pool benchmark: decompose Route-A style returns into
  (1) pool-selection effect  = discussed-pool EW  minus  full-universe EW
  (2) stock-selection effect = strategy           minus  discussed-pool EW
All using the SAME forward-return convention as the backtest (ret_1d).
"""
import duckdb, pandas as pd, numpy as np

START, END = "2025-07-01", "2026-08-26"
con = duckdb.connect()
df = con.execute(f"""
  select date, ticker, close, ret_1d, factor_clean_alpha
  from 'data/duckdb/aligned_dataset_a_ortho.parquet'
  where date >= '2025-01-01' and date <= '{END}'
""").df()
df["date"] = pd.to_datetime(df["date"])
df["covered_today"] = df.factor_clean_alpha.notna()

# rolling coverage: ticker was discussed at least once in trailing 60 calendar days
df = df.sort_values(["ticker", "date"])
df["cov_roll"] = (df.groupby("ticker")["covered_today"]
                    .transform(lambda s: s.rolling(60, min_periods=1).max()).fillna(0).astype(bool))
# ever-discussed (static pool, uses future info -> reported for reference only)
ever = df.groupby("ticker").covered_today.max()
df["cov_ever"] = df.ticker.map(ever)

bt = df[(df.date >= START) & (df.date <= END)].copy()

def ew(mask, label):
    s = bt[mask].groupby("date").ret_1d.mean().dropna()
    return label, s

series = dict([
    ew(slice(None) if False else bt.index == bt.index, "full_universe_EW"),
    ew(bt.cov_roll.values, "discussed_pool_EW_rolling60d"),
    ew(bt.cov_ever.values, "discussed_pool_EW_ever(lookahead)"),
    ew(bt.covered_today.values, "discussed_today_EW"),
])

# benchmarks SPY/QQQ on same forward convention
ref = pd.read_csv(r"F:\even-codex\panda\backtest\prices_2016.csv")
dc = ref.columns[0]
ref[dc] = pd.to_datetime(ref[dc]); ref = ref.set_index(dc).sort_index()
for b in ["SPY", "QQQ"]:
    if b in ref.columns:
        fwd = ref[b].shift(-1) / ref[b] - 1
        series[b] = fwd.loc[START:END].dropna()

def stats(s):
    n = len(s); m = s.mean(); sd = s.std(ddof=1)
    ann = (1 + s).prod() ** (252 / n) - 1
    sharpe = m / sd * np.sqrt(252) if sd > 0 else np.nan
    cum = (1 + s).prod() - 1
    eq = (1 + s).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    return dict(days=n, ann=ann, sharpe=sharpe, cum=cum, mdd=mdd)

print(f"{'series':<36}{'days':>6}{'ann':>10}{'sharpe':>8}{'cum':>10}{'maxDD':>9}")
for k, s in series.items():
    st = stats(s)
    print(f"{k:<36}{st['days']:>6}{st['ann']:>9.1%}{st['sharpe']:>8.2f}{st['cum']:>9.1%}{st['mdd']:>9.1%}")

# pool effect as a spread series
a = series["discussed_pool_EW_rolling60d"]; b = series["full_universe_EW"]
sp = (a - b).dropna()
t = sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp)))
print(f"\npool effect (discussed_rolling60 - full universe): ann={sp.mean()*252:+.1%}  t={t:+.2f}  n={len(sp)}")

# average pool size
print("\navg pool size: rolling60=%.0f  today=%.0f  ever=%.0f  full=%.0f" % (
    bt[bt.cov_roll].groupby("date").ticker.nunique().mean(),
    bt[bt.covered_today].groupby("date").ticker.nunique().mean(),
    bt[bt.cov_ever].groupby("date").ticker.nunique().mean(),
    bt.groupby("date").ticker.nunique().mean()))

pd.DataFrame(series).to_csv("backtest_output/pool_benchmarks.csv", encoding="utf-8-sig")
