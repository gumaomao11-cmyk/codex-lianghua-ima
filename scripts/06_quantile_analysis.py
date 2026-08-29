# -*- coding: utf-8 -*-
"""
Book 3.4.6 / 3.4.7: cross-sectional quantile grouping + long-short spread test.

Key design decisions (documented so results are auditable):
1. Returns are FORWARD returns already (ret_1d = close[t+1]/close[t]-1).
2. We evaluate on CROSS-SECTIONALLY DEMEANED returns (excess vs same-day universe
   mean) so that quintile spreads are pure alpha, not market beta.
3. Quintiles are formed per date, only on dates with >= MIN_N non-null factor obs.
4. For overlapping horizons (5/10/21d) we ALSO report a non-overlapping subsample
   t-stat, because overlapping windows inflate significance.
"""
import os, sys, json
import numpy as np
import pandas as pd
import duckdb

MIN_N = 10
NQ = 5
OUT_DIR = "backtest_output"
REP = "reports"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REP, exist_ok=True)

ROUTES = {
    "a": ["factor_clean_alpha", "factor_clean_alpha_ortho"],
    "b": ["factor_research_20d", "factor_event_3d", "factor_news_1d", "factor_opinion_1d",
          "factor_research_20d_ortho", "factor_event_3d_ortho", "factor_news_1d_ortho",
          "factor_opinion_1d_ortho"],
}
HORIZONS = ["ret_1d", "ret_5d", "ret_10d", "ret_21d"]

def load(route):
    con = duckdb.connect()
    cols = ["date", "ticker"] + ROUTES[route] + HORIZONS
    q = "select {} from 'data/duckdb/aligned_dataset_{}_ortho.parquet'".format(",".join(cols), route)
    df = con.execute(q).df()
    df["date"] = pd.to_datetime(df["date"])
    return df

def demean(df):
    """cross-sectional excess return vs universe mean of that date"""
    for h in HORIZONS:
        df["x_" + h] = df[h] - df.groupby("date")[h].transform("mean")
    return df

def quantile_table(df, fac, h):
    sub = df[["date", "ticker", fac, h, "x_" + h]].dropna(subset=[fac, h])
    cnt = sub.groupby("date")[fac].transform("count")
    sub = sub[cnt >= MIN_N].copy()
    if sub.empty:
        return None
    def qcut(s):
        try:
            return pd.qcut(s.rank(method="first"), NQ, labels=False)
        except Exception:
            return pd.Series(np.nan, index=s.index)
    sub["q"] = sub.groupby("date")[fac].transform(qcut)
    sub = sub.dropna(subset=["q"])
    sub["q"] = sub["q"].astype(int)
    return sub

def ann_stats(daily, periods_per_year):
    daily = daily.dropna()
    n = len(daily)
    if n < 5:
        return dict(n=n, mean=np.nan, ann=np.nan, sharpe=np.nan, t=np.nan)
    m, s = daily.mean(), daily.std(ddof=1)
    sharpe = m / s * np.sqrt(periods_per_year) if s > 0 else np.nan
    t = m / (s / np.sqrt(n)) if s > 0 else np.nan
    return dict(n=n, mean=m, ann=m * periods_per_year, sharpe=sharpe, t=t)

def analyze(df, fac, h, tag=""):
    sub = quantile_table(df, fac, h)
    if sub is None or sub.empty:
        return None
    step = int(h.replace("ret_", "").replace("d", ""))
    ppy = 252 / step

    # quintile mean excess return (pooled)
    grp = sub.groupby("q")["x_" + h].agg(["mean", "count"])
    grp_raw = sub.groupby("q")[h].mean()

    # daily long-short series (equal weight within quintile)
    piv = sub.pivot_table(index="date", columns="q", values="x_" + h, aggfunc="mean")
    if 0 not in piv.columns or (NQ - 1) not in piv.columns:
        return None
    ls = piv[NQ - 1] - piv[0]

    full = ann_stats(ls, ppy)
    # non-overlapping subsample for horizons > 1
    if step > 1:
        ls_no = ls.iloc[::step]
        no = ann_stats(ls_no, ppy)
    else:
        no = full

    # monotonicity: spearman of quintile index vs mean excess return
    mono = pd.Series(grp["mean"].values).corr(pd.Series(range(NQ)), method="spearman")

    # rank IC per date
    ic_by_date = sub.groupby("date").apply(
        lambda g: g[fac].corr(g["x_" + h], method="spearman") if g[fac].nunique() > 2 else np.nan
    ).dropna()
    ic_mean = ic_by_date.mean()
    ic_ir = ic_mean / ic_by_date.std(ddof=1) if ic_by_date.std(ddof=1) > 0 else np.nan
    ic_t = ic_mean / (ic_by_date.std(ddof=1) / np.sqrt(len(ic_by_date))) if len(ic_by_date) > 2 else np.nan

    return dict(
        tag=tag, factor=fac, horizon=h, dates=int(piv.shape[0]),
        q1=grp["mean"].get(0, np.nan), q2=grp["mean"].get(1, np.nan),
        q3=grp["mean"].get(2, np.nan), q4=grp["mean"].get(3, np.nan),
        q5=grp["mean"].get(NQ - 1, np.nan),
        q1_raw=grp_raw.get(0, np.nan), q5_raw=grp_raw.get(NQ - 1, np.nan),
        mono=mono,
        ls_mean=full["mean"], ls_ann=full["ann"], ls_sharpe=full["sharpe"], ls_t=full["t"],
        ls_t_nonoverlap=no["t"], ls_n_nonoverlap=no["n"],
        ic_mean=ic_mean, ic_ir=ic_ir, ic_t=ic_t, ic_n=len(ic_by_date),
    )

def main():
    rows = []
    subperiods = {
        "2025H1": ("2025-01-01", "2025-06-30"),
        "2025H2": ("2025-07-01", "2025-12-31"),
        "2026H1": ("2026-01-01", "2026-06-30"),
        "2026Q3": ("2026-07-01", "2026-12-31"),
    }
    for route in ["a", "b"]:
        df = load(route)
        df = df[df["date"] >= "2025-01-01"].copy()
        df = demean(df)
        for fac in ROUTES[route]:
            if fac not in df.columns:
                continue
            for h in HORIZONS:
                r = analyze(df, fac, h, tag="FULL")
                if r:
                    r["route"] = route
                    rows.append(r)
            # subperiod only on ret_21d and ret_5d to keep table readable
            for name, (s, e) in subperiods.items():
                d2 = df[(df["date"] >= s) & (df["date"] <= e)]
                for h in ["ret_5d", "ret_21d"]:
                    r = analyze(d2, fac, h, tag=name)
                    if r:
                        r["route"] = route
                        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "quantile_analysis.csv"), index=False, encoding="utf-8-sig")
    print("saved", len(out), "rows")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    show = out[out.tag == "FULL"][["route","factor","horizon","dates","q1","q2","q3","q4","q5","mono","ls_ann","ls_sharpe","ls_t","ls_t_nonoverlap","ic_mean","ic_ir","ic_t"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

if __name__ == "__main__":
    main()
