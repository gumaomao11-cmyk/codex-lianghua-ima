# -*- coding: utf-8 -*-
"""Walk-forward v3 周频版：持有 5 天，每周五调仓，利用 7-8 月短周期文本因子。"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from risk.optimizer import PortfolioOptimizer
from risk.industry_map import get_industry_map

PROJ = Path(r"F:\even-codex\lianghua+IMA")
DB_DIR = PROJ / "data" / "duckdb"
COST_BPS = 0.0010
all_selections = []


def load_dataset(route):
    path = DB_DIR / f"aligned_dataset_{route.lower()}_ortho.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_factor_cols(df):
    return [c for c in df.columns if c.startswith("factor_") and "ortho" in c]


def calc_daily_rets(df):
    px = df.pivot_table(index="date", columns="ticker", values="close")
    return px.pct_change().fillna(0)


def walkforward(df, route):
    df = df.sort_values(["ticker", "date"])
    factor_cols = get_factor_cols(df)
    print(f"[route {route}] factors: {factor_cols}")

    daily_rets = calc_daily_rets(df)
    all_dates = pd.to_datetime(df["date"].sort_values().unique())
    # 每周五调仓
    rebal_dates = pd.bdate_range(all_dates.min(), all_dates.max(), freq="W-FRI").tolist()
    rebal_dates = [d for d in rebal_dates if d in all_dates]

    sector_map = get_industry_map().set_index("ticker")["sector"].to_dict()
    portfolio_rets = []
    ret_col = "ret_5d"

    for i, rd in enumerate(rebal_dates[:-1]):
        # 训练窗口：调仓日前 6 个月
        train_start = rd - pd.DateOffset(months=6)
        train_df = df[(df["date"] >= train_start) & (df["date"] < rd)].copy()
        test_df = df[df["date"] == rd].copy()
        if test_df.empty or len(train_df) < 50:
            continue

        train = train_df.dropna(subset=[ret_col]).copy()
        if len(train) < 50:
            continue
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
            random_state=42, n_jobs=4
        )
        model.fit(train[factor_cols].fillna(0), train[ret_col])
        test_df["alpha_pred"] = model.predict(test_df[factor_cols].fillna(0))
        test_df["mom_pred"] = test_df["mom_20d"]

        top = test_df.dropna(subset=["alpha_pred", ret_col]).sort_values("alpha_pred", ascending=False).head(30)
        if len(top) < 8:
            continue
        try:
            end_idx = daily_rets.index.get_indexer([rd], method="nearest")[0]
            start_idx = max(0, end_idx - 60)
            hist_rets = daily_rets.iloc[start_idx:end_idx]
            cov = hist_rets[top["ticker"].unique()].cov() * 252
            cov = cov.reindex(index=top["ticker"], columns=top["ticker"]).fillna(0)

            opt = PortfolioOptimizer(risk_aversion=1.0, max_single=0.10, max_sector=0.25, min_count=8, max_count=15)
            alpha = top.set_index("ticker")["alpha_pred"]
            weights = opt.optimize(alpha, cov, tickers=top["ticker"].tolist())
            w = weights.reindex(top["ticker"]).fillna(0)
            w = w / w.sum() if w.sum() > 0 else w
            port_ret = (w * top.set_index("ticker")[ret_col]).sum() - COST_BPS

            for tk in w.index:
                if w[tk] > 0.001:
                    row = top[top["ticker"] == tk].iloc[0]
                    all_selections.append({
                        "date": rd, "route": route, "ticker": tk,
                        "weight": w[tk], "alpha_pred": row["alpha_pred"],
                        "mom_pred": row["mom_pred"], "ret_5d": row[ret_col],
                        "sector": sector_map.get(tk, "其他"),
                    })
        except Exception as e:
            print(f"  optimize error at {rd}: {e}")
            port_ret = top[ret_col].head(10).mean() - COST_BPS

        mom_top = test_df.dropna(subset=["mom_pred", ret_col]).sort_values("mom_pred", ascending=False).head(10)
        mom_ret = mom_top[ret_col].mean() - COST_BPS

        portfolio_rets.append({
            "date": rd, "route": route,
            "xgb_cvxpy": port_ret, "momentum_eq": mom_ret,
            "n_stocks": len(top),
        })

    return pd.DataFrame(portfolio_rets)


def summarize(rets):
    s = rets.dropna()
    if len(s) == 0:
        return {"n": 0, "mean": np.nan, "vol": np.nan, "sharpe": np.nan, "cum": np.nan, "maxdd": np.nan}
    mr = s.mean()
    v = s.std(ddof=0)
    cum = (1 + s).cumprod()
    maxdd = (cum / cum.cummax() - 1).min()
    return {"n": len(s), "mean": mr * 100, "vol": v * 100, "sharpe": mr / v if v > 0 else np.nan, "cum": (cum.iloc[-1] - 1) * 100, "maxdd": maxdd * 100}


def main():
    results = []
    for route in ["A", "B"]:
        print(f"\n{'='*60}\n[route {route}]\n{'='*60}")
        df = load_dataset(route)
        perf = walkforward(df, route)
        if not perf.empty:
            print(f"XGB+CVXPY: {summarize(perf['xgb_cvxpy'])}")
            print(f"Momentum Eq: {summarize(perf['momentum_eq'])}")
            results.append(perf)

    if results:
        all_perf = pd.concat(results, ignore_index=True)
        out = PROJ / "backtest_output" / "walkforward_v3_weekly5d_results.csv"
        all_perf.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n[saved] {out}")

    if all_selections:
        sel_df = pd.DataFrame(all_selections)
        sel_out = PROJ / "backtest_output" / "walkforward_v3_weekly5d_selections.csv"
        sel_df.to_csv(sel_out, index=False, encoding="utf-8-sig")
        print(f"[saved] {sel_out}")


if __name__ == "__main__":
    main()
