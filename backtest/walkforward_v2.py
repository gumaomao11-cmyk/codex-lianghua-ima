# -*- coding: utf-8 -*-
"""工业级 walk-forward 回测引擎 v2：DuckDB + As-of Join + 正交化 + CVXPY 风控优化。"""
import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.dataloader import UnifiedLoader, lag_text_factors
from alpha.orthogonal import orthogonalize_factors
from alpha.ic_analysis import FactorIC
from risk.optimizer import PortfolioOptimizer
from risk.barra_risk import BarraRisk

DAYS = 252
COST_BPS = 10


def rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def load_all_features(loader, all_tickers, start, end, text_cols, use_ortho=True):
    """一次性加载所有特征，按交易日对齐，lag 1 日，无未来。"""
    print("loading prices...")
    px = loader.load_prices(start=start, end=end, tickers=all_tickers)
    daily_ret = px.pct_change().fillna(0.0)

    print("loading text factors with asof join + lag...")
    tf = loader.load_text_factors(start=start, end=end, tickers=all_tickers, asof=True)
    tf = lag_text_factors(tf, periods=1)

    print("building price features...")
    ml = px.resample("ME").last()
    mom7 = ml.shift(1) / ml.shift(7) - 1.0
    # 把月度 mom7 展开到日度：每个交易日取最新月度值
    mom7_daily = mom7.reindex(px.index, method="ffill")

    feat_px = pd.DataFrame({
        "date": px.index.repeat(len(px.columns)),
        "ticker": list(px.columns) * len(px),
        "price": px.values.ravel(),
        "mom_7m": mom7_daily.values.ravel(),
        "mom_10d": (px / px.shift(10) - 1.0).values.ravel(),
        "mom_20d": (px / px.shift(20) - 1.0).values.ravel(),
        "rsi_14": rsi(px, 14).values.ravel(),
        "vol_20d": (daily_ret.rolling(20).std() * np.sqrt(DAYS)).values.ravel(),
    })

    feat = feat_px.merge(tf, on=["date", "ticker"], how="left")
    for c in text_cols:
        if c in feat.columns:
            feat[c] = feat[c].fillna(0)

    if use_ortho:
        control_cols = ["mom_7m", "mom_10d", "mom_20d", "rsi_14", "vol_20d"]
        valid_text = [c for c in text_cols if c in feat.columns]
        print(f"orthogonalizing text factors: {valid_text}")
        feat = orthogonalize_factors(feat, valid_text, control_cols, method="gs")

    return px, daily_ret, feat


def get_forward_return(px, rebal_date, horizon=21):
    idx = px.index.get_indexer([rebal_date], method="nearest")[0]
    if idx < 0 or idx + horizon >= len(px):
        return pd.Series(np.nan, index=px.columns)
    return px.iloc[idx + horizon] / px.iloc[idx] - 1.0


def summarize_monthly(rets):
    s = pd.Series(rets).dropna()
    if len(s) == 0:
        return {"n": 0, "mean": np.nan, "vol": np.nan, "sharpe": np.nan, "cum": np.nan, "maxdd": np.nan}
    if len(s) == 1:
        return {"n": 1, "mean": float(s.mean()), "vol": np.nan, "sharpe": np.nan, "cum": float(s.sum()), "maxdd": np.nan}
    mr = float(s.mean())
    v = float(s.std(ddof=0))
    cum = (1 + s).cumprod()
    maxdd = float((cum / cum.cummax() - 1).min())
    return {"n": len(s), "mean": mr * 100, "vol": v * 100, "sharpe": mr / v if v > 0 else np.nan, "cum": float(cum.iloc[-1] - 1) * 100, "maxdd": maxdd * 100}


def run(config=None):
    loader = UnifiedLoader()
    all_tickers = loader.load_prices(end="2026-08-26").columns.tolist()
    text_cols = [c for c in loader.load_text_factors(start="2025-01-01", end="2026-08-26").columns if c not in ["date", "ticker"]]

    px, daily_ret, feat = load_all_features(loader, all_tickers, "2025-01-01", "2026-08-26", text_cols, use_ortho=True)

    ml = px.resample("ME").last()
    rebal_raw = list(ml.truncate("2025-09-30", px.index[-1]).index)
    # 把 month-end 映射到最近交易日，确保 feat 中有该日期
    rebal_idx = px.index.get_indexer(rebal_raw, method="nearest")
    rebal = [px.index[i] for i in rebal_idx if i >= 0]
    print(f"rebal dates: {len(rebal)} from {rebal[0].date()} to {rebal[-1].date()}")
    ortho_cols = [f"{c}_ortho" for c in text_cols]

    # ---- IC 评估 ----
    print("\n" + "=" * 60)
    print("因子 IC 评估")
    print("=" * 60)
    ic_frames = []
    for d in rebal:
        f_day = feat[feat["date"] == d].copy()
        if f_day.empty: continue
        fwd = get_forward_return(px, d, 21)
        f_day["ret_21d"] = f_day["ticker"].map(fwd)
        ic_frames.append(f_day)
    ic_df = pd.concat(ic_frames, ignore_index=True)
    fac = FactorIC(ic_df)
    report, selected = fac.evaluate_all(text_cols + ortho_cols, forward_col="ret_21d", ic_threshold=0.02, ir_threshold=0.2)
    print(report.to_string(index=False))
    # 选择 IC 绝对值最高且方向稳定的原始/正交因子
    report_sorted = report.sort_values("mean_ic", key=abs, ascending=False)
    selected = report_sorted.head(5)["factor"].tolist()
    print(f"selected factors for model: {selected}")
    out = Path(__file__).resolve().parent.parent / "backtest_output"
    report.to_csv(out / "factor_ic.csv", index=False, encoding="utf-8-sig")

    feature_cols = ["mom_7m", "mom_10d", "mom_20d", "rsi_14", "vol_20d"] + [c for c in selected if c in feat.columns]
    print(f"model feature cols: {feature_cols}")

    # ---- walk-forward 训练样本 ----
    print("\nbuilding walk-forward training samples...")
    samples = []
    for d in rebal[:-1]:
        f_day = feat[feat["date"] == d].copy()
        if f_day.empty: continue
        fwd = get_forward_return(px, d, 21)
        f_day["ret_21d"] = f_day["ticker"].map(fwd)
        valid = f_day[feature_cols].notna().all(axis=1) & f_day["ret_21d"].notna()
        if valid.sum() >= 10:
            samples.append((d, f_day[valid]))
    print(f"samples: {len(samples)}")

    def train_model(up_to_date):
        X_all, y_all = [], []
        for d, f in samples:
            if d < up_to_date:
                X_all.append(f[feature_cols])
                y_all.append(f["ret_21d"])
        if len(X_all) < 2:
            return None
        Xtr = pd.concat(X_all, ignore_index=True)
        ytr = pd.concat(y_all, ignore_index=True)
        model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=4)
        model.fit(Xtr, ytr)
        return model

    text_feat_cols = [c for c in feature_cols if "_ortho" in c]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(pd.concat([f[text_feat_cols] for _, f in samples], ignore_index=True))

    # ---- backtest ----
    print("\n" + "=" * 60)
    print("Walk-forward 回测")
    print("=" * 60)

    barra = BarraRisk()
    optimizer = PortfolioOptimizer(risk_aversion=1.0, max_single=0.10, max_sector=0.25, min_count=8, max_count=15)

    strategies = {
        "纯动量_等权top10": [],
        "纯动量_CVXPY风控": [],
        "XGBoost_等权top10": [],
        "XGBoost_CVXPY风控": [],
    }
    positions = {k: [] for k in strategies}

    for i, d in enumerate(rebal[:-1]):
        f_day = feat[feat["date"] == d].copy()
        if f_day.empty: continue
        end = rebal[i + 1]
        days = px.index[(px.index > d) & (px.index <= end)]
        if len(days) == 0: continue

        tk = f_day.set_index("ticker")
        mom_score = tk["mom_7m"].sort_values(ascending=False)
        top10 = mom_score.dropna().head(10).index.tolist()
        w_mom = pd.Series(1.0 / len(top10), index=top10) if top10 else pd.Series(dtype=float)

        model = train_model(d)
        valid = tk[feature_cols].notna().all(axis=1)
        pred = pd.Series(np.nan, index=tk.index)
        if model is not None and valid.sum() >= 5:
            pred[valid] = model.predict(tk.loc[valid, feature_cols])
        xgb_score = pred.sort_values(ascending=False)
        top10_xgb = xgb_score.dropna().head(10).index.tolist()
        w_xgb_eq = pd.Series(1.0 / len(top10_xgb), index=top10_xgb) if top10_xgb else pd.Series(dtype=float)

        cov_tickers = [t for t in xgb_score.dropna().head(30).index if t in px.columns]
        cov = daily_ret.loc[:d].tail(63)[cov_tickers].cov() if len(cov_tickers) >= 5 else None
        if cov is not None and model is not None and not w_xgb_eq.empty:
            w_xgb_opt = optimizer.optimize(xgb_score, cov, tickers=cov_tickers)
        else:
            w_xgb_opt = w_xgb_eq.copy()

        # 纯动量 + CVXPY
        cov_tickers_mom = [t for t in mom_score.dropna().head(30).index if t in px.columns]
        cov_mom = daily_ret.loc[:d].tail(63)[cov_tickers_mom].cov() if len(cov_tickers_mom) >= 5 else None
        if cov_mom is not None and not w_mom.empty:
            w_mom_opt = optimizer.optimize(mom_score, cov_mom, tickers=cov_tickers_mom)
        else:
            w_mom_opt = w_mom.copy()

        for name, w in [("纯动量_等权top10", w_mom), ("纯动量_CVXPY风控", w_mom_opt), ("XGBoost_等权top10", w_xgb_eq), ("XGBoost_CVXPY风控", w_xgb_opt)]:
            if w.empty or w.sum() == 0:
                strategies[name].append(0.0)
                positions[name].append((d, {}))
                continue
            r = (w.reindex(px.columns).fillna(0).values * daily_ret.loc[days, :].values).sum(axis=1)
            turnover = w.reindex(px.columns).fillna(0).abs().sum() / 2.0
            cost = turnover * COST_BPS / 10000.0
            monthly_ret = float(r.sum()) - cost
            strategies[name].append(monthly_ret)
            positions[name].append((d, w[w > 0].sort_values(ascending=False).to_dict()))

    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    results = {}
    for name, rets in strategies.items():
        s = summarize_monthly(rets)
        results[name] = s
        print(f"[{name}] 月均{s['mean']:.2f}% 夏普{s['sharpe']:.2f} 累计{s['cum']:.1f}% 最大回撤{s['maxdd']:.1f}% n={s['n']}")

    print("\n最近一期持仓（CVXPY风控）:")
    last_pos = positions["XGBoost_CVXPY风控"][-1]
    print(f"date: {last_pos[0].date()}")
    for t, w in sorted(last_pos[1].items(), key=lambda x: -x[1]):
        print(f"  {t}: {w:.2%}")
    print("行业暴露:")
    print(barra.sector_exposure(pd.Series(last_pos[1])))

    summary = pd.DataFrame(results).T
    summary.to_csv(out / "walkforward_v2_summary.csv", encoding="utf-8-sig")

    loader.close()
    return results


if __name__ == "__main__":
    run()
