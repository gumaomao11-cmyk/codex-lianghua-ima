# -*- coding: utf-8 -*-
"""Walk-forward v4：滚动调仓 + 动态跟踪止盈/止损 + Delta Rebalancing。

改进点：
- 每日重新打分，但只交易目标权重与当前持仓的差额（Delta Rebalancing）
- 对强势持仓增加 Alpha 溢价，避免“到期强平”错杀主升浪
- 引入移动止盈：从买入后最高点回撤超过 trail_pct 强制平仓
- 出局条件：因子转负 + 价格跌破 5 日 EMA
"""
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
DELTA_THRESHOLD = 0.01     # 权重变动超过 1% 才调仓
TRAIL_PCT = 0.05           # 移动止盈：从最高点回撤 5%
TREND_PREMIUM = 0.02       # 趋势延续 Alpha 溢价
PROFIT_THRESHOLD = 0.05    # 浮盈超过 5% 才考虑延期平仓
EMA_FAST = 5
EMA_SLOW = 20
HIGH_WINDOW = 5


def load_dataset(route):
    path = DB_DIR / f"aligned_dataset_{route.lower()}_ortho.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    df["ema5"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=EMA_FAST, min_periods=1).mean())
    df["ema20"] = df.groupby("ticker")["close"].transform(lambda x: x.ewm(span=EMA_SLOW, min_periods=1).mean())
    df["high5"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(HIGH_WINDOW, min_periods=1).max())
    return df


def get_factor_cols(df):
    return [c for c in df.columns if c.startswith("factor_") and "ortho" in c]


def walkforward(df, route):
    df = df.sort_values(["ticker", "date"])
    factor_cols = get_factor_cols(df)
    print(f"[route {route}] factors: {factor_cols}")

    sector_map = get_industry_map().set_index("ticker")["sector"].to_dict()

    dates = df["date"].unique()
    dates = sorted(dates)
    date_idx = {d: i for i, d in enumerate(dates)}

    # 每个 ticker 的 close 序列，用于 trailing stop 判断
    px_wide = df.pivot_table(index="date", columns="ticker", values="close")

    current_weights = {}   # ticker -> weight
    entry_prices = {}      # ticker -> entry price
    max_prices = {}        # ticker -> max price since entry
    daily_records = []
    trade_records = []

    ret_target = "ret_21d"  # XGB 预测目标仍用 21 日收益

    last_model = None
    start_date = max(dates[30], pd.Timestamp("2026-01-01"))
    for i, t in enumerate(dates):
        if t < start_date:
            continue

        today_df = df[df["date"] == t].copy()
        if today_df.empty:
            continue

        # 每 5 个交易日重新训练一次，中间沿用上一次的模型
        if last_model is None or i % 5 == 0:
            train_start = t - pd.DateOffset(months=6)
            train_df = df[(df["date"] >= train_start) & (df["date"] < t)].copy()
            if len(train_df) < 50:
                continue
            train = train_df.dropna(subset=[ret_target]).copy()
            if len(train) < 50:
                continue
            model = xgb.XGBRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
                random_state=42, n_jobs=4
            )
            model.fit(train[factor_cols].fillna(0), train[ret_target])
            last_model = model
        else:
            model = last_model

        today_df["alpha_pred"] = model.predict(today_df[factor_cols].fillna(0))

        # 今日价格相关指标
        today_row = today_df.set_index("ticker")

        # 对每个当前持仓检查移动止盈和趋势延续
        alpha_scores = today_row["alpha_pred"].to_dict()
        for tk, w in list(current_weights.items()):
            if tk not in today_row.index:
                continue
            close_t = today_row.loc[tk, "close"]
            ema5 = today_row.loc[tk, "ema5"]
            ema20 = today_row.loc[tk, "ema20"]
            high5 = today_row.loc[tk, "high5"]
            pnl = close_t / entry_prices[tk] - 1 if entry_prices[tk] > 0 else 0
            max_prices[tk] = max(max_prices.get(tk, close_t), close_t)

            # 移动止盈：从最高点回撤超过 5% 强制平仓
            if close_t < max_prices[tk] * (1 - TRAIL_PCT):
                alpha_scores[tk] = -1.0  # 强制卖出
                trade_records.append({"date": t, "ticker": tk, "action": "trailing_stop", "pnl": pnl})
                continue

            # 出局条件：因子转负 + 跌破 5 日 EMA
            if alpha_scores.get(tk, 0) < 0 and close_t < ema5:
                alpha_scores[tk] = -1.0
                trade_records.append({"date": t, "ticker": tk, "action": "factor_exit", "pnl": pnl})
                continue

            # 让利润奔跑：浮盈 > 5% 且价格在 20 日均线上方并创 5 日新高
            if pnl > PROFIT_THRESHOLD and close_t > ema20 and close_t >= high5 * 0.999:
                alpha_scores[tk] = alpha_scores.get(tk, 0) + TREND_PREMIUM

        # 构建候选池：Alpha 前 30
        candidates = pd.Series(alpha_scores).dropna().sort_values(ascending=False).head(30)
        candidates = candidates[candidates > 0]
        if len(candidates) < 8:
            # 没有信号则空仓
            target_weights = pd.Series(dtype=float)
        else:
            sub = today_row.loc[candidates.index].copy()
            sub["alpha_pred"] = candidates
            try:
                # 计算过去 60 日收益协方差
                end_idx = px_wide.index.get_indexer([t], method="nearest")[0]
                start_idx = max(0, end_idx - 60)
                hist_rets = px_wide.iloc[start_idx:end_idx].pct_change().fillna(0)
                cov = hist_rets[sub.index].cov() * 252
                cov = cov.reindex(index=sub.index, columns=sub.index).fillna(0)

                opt = PortfolioOptimizer(risk_aversion=1.0, max_single=0.10, max_sector=0.25, min_count=8, max_count=15)
                alpha = sub["alpha_pred"]
                weights = opt.optimize(alpha, cov, tickers=sub.index.tolist())
                target_weights = weights.reindex(sub.index).fillna(0)
                target_weights = target_weights[target_weights > 0.001]
                target_weights = target_weights / target_weights.sum() if target_weights.sum() > 0 else target_weights
            except Exception as e:
                print(f"  optimize error at {t}: {e}")
                target_weights = candidates.head(10)
                target_weights = target_weights / target_weights.sum()

        # Delta Rebalancing
        all_tickers = set(current_weights.keys()) | set(target_weights.index)
        delta = {}
        for tk in all_tickers:
            old_w = current_weights.get(tk, 0.0)
            new_w = target_weights.get(tk, 0.0) if tk in target_weights.index else 0.0
            if abs(new_w - old_w) > DELTA_THRESHOLD:
                delta[tk] = new_w - old_w

        # 交易成本按总买入金额计算
        buy_total = sum(v for v in delta.values() if v > 0)
        cost = COST_BPS * buy_total

        # 当日收益：使用调仓后权重 * 当日未来 1 日收益
        today_ret = today_row["ret_1d"].reindex(target_weights.index).fillna(0)
        port_ret = (target_weights * today_ret).sum() - cost

        # 记录
        daily_records.append({
            "date": t, "route": route,
            "xgb_dynamic": port_ret,
            "n_holdings": len(target_weights),
            "turnover": buy_total,
            "cost": cost,
        })

        # 更新持仓
        for tk, dw in delta.items():
            new_w = current_weights.get(tk, 0.0) + dw
            if new_w <= 0.001:
                if tk in current_weights:
                    del current_weights[tk]
                entry_prices.pop(tk, None)
                max_prices.pop(tk, None)
            else:
                if tk not in current_weights:
                    entry_prices[tk] = today_row.loc[tk, "close"]
                    max_prices[tk] = today_row.loc[tk, "close"]
                current_weights[tk] = new_w

        # 重新归一化，确保总权重为 1
        total = sum(current_weights.values())
        if total > 0:
            current_weights = {k: v / total for k, v in current_weights.items()}

    return pd.DataFrame(daily_records), pd.DataFrame(trade_records)


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
    all_trades = []
    for route in ["A", "B"]:
        print(f"\n{'='*60}\n[route {route}]\n{'='*60}")
        df = load_dataset(route)
        perf, trades = walkforward(df, route)
        if not perf.empty:
            print(f"Dynamic Delta Rebal: {summarize(perf['xgb_dynamic'])}")
            results.append(perf)
            all_trades.append(trades)

    if results:
        all_perf = pd.concat(results, ignore_index=True)
        out = PROJ / "backtest_output" / "walkforward_v4_dynamic_results.csv"
        all_perf.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n[saved] {out}")

    if all_trades:
        trades_df = pd.concat(all_trades, ignore_index=True)
        trades_out = PROJ / "backtest_output" / "walkforward_v4_dynamic_trades.csv"
        trades_df.to_csv(trades_out, index=False, encoding="utf-8-sig")
        print(f"[saved] {trades_out}")


if __name__ == "__main__":
    main()
