# -*- coding: utf-8 -*-
"""构建对齐数据集：细粒度文本因子 + 价格 + 未来收益。
输入: data/duckdb/zsxq_19_26_granular_events.parquet, data/duckdb/prices.parquet
输出: data/duckdb/aligned_dataset_a.parquet (路线A), aligned_dataset_b.parquet (路线B)
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(r"F:\even-codex\lianghua+IMA")
DB_DIR = PROJ / "data" / "duckdb"


def load_prices():
    px = pd.read_parquet(DB_DIR / "prices.parquet")
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values("date").set_index("date")
    return px


def load_events():
    ev = pd.read_parquet(DB_DIR / "zsxq_19_26_granular_events.parquet")
    ev["event_time"] = pd.to_datetime(ev["event_time"])
    ev["date"] = ev["event_time"].dt.normalize()
    return ev


def build_route_a_factors(events):
    """路线A：只保留 research_report + single_event + tier_1_hard_data。"""
    clean = events[
        
        (events["text_type"].isin(["research_report", "single_event"]) &
         (events["materiality_tier"] == "tier_1_hard_data"))
    ].copy()
    print(f"[route A] clean events: {len(clean)}")
    # 按 date+ticker 聚合
    fa = clean.groupby(["date", "ticker"])["raw_signal"].mean().reset_index()
    fa = fa.rename(columns={"raw_signal": "factor_clean_alpha"})
    return fa


def build_route_b_factors(events):
    """路线B：分池衰减多因子。"""
    # 先构建日级别原始信号
    daily = events.groupby(["date", "ticker", "text_type"]).agg(
        raw_signal=("raw_signal", "mean"),
        count=("raw_signal", "count")
    ).reset_index()

    # 展开为宽表
    pivoted = daily.pivot_table(
        index=["date", "ticker"],
        columns="text_type",
        values="raw_signal",
        aggfunc="mean"
    ).reset_index()

    # 重命名列
    rename_map = {}
    for col in ["research_report", "single_event", "news_summary", "personal_opinion"]:
        if col in pivoted.columns:
            rename_map[col] = f"raw_{col}"
    pivoted = pivoted.rename(columns=rename_map)

    # 计算各池子的衰减因子
    result = []
    for ticker, g in pivoted.groupby("ticker"):
        g = g.sort_values("date").set_index("date")

        if "raw_research_report" in g.columns:
            g["factor_research_20d"] = g["raw_research_report"].ewm(span=20, min_periods=1).mean()
        if "raw_single_event" in g.columns:
            g["factor_event_3d"] = g["raw_single_event"].rolling(window=3, min_periods=1).mean()
        if "raw_news_summary" in g.columns:
            g["factor_news_1d"] = g["raw_news_summary"]
        if "raw_personal_opinion" in g.columns:
            g["factor_opinion_1d"] = -g["raw_personal_opinion"]  # 默认反向

        g = g.reset_index()
        result.append(g)

    if result:
        fb = pd.concat(result, ignore_index=True)
    else:
        fb = pd.DataFrame(columns=["date", "ticker"])
    print(f"[route B] tickers: {fb['ticker'].nunique()}")
    return fb


def align_to_prices(factors, prices, route_name):
    """将因子对齐到价格数据，并计算未来收益。"""
    # 把价格从宽格式转为长格式
    price_long = prices.reset_index().melt(id_vars=["date"], var_name="ticker", value_name="close")
    price_long = price_long.dropna(subset=["close"])
    price_long["date"] = pd.to_datetime(price_long["date"])

    # 合并因子
    merged = price_long.merge(factors, on=["date", "ticker"], how="left")

    # 对每个 ticker 计算未来收益和滞后因子
    result = []
    for ticker, g in merged.groupby("ticker"):
        g = g.sort_values("date")
        # lag 1 日：确保 T 日因子最早 T+1 日使用
        factor_cols = [c for c in g.columns if c.startswith("factor_")]
        for c in factor_cols:
            g[c] = g[c].shift(1)
        # 前向填充：事件信号在后续交易日保持有效，避免因子过于稀疏
        for c in factor_cols:
            g[c] = g[c].ffill()
        # 未来收益
        for h in [1, 5, 10, 21]:
            g[f"ret_{h}d"] = g["close"].shift(-h) / g["close"] - 1.0
        # 传统量价控制变量
        g["mom_20d"] = g["close"].pct_change(20)
        g["vol_20d"] = g["close"].pct_change().rolling(20).std() * np.sqrt(252)
        delta = g["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        g["rsi_14"] = 100 - (100 / (1 + rs))
        g["turnover_20d"] = g["close"].rolling(20).mean()  # 没有 volume 数据，用价格移动平均占位
        result.append(g)

    df = pd.concat(result, ignore_index=True)
    df["route"] = route_name
    return df


def main():
    print("[load] prices and events")
    prices = load_prices()
    events = load_events()
    print(f"[prices] {prices.shape}, date range: {prices.index.min().date()} ~ {prices.index.max().date()}")
    print(f"[events] {len(events)}, date range: {events['date'].min().date()} ~ {events['date'].max().date()}")

    print("\n[build] route A: clean alpha")
    fa = build_route_a_factors(events)
    aligned_a = align_to_prices(fa, prices, "A")
    out_a = DB_DIR / "aligned_dataset_a.parquet"
    aligned_a.to_parquet(out_a, index=False)
    print(f"[saved] {out_a}: {aligned_a.shape}")

    print("\n[build] route B: multi-factor decay")
    fb = build_route_b_factors(events)
    aligned_b = align_to_prices(fb, prices, "B")
    out_b = DB_DIR / "aligned_dataset_b.parquet"
    aligned_b.to_parquet(out_b, index=False)
    print(f"[saved] {out_b}: {aligned_b.shape}")

    print("\n[done]")

if __name__ == "__main__":
    main()
