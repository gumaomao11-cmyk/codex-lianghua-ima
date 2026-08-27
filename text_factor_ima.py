# -*- coding: utf-8 -*-
"""从 IMA 美国科技日报摘要因子构建文本-动能因子。
输入: backtest_output/kb_abstract_factors.csv
输出: backtest_output/text_sentiment_ima.csv
"""
import os
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).parent / "backtest_output"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(OUT / "kb_abstract_factors.csv", encoding="utf-8-sig")
# 只取美国科技日报
ima = df[df["source_folder"].str.contains("美国科技日报", na=False)].copy()
print(f"IMA 科技日报: {len(ima)} 条, tickers={ima['ticker'].nunique()}, dates={ima['pdf_date'].nunique()}")

ima["date"] = pd.to_datetime(ima["pdf_date"])
# 基础日度 sentiment
ima["sentiment"] = ima["sign"]  # -1/0/1
ima["sentiment_score"] = (ima["n_pos"] - ima["n_neg"]) / (ima["n_pos"] + ima["n_neg"] + 1)

# 按 ticker-date 聚合
g = ima.groupby(["ticker", "date"]).agg(
    n_reports=("title", "nunique"),
    n_pos=("n_pos", "sum"),
    n_neg=("n_neg", "sum"),
    sentiment_sum=("sentiment", "sum"),
    sentiment_mean=("sentiment", "mean"),
    score_mean=("sentiment_score", "mean"),
).reset_index()

# 生成完整日期网格，方便滚动计算
tickers = g["ticker"].unique()
dates = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
grid = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
grid_df = pd.DataFrame(index=grid).reset_index()
merged = grid_df.merge(g, on=["date", "ticker"], how="left")
merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

# 填充 0 表示当天无报道
for c in ["n_reports", "n_pos", "n_neg", "sentiment_sum", "sentiment_mean", "score_mean"]:
    merged[c] = merged[c].fillna(0)

# 滚动特征
for ticker, grp in merged.groupby("ticker"):
    grp = grp.sort_values("date")
    # EMA 情绪
    merged.loc[grp.index, "ema3"] = grp["score_mean"].ewm(span=3, adjust=False).mean()
    merged.loc[grp.index, "ema20"] = grp["score_mean"].ewm(span=20, adjust=False).mean()
    # 覆盖度（是否有报道）
    merged.loc[grp.index, "cov_7d"] = (grp["n_reports"] > 0).rolling(7, min_periods=1).mean()
    merged.loc[grp.index, "cov_20d"] = (grp["n_reports"] > 0).rolling(20, min_periods=1).mean()
    # 累计 pos/neg
    merged.loc[grp.index, "pos_7d"] = grp["n_pos"].rolling(7, min_periods=1).sum()
    merged.loc[grp.index, "neg_7d"] = grp["n_neg"].rolling(7, min_periods=1).sum()
    merged.loc[grp.index, "pos_20d"] = grp["n_pos"].rolling(20, min_periods=1).sum()
    merged.loc[grp.index, "neg_20d"] = grp["n_neg"].rolling(20, min_periods=1).sum()

# TSM/TNA/Disagreement
merged["tsm"] = merged["ema3"] - merged["ema20"]
# 新颖度：用 7 天累计 pos+neg 的变化幅度近似
merged["momentum_7d"] = merged.groupby("ticker")["score_mean"].transform(lambda x: x.rolling(7, min_periods=1).mean())
merged["momentum_20d"] = merged.groupby("ticker")["score_mean"].transform(lambda x: x.rolling(20, min_periods=1).mean())
merged["tna"] = merged["score_mean"] * (1 - (merged["momentum_7d"] / (merged["momentum_20d"].abs() + 1e-6)).abs().clip(0, 1))
# 分歧：7 天 vs 20 天情绪方差差异
merged["var_7d"] = merged.groupby("ticker")["score_mean"].transform(lambda x: x.rolling(7, min_periods=1).var())
merged["var_20d"] = merged.groupby("ticker")["score_mean"].transform(lambda x: x.rolling(20, min_periods=1).var())
merged["disagreement"] = (merged["var_7d"] - merged["var_20d"]).clip(lower=-1, upper=1)
# 风险标记：最近 3 天累计 neg > pos * 2 且 neg >= 2
merged["risk_flag"] = ((merged["neg_7d"] > merged["pos_7d"] * 2) & (merged["neg_7d"] >= 2)).astype(int)
# 文本覆盖标记
merged["text_cov"] = (merged["cov_20d"] > 0).astype(int)

# 整理输出列
out = merged[["date", "ticker", "n_reports", "n_pos", "n_neg",
              "score_mean", "ema3", "ema20", "tsm", "tna", "disagreement",
              "risk_flag", "text_cov", "cov_7d", "cov_20d"]].copy()
out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
out.to_csv(OUT / "text_sentiment_ima.csv", index=False, encoding="utf-8-sig")
print(f"saved {OUT / 'text_sentiment_ima.csv'} rows={len(out)}")
print(out.tail(10))
