# -*- coding: utf-8 -*-
"""一次性迁移脚本：把 CSV 数据导入 DuckDB + Parquet 统一存储。"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "duckdb"
DB_DIR.mkdir(parents=True, exist_ok=True)

PRICES_CSV = Path(r"F:\even-codex\us-stock-data") / "prices.csv"
ETF_CSV = Path(r"F:\even-codex\panda\backtest") / "prices_2016.csv"
OUT = DB_DIR

# ---- 1. prices.parquet ----
print("migrating prices...")
px = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
px = px.apply(pd.to_numeric, errors="coerce")
px = px.reset_index().rename(columns={"index": "date"})
px.to_parquet(OUT / "prices.parquet", index=False)
print(f"  prices rows={len(px)}, cols={len(px.columns)-1}, date_range={px['date'].min()} ~ {px['date'].max()}")

# ---- 2. text_factors.parquet ----
print("migrating text factors...")
files = [
    (ROOT / "backtest_output" / "text_sentiment_lexicon_浑水调研Plus.csv", "hs"),
    (ROOT / "backtest_output" / "text_sentiment_ima.csv", "ima"),
    (ROOT / "backtest_output" / "text_sentiment_lexicon_duanping.csv", "dp"),
]
frames = []
for f, prefix in files:
    if not f.exists():
        print(f"  skip missing {f}")
        continue
    df = pd.read_csv(f, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    # 重命名列为 prefix_*
    rename = {}
    for c in df.columns:
        if c not in ["date", "ticker", "media_id", "text"]:
            rename[c] = f"{prefix}_{c}"
    df = df.rename(columns=rename)
    keep = ["date", "ticker"] + [c for c in df.columns if c.startswith(f"{prefix}_")]
    frames.append(df[keep])

# 合并所有文本因子
text = frames[0]
for f in frames[1:]:
    text = text.merge(f, on=["date", "ticker"], how="outer")
for c in text.columns:
    if c not in ["date", "ticker"]:
        text[c] = text[c].fillna(0)
text = text.sort_values(["date", "ticker"]).reset_index(drop=True)
text.to_parquet(OUT / "text_factors.parquet", index=False)
print(f"  text_factors rows={len(text)}, tickers={text['ticker'].nunique()}, dates={text['date'].nunique()}")

# ---- 3. etf_ref.parquet ----
print("migrating ETF reference...")
ref = pd.read_csv(ETF_CSV, index_col=0, parse_dates=True)
ref = ref[["SPY", "QQQ"]].apply(pd.to_numeric, errors="coerce")
ref = ref.reset_index().rename(columns={"index": "date"})
ref.to_parquet(OUT / "etf_ref.parquet", index=False)
print(f"  etf_ref rows={len(ref)}, date_range={ref['date'].min()} ~ {ref['date'].max()}")

# ---- 4. industry_map.parquet ----
print("migrating industry map...")
sector_map = {
    # 科技 / AI 芯片
    "NVDA": "科技", "AMD": "科技", "AVGO": "科技", "QCOM": "科技",
    "ARM": "科技", "MRVL": "科技", "INTC": "科技", "SMCI": "科技",
    # 半导体设备/材料
    "AMAT": "半导体设备", "LRCX": "半导体设备", "KLAC": "半导体设备",
    # 存储/硬件
    "MU": "存储硬件", "WDC": "存储硬件", "STX": "存储硬件", "SNDK": "存储硬件",
    # 互联网/软件
    "META": "互联网软件", "MSFT": "互联网软件", "GOOGL": "互联网软件",
    "AMZN": "互联网软件", "NFLX": "互联网软件", "UBER": "互联网软件",
    "PLTR": "互联网软件", "DDOG": "互联网软件", "NOW": "互联网软件",
    # 其他/消费电子
    "AAPL": "消费电子", "TSLA": "消费电子", "DELL": "消费电子",
    "HPE": "消费电子", "GLW": "消费电子", "FLEX": "消费电子",
    "LITE": "消费电子", "CCL": "消费电子", "NBIS": "消费电子",
    "CSCO": "消费电子", "HST": "其他", "DVA": "其他", "MPC": "其他",
    "VLO": "其他", "TER": "半导体设备", "CDW": "消费电子",
    "CIEN": "通信", "AKAM": "互联网软件", "APP": "互联网软件",
    "DASH": "互联网软件", "CRWV": "互联网软件", "MRNA": "生物科技",
    "PYPL": "金融科技", "ANET": "通信", "COIN": "金融科技",
    "TSM": "科技", "BABA": "互联网软件", "PDD": "互联网软件",
    "JD": "互联网软件", "NTES": "互联网软件", "BIDU": "互联网软件",
    "TCEHY": "互联网软件",
}
ind = pd.DataFrame([{"ticker": k, "sector": v} for k, v in sector_map.items()])
ind.to_parquet(OUT / "industry_map.parquet", index=False)
print(f"  industry_map rows={len(ind)}")

# ---- 5. 创建 DuckDB 视图 ----
print("creating DuckDB views...")
con = duckdb.connect(str(OUT / "quant_store.duckdb"))
con.execute(f"CREATE OR REPLACE VIEW prices AS SELECT * FROM read_parquet('{OUT / 'prices.parquet'}')")
con.execute(f"CREATE OR REPLACE VIEW text_factors AS SELECT * FROM read_parquet('{OUT / 'text_factors.parquet'}')")
con.execute(f"CREATE OR REPLACE VIEW etf_ref AS SELECT * FROM read_parquet('{OUT / 'etf_ref.parquet'}')")
con.execute(f"CREATE OR REPLACE VIEW industry_map AS SELECT * FROM read_parquet('{OUT / 'industry_map.parquet'}')")
con.close()
print("done.")
