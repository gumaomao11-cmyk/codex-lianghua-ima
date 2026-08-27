# -*- coding: utf-8 -*-
"""行业映射表维护与加载。"""
from pathlib import Path
import pandas as pd

DB_DIR = Path(__file__).resolve().parent.parent / "data" / "duckdb"

SECTOR_MAP = {
    "科技": ["NVDA", "AMD", "AVGO", "QCOM", "ARM", "MRVL", "INTC", "SMCI", "TSM"],
    "半导体设备": ["AMAT", "LRCX", "KLAC", "TER"],
    "存储硬件": ["MU", "WDC", "STX", "SNDK"],
    "互联网软件": ["META", "MSFT", "GOOGL", "AMZN", "NFLX", "UBER", "PLTR", "DDOG", "NOW", "BABA", "PDD", "JD", "NTES", "BIDU", "TCEHY", "AKAM", "APP", "DASH", "CRWV"],
    "消费电子": ["AAPL", "TSLA", "DELL", "HPE", "GLW", "FLEX", "LITE", "CDW", "CSCO"],
    "通信": ["CIEN", "ANET"],
    "生物科技": ["MRNA"],
    "金融科技": ["PYPL", "COIN"],
    "其他": ["NBIS", "CCL", "HST", "DVA", "MPC", "VLO"],
}

def get_industry_map():
    """返回 ticker -> sector 的 DataFrame。"""
    rows = []
    for sector, tickers in SECTOR_MAP.items():
        for t in tickers:
            rows.append({"ticker": t, "sector": sector})
    return pd.DataFrame(rows).drop_duplicates(subset=["ticker"])

def save_default():
    df = get_industry_map()
    df.to_parquet(DB_DIR / "industry_map.parquet", index=False)

if __name__ == "__main__":
    save_default()
    print(get_industry_map()["sector"].value_counts())
