# -*- coding: utf-8 -*-
"""统一数据加载器：DuckDB + Parquet + As-of Join（无未来函数）。"""
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb

DB_DIR = Path(__file__).resolve().parent / "duckdb"
DB_PATH = DB_DIR / "quant_store.duckdb"

class UnifiedLoader:
    def __init__(self, db_path=None):
        self.db_path = Path(db_path or DB_PATH)
        self.con = duckdb.connect(str(self.db_path), read_only=True)

    def _dates_in_range(self, start, end):
        return pd.date_range(start=start, end=end, freq="B")

    def load_prices(self, start=None, end=None, tickers=None):
        """加载日 K 线价格。"""
        where = []
        if start: where.append(f"date >= '{pd.Timestamp(start).date()}'")
        if end: where.append(f"date <= '{pd.Timestamp(end).date()}'")
        sql = "SELECT * FROM prices"
        if where:
            sql += " WHERE " + " AND ".join(where)
        df = self.con.execute(sql).fetchdf()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        if tickers:
            avail = [c for c in tickers if c in df.columns]
            df = df[avail]
        return df

    def load_text_factors(self, start=None, end=None, tickers=None, asof=True):
        """加载文本因子；asof=True 时用 As-of Join 对齐到交易日 T+1。"""
        start = pd.Timestamp(start) if start else pd.Timestamp("2020-01-01")
        end = pd.Timestamp(end) if end else pd.Timestamp.now()
        # 交易日网格
        trading_days = self.con.execute(
            f"SELECT DISTINCT date FROM prices WHERE date BETWEEN '{start.date()}' AND '{end.date()}' ORDER BY date"
        ).fetchdf()
        trading_days["date"] = pd.to_datetime(trading_days["date"])
        if asof:
            # DuckDB ASOF JOIN：把文本因子对齐到每个交易日，取 <= 该交易日的最新因子
            sql = f"""
                SELECT t.date, tf.* EXCLUDE(date)
                FROM (
                    SELECT date FROM prices
                    WHERE date BETWEEN '{start.date()}' AND '{end.date()}'
                ) t
                ASOF JOIN text_factors tf
                ON t.date >= tf.date
            """
            df = self.con.execute(sql).fetchdf()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
            # 解释：文本在 T 日发布，ASOF JOIN 后会在 T 日及之后的交易日都可见；
            # 策略调用时应再 lag 1 日，确保 T 日因子最早 T+1 日使用。
            df = df.dropna(subset=["ticker"])
        else:
            sql = "SELECT * FROM text_factors"
            cond = []
            if start: cond.append(f"date >= '{start.date()}'")
            if end: cond.append(f"date <= '{end.date()}'")
            if cond: sql += " WHERE " + " AND ".join(cond)
            df = self.con.execute(sql).fetchdf()
            df["date"] = pd.to_datetime(df["date"])
        if tickers:
            df = df[df["ticker"].isin(tickers)]
        return df.reset_index(drop=True)

    def load_returns(self, start=None, end=None, tickers=None):
        """加载日收益率。"""
        px = self.load_prices(start, end, tickers)
        return px.pct_change().fillna(0.0)

    def load_etf_ref(self, start=None, end=None):
        """加载 SPY/QQQ 价格。"""
        where = []
        if start: where.append(f"date >= '{pd.Timestamp(start).date()}'")
        if end: where.append(f"date <= '{pd.Timestamp(end).date()}'")
        sql = "SELECT * FROM etf_ref"
        if where: sql += " WHERE " + " AND ".join(where)
        df = self.con.execute(sql).fetchdf()
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()

    def load_industry_map(self):
        """加载行业映射。"""
        return self.con.execute("SELECT * FROM industry_map").fetchdf()

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def lag_text_factors(df, periods=1):
    """对文本因子再做一日滞后，确保 T 日因子最早 T+1 日开盘前可用。"""
    df = df.sort_values(["ticker", "date"]).copy()
    factor_cols = [c for c in df.columns if c not in ["date", "ticker"]]
    df[factor_cols] = df.groupby("ticker")[factor_cols].shift(periods)
    return df.dropna(subset=factor_cols, how="all").reset_index(drop=True)
