# -*- coding: utf-8 -*-
"""因子正交化：剔除传统量价 Beta，保留纯文本 Alpha。
输入: data/duckdb/aligned_dataset_a.parquet, aligned_dataset_b.parquet
输出: data/duckdb/aligned_dataset_a_ortho.parquet, aligned_dataset_b_ortho.parquet
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROJ = Path(r"F:\even-codex\lianghua+IMA")
DB_DIR = PROJ / "data" / "duckdb"

CONTROL_COLS = ["mom_20d", "vol_20d", "rsi_14"]

def orthogonalize_series(y, X):
    """对单截面做线性回归，返回残差。"""
    valid = y.notna() & X.notna().all(axis=1)
    yv = y[valid]
    Xv = X[valid]
    if len(yv) < 10 or Xv.shape[1] == 0:
        return pd.Series(np.nan, index=y.index)
    model = LinearRegression().fit(Xv, yv)
    pred = pd.Series(np.nan, index=y.index)
    pred[valid] = model.predict(Xv)
    return y - pred

def orthogonalize_dataset(path_in, path_out):
    df = pd.read_parquet(path_in)
    print(f"[load] {path_in}: {df.shape}")

    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    print(f"[factors] {factor_cols}")

    # 截面正交化：每个交易日横截面回归
    out_frames = []
    for date, g in df.groupby("date"):
        g = g.copy()
        X = g[CONTROL_COLS].copy()
        for col in factor_cols:
            g[f"{col}_ortho"] = orthogonalize_series(g[col], X)
        out_frames.append(g)

    df_out = pd.concat(out_frames, ignore_index=True)
    df_out.to_parquet(path_out, index=False)
    print(f"[saved] {path_out}: {df_out.shape}")

    # 打印正交前后与 control 的相关性
    for col in factor_cols:
        corr_before = df[col].corr(df["mom_20d"])
        corr_after = df_out[f"{col}_ortho"].corr(df_out["mom_20d"])
        print(f"  {col}: corr(mom_20d) before={corr_before:.3f} after={corr_after:.3f}")

def main():
    orthogonalize_dataset(DB_DIR / "aligned_dataset_a.parquet", DB_DIR / "aligned_dataset_a_ortho.parquet")
    orthogonalize_dataset(DB_DIR / "aligned_dataset_b.parquet", DB_DIR / "aligned_dataset_b_ortho.parquet")
    print("[done]")

if __name__ == "__main__":
    main()
