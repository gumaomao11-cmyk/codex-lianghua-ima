# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from alpha.orthogonal import orthogonalize_factors

def test_orthogonal_low_corr():
    np.random.seed(42)
    df = pd.DataFrame({
        "ticker": ["A"] * 100 + ["B"] * 100,
        "date": pd.date_range("2025-01-01", periods=100).tolist() * 2,
        "mom_7m": np.random.randn(200),
        "text_tsm": np.random.randn(200) + 0.5 * np.random.randn(200),
    })
    df["text_tsm"] = df["text_tsm"] + 0.3 * df["mom_7m"]
    df = orthogonalize_factors(df, ["text_tsm"], ["mom_7m"], method="gs")
    for tk, g in df.groupby("ticker"):
        corr = g["text_tsm_ortho"].corr(g["mom_7m"])
        assert abs(corr) < 0.1, f"corr={corr}"
    print("test_orthogonal_low_corr passed")

if __name__ == "__main__":
    test_orthogonal_low_corr()
