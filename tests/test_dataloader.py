# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from data.dataloader import lag_text_factors

def test_asof_no_lookahead():
    """构造一个已知日期的因子，验证滞后后在 T 日不可用、T+1 日可用。"""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2025-06-01", "2025-06-02", "2025-06-03"]),
        "ticker": ["NVDA"] * 3,
        "sentiment": [1.0, 2.0, 3.0],
    })
    lagged = lag_text_factors(df, periods=1)
    # 2025-06-01 行应被删除（无滞后后可用因子）
    assert "2025-06-01" not in lagged["date"].dt.strftime("%Y-%m-%d").values
    # 2025-06-02 应拿到 2025-06-01 的因子值 1.0
    row = lagged[lagged["date"] == "2025-06-02"]
    assert not row.empty
    assert row["sentiment"].iloc[0] == 1.0
    print("test_asof_no_lookahead passed")

if __name__ == "__main__":
    test_asof_no_lookahead()
