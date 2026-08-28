# -*- coding: utf-8 -*-
"""因子 IC / IR 评估（日频）。
输入: data/duckdb/aligned_dataset_a_ortho.parquet, aligned_dataset_b_ortho.parquet
输出: reports/zsxq_19_26_factor_ic_report.md
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJ = Path(r"F:\even-codex\lianghua+IMA")
DB_DIR = PROJ / "data" / "duckdb"
REPORT = PROJ / "reports" / "zsxq_19_26_factor_ic_report.md"


def rank_ic(x, y):
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    idx = x.index.intersection(y.index)
    if len(idx) < 5 or x.loc[idx].nunique() <= 1 or y.loc[idx].nunique() <= 1:
        return np.nan
    ic, _ = spearmanr(x.loc[idx], y.loc[idx], nan_policy="omit")
    return ic


def evaluate(df, factor_col, forward_col="ret_21d"):
    daily_ics = []
    for date, g in df.groupby("date"):
        g = g.dropna(subset=[factor_col, forward_col])
        if len(g) < 5:
            continue
        ic = rank_ic(g[factor_col], g[forward_col])
        if not np.isnan(ic):
            daily_ics.append({"day": date, "ic": ic, "n": len(g)})
    ic_df = pd.DataFrame(daily_ics)
    if len(ic_df) < 2:
        return {"mean_ic": np.nan, "ir": np.nan, "ic_std": np.nan, "n_days": 0}
    mean_ic = ic_df["ic"].mean()
    ic_std = ic_df["ic"].std(ddof=1)
    ir = mean_ic / ic_std if ic_std != 0 else np.nan
    return {"mean_ic": mean_ic, "ir": ir, "ic_std": ic_std, "n_days": len(ic_df)}


def process_route(route_name, path_in):
    df = pd.read_parquet(path_in)
    factor_cols = [c for c in df.columns if c.endswith("_ortho")]
    if not factor_cols:
        factor_cols = [c for c in df.columns if c.startswith("factor_")]

    results = []
    for col in factor_cols:
        for fwd in ["ret_1d", "ret_5d", "ret_10d", "ret_21d"]:
            if fwd not in df.columns:
                continue
            r = evaluate(df, col, fwd)
            r["factor"] = col
            r["forward"] = fwd
            results.append(r)

    return pd.DataFrame(results)


def main():
    REPORT.parent.mkdir(exist_ok=True)

    res_a = process_route("A", DB_DIR / "aligned_dataset_a_ortho.parquet")
    res_b = process_route("B", DB_DIR / "aligned_dataset_b_ortho.parquet")
    res_a["route"] = "A"
    res_b["route"] = "B"
    res = pd.concat([res_a, res_b], ignore_index=True)

    best_rows = []
    for route_name, g in res.groupby("route"):
        g2 = g.dropna(subset=["ir"])
        if not g2.empty:
            best_rows.append(g2.loc[g2["ir"].idxmax()])
    best = pd.DataFrame(best_rows) if best_rows else pd.DataFrame(columns=res.columns)

    lines = ["# 19-26 年星球文本因子 IC / IR 评估报告\n"]
    lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("\n## 全部因子评估\n")
    lines.append(res.to_string(index=False))
    lines.append("\n\n## 各路线最优因子\n")
    lines.append(best.to_string(index=False))

    lines.append("\n\n## 因子门禁结果\n")
    passed = res[(res["mean_ic"].abs() > 0.02) & (res["ir"] > 0.2)]
    if len(passed) > 0:
        lines.append(f"通过门禁的因子数: {len(passed)}\n")
        lines.append(passed.to_string(index=False))
    else:
        lines.append("没有因子通过门禁 (|IC|>0.02, IR>0.2)\n")

    REPORT.write_text("".join(lines), encoding="utf-8")
    print(f"[saved] {REPORT}")
    print(res.to_string(index=False))

if __name__ == "__main__":
    main()
