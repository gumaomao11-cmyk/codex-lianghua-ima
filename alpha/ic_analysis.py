# -*- coding: utf-8 -*-
"""因子 IC / IR 评估。"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)


class FactorIC:
    def __init__(self, df, date_col="date", ticker_col="ticker"):
        self.df = df.copy()
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.df[date_col] = pd.to_datetime(self.df[date_col])

    def _corr(self, x, y):
        if x.nunique() <= 1 or y.nunique() <= 1:
            return np.nan
        try:
            ic, _ = spearmanr(x, y, nan_policy="omit")
            return ic
        except Exception:
            return np.nan

    def ic_by_month(self, factor_col, forward_col="ret_21d"):
        """计算每个月的 Spearman IC。"""
        ics = []
        self.df["ym"] = self.df[self.date_col].dt.to_period("M")
        for (ym,), g in self.df.groupby(["ym"]):
            g = g.dropna(subset=[factor_col, forward_col])
            if len(g) < 5:
                continue
            ic = self._corr(g[factor_col], g[forward_col])
            if not np.isnan(ic):
                ics.append({"month": str(ym), "ic": ic, "n": len(g)})
        return pd.DataFrame(ics)

    def ir(self, factor_col, forward_col="ret_21d"):
        """IC 均值 / IC 标准差。"""
        ic_df = self.ic_by_month(factor_col, forward_col)
        if len(ic_df) < 2 or ic_df["ic"].std() == 0:
            return np.nan
        return ic_df["ic"].mean() / ic_df["ic"].std(ddof=1)

    def ic_decay(self, factor_col, horizons=[1, 5, 10, 21], price_col="price"):
        """IC 随持有期衰减。"""
        results = []
        for h in horizons:
            fwd_col = f"ret_{h}d"
            rets = []
            for tk, g in self.df.groupby(self.ticker_col):
                g = g.sort_values(self.date_col)
                fwd = g[price_col].shift(-h) / g[price_col] - 1.0
                rets.append(fwd)
            self.df[fwd_col] = pd.concat(rets).reindex(self.df.index)
            g = self.df.dropna(subset=[factor_col, fwd_col])
            if len(g) > 10:
                ic = self._corr(g[factor_col], g[fwd_col])
                if not np.isnan(ic):
                    results.append({"horizon": h, "ic": ic, "n": len(g)})
        return pd.DataFrame(results)

    def evaluate_all(self, factor_cols, forward_col="ret_21d", ic_threshold=0.02, ir_threshold=0.2):
        """评估所有因子，返回通过筛选的因子列表和报告。"""
        records = []
        selected = []
        for col in factor_cols:
            ic_df = self.ic_by_month(col, forward_col)
            if len(ic_df) < 2:
                continue
            mean_ic = ic_df["ic"].mean()
            std_ic = ic_df["ic"].std(ddof=1)
            ir = mean_ic / std_ic if std_ic > 0 else np.nan
            pct_pos = (ic_df["ic"] > 0).mean()
            passed = abs(mean_ic) >= ic_threshold and abs(ir) >= ir_threshold
            records.append({
                "factor": col,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "ir": ir,
                "pct_positive": pct_pos,
                "n_months": len(ic_df),
                "passed": passed,
            })
            if passed:
                selected.append(col)
        report = pd.DataFrame(records).sort_values("ir", ascending=False)
        return report, selected

    def save_report(self, report, out_path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out_path, index=False, encoding="utf-8-sig")
