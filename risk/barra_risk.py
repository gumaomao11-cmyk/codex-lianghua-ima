# -*- coding: utf-8 -*-
"""Barra 风格/行业风险暴露计算与约束检查。"""
import numpy as np
import pandas as pd
from .industry_map import get_industry_map


class BarraRisk:
    def __init__(self, industry_map=None):
        self.industry_map = industry_map if industry_map is not None else get_industry_map()
        self.sector_map = dict(zip(self.industry_map["ticker"], self.industry_map["sector"]))

    def sector_exposure(self, weights):
        """
        weights: Series/ticker -> weight
        返回 sector -> exposure
        """
        df = pd.DataFrame({"ticker": weights.index, "weight": weights.values})
        df["sector"] = df["ticker"].map(self.sector_map).fillna("其他")
        return df.groupby("sector")["weight"].sum()

    def check_constraints(self, weights, max_single=0.10, max_sector=0.25):
        """检查是否满足个股/行业约束，返回 bool 和详细信息。"""
        ok = True
        msgs = []
        max_w = weights.max()
        if max_w > max_single + 1e-6:
            ok = False
            msgs.append(f"single stock max {max_w:.2%} > {max_single:.0%}")
        sector_exp = self.sector_exposure(weights)
        max_s = sector_exp.max()
        if max_s > max_sector + 1e-6:
            ok = False
            msgs.append(f"sector max {max_s:.2%} > {max_sector:.0%}")
        return ok, msgs, sector_exp

    def cov_matrix(self, returns, tickers):
        """计算历史收益协方差矩阵（用 EWMA 或样本协方差）。"""
        R = returns[tickers].dropna()
        return R.cov().fillna(0)
