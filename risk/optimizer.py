# -*- coding: utf-8 -*-
"""CVXPY 组合优化器：均值-方差 + 行业/个股约束。"""
import numpy as np
import pandas as pd
import cvxpy as cp
from .barra_risk import BarraRisk


class PortfolioOptimizer:
    def __init__(self, risk_aversion=1.0, max_single=0.10, max_sector=0.25,
                 min_count=8, max_count=15, target_vol=None):
        self.risk_aversion = risk_aversion
        self.max_single = max_single
        self.max_sector = max_sector
        self.min_count = min_count
        self.max_count = max_count
        self.target_vol = target_vol
        self.barra = BarraRisk()

    def optimize(self, alpha, cov, tickers=None, sectors=None):
        """
        alpha: Series/ticker -> alpha 预测值（已排序，取 top N）
        cov: DataFrame，历史收益协方差
        tickers: 可选，限定股票池
        sectors: 可选，ticker -> sector 映射
        返回：Series/ticker -> 最优权重
        """
        if tickers is None:
            tickers = alpha.index.tolist()
        # 只保留 alpha 和 cov 都有的 ticker
        tickers = [t for t in tickers if t in alpha.index and t in cov.index and t in cov.columns]
        if len(tickers) < self.min_count:
            return pd.Series(0.0, index=tickers)

        # 默认取 alpha 最高的 max_count 只
        top = alpha.reindex(tickers).dropna().sort_values(ascending=False)
        selected = top.head(self.max_count).index.tolist()
        if len(selected) < self.min_count:
            selected = top.head(self.min_count).index.tolist()
        if len(selected) < self.min_count:
            return pd.Series(0.0, index=tickers)

        a = top.reindex(selected).fillna(0).values
        Sigma = cov.reindex(index=selected, columns=selected).fillna(0).values
        # 保证正半定
        Sigma = (Sigma + Sigma.T) / 2
        eigvals = np.linalg.eigvalsh(Sigma)
        if eigvals.min() < 1e-8:
            Sigma += np.eye(len(Sigma)) * (1e-8 - eigvals.min())

        n = len(selected)
        w = cp.Variable(n)
        objective = cp.Maximize(a @ w - self.risk_aversion * cp.quad_form(w, Sigma))

        constraints = [cp.sum(w) == 1, w >= 0, w <= self.max_single]

        # 行业约束
        if sectors is None:
            sectors = self.barra.sector_map
        sector_list = [sectors.get(t, "其他") for t in selected]
        unique_sectors = sorted(set(sector_list))
        for sec in unique_sectors:
            idx = [i for i, s in enumerate(sector_list) if s == sec]
            if idx:
                constraints.append(cp.sum(w[idx]) <= self.max_sector)

        # 目标波动率约束
        if self.target_vol is not None and self.target_vol > 0:
            constraints.append(cp.quad_form(w, Sigma) <= self.target_vol ** 2)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            try:
                prob.solve(solver=cp.SCS, verbose=False)
            except Exception:
                return pd.Series(0.0, index=selected)

        if w.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
            return pd.Series(0.0, index=selected)

        weights = np.array(w.value).flatten()
        # 过滤极小权重
        weights[weights < 1e-6] = 0
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        return pd.Series(weights, index=selected)
