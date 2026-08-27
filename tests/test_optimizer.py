# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from risk.optimizer import PortfolioOptimizer
from risk.barra_risk import BarraRisk

def test_optimizer_constraints():
    np.random.seed(42)
    tickers = ["NVDA", "AMD", "AVGO", "AMAT", "MU", "META", "MSFT", "AAPL", "TSLA", "DELL", "GLW", "STX"]
    alpha = pd.Series(np.random.randn(len(tickers)), index=tickers)
    cov = pd.DataFrame(np.eye(len(tickers)) * 0.01, index=tickers, columns=tickers)
    opt = PortfolioOptimizer(max_single=0.10, max_sector=0.25, min_count=8, max_count=15)
    w = opt.optimize(alpha, cov, tickers=tickers)
    assert abs(w.sum() - 1.0) < 1e-4
    assert w.max() <= 0.10 + 1e-6
    barra = BarraRisk()
    exp = barra.sector_exposure(w)
    assert exp.max() <= 0.25 + 1e-6
    print("test_optimizer_constraints passed")

if __name__ == "__main__":
    test_optimizer_constraints()
