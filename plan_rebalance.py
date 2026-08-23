# -*- coding: utf-8 -*-
"""
柔性分批入场规划器（主干策略执行层）：
- 目标清单仍按月频 top10（保留回测基准）
- 但不一次性市价买，改为分批限价单，5 个交易日内完成
- 触发规则:
    * 回踩加速: 若价格从参考价回落 >=5%，提前买入下一批（买跌不追涨）
    * 冲高降仓: 若两天内涨 >8%，剩余批次降为 50% 或转现金（不追高）
    * 留现金:   月底前没建完的份额留现金（热度保护垫）
用法:
  python plan_rebalance.py --csv backtest_output/current_holdings_6m_skip1_top10.csv --tranches 4
输出:
  backtest_output/rebalance_plan_YYYYMMDD.csv
"""
import os, argparse
from datetime import date, timedelta
from pathlib import Path
import pandas as pd, numpy as np

try:
    from _paths import OUT
except Exception:
    OUT = Path(__file__).resolve().parent / "backtest_output"
    OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data")

def load_prices():
    for cand in [Path(os.environ.get("STOCK_DATA_DIR") or "")/"prices.csv",
                 DATA/"prices.csv",
                 Path(__file__).resolve().parent/"data"/"prices.csv"]:
        if cand.exists():
            return pd.read_csv(cand, index_col=0, parse_dates=True).apply(pd.to_numeric, errors="coerce").sort_index()
    raise FileNotFoundError("找不到 prices.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--tranches", type=int, default=4)          # 分几批
    ap.add_argument("--buffer", type=float, default=0.015)      # 限价缓冲(1.5%)
    ap.add_argument("--pullback-trigger", type=float, default=0.05) # 回落>=5%提前买下一批
    ap.add_argument("--spike-defer", type=float, default=0.08)     # 2天涨>=8%降低批
    ap.add_argument("--high-win", type=int, default=60)
    ap.add_argument("--chase-th", type=float, default=0.97)
    ap.add_argument("--budget", type=float, default=20000.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    holdings = Path(a.csv) if a.csv else next(OUT.glob("current_holdings_*.csv"))
    df = pd.read_csv(holdings)
    if "ticker" not in df.columns:
        raise SystemExit("CSV 缺 ticker 列")
    tickers = df["ticker"].astype(str).tolist()
    per_name = round(a.budget / len(tickers), 2)
    per_batch = round(per_name / a.tranches, 2)

    p = load_prices()
    close = p.iloc[-1]
    # 近2日涨幅（判断冲高）
    ret2 = {}
    if len(p) >= 3:
        for t in tickers:
            try:
                ret2[t] = float(p[t].iloc[-1]/p[t].iloc[-3]-1)
            except Exception:
                ret2[t] = 0.0
    today = date.today()
    print("="*84)
    print(f"柔性分批入场计划   生成:{today}   目标:{len(tickers)}只 x ${per_name:,.2f}")
    print(f"分 {a.tranches} 批 x ${per_batch:,.2f}/批 | 5个交易日完成 | 触发:回踩>={a.pullback_trigger*100:.0f}%提前 / 2日涨>={a.spike_defer*100:.0f}%降批")
    print("="*84)
    plan = []
    for t in tickers:
        px = float(close[t]) if t in close.index and pd.notna(close[t]) else np.nan
        if pd.isna(px):
            print(f"{t:<6}{'--':>9}  数据缺失,跳过"); continue
        hi = float(p[t].tail(a.high_win).max()) if t in p.columns else px
        dist = px/hi
        r2 = ret2.get(t, 0.0)
        chase = dist >= a.chase_th
        spike = r2 >= a.spike_defer
        note = ("冲高!降批" if spike else "") + (" +接近高点等回踩" if chase else "")
        for tr in range(1, a.tranches+1):
            notional = per_batch
            limit = round(px*(1-a.buffer*tr), 2)
            act = "buy"
            if chase and tr > 1:
                limit = min(limit, round(hi*0.97, 2))
            # 冲高: 第2批起降为50%; 接近高点: 末尾1批转现金
            if spike and tr >= 2:
                notional = round(per_batch*0.5, 2)
            if chase and tr == a.tranches:
                act = "cash"; notional = 0.0
            plan.append((t, tr, (today+timedelta(days=tr)).isoformat(), act, limit, notional))
        print(f"{t:<6}{px:>9.2f}{hi:>9.2f}{dist*100:>6.1f}%  2日{r2*100:>+5.1f}% {note:<20} 末批:{plan[-1][3]}")
    print("-"*84)
    print("执行: 每日挂1批限价单; 未成交撤单次日重挂; 回落>=5%可提前买下一批; 冲高/接近高点的批次自动降仓或留现金。")
    out_csv = Path(a.out) if a.out else OUT / f"rebalance_plan_{today:%Y%m%d}.csv"
    pd.DataFrame(plan, columns=["ticker","batch","date","action","limit_price","notional"]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n计划已保存: {out_csv}")

if __name__ == "__main__":
    main()
