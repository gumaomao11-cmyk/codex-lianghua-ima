# -*- coding: utf-8 -*-
"""
卫星仓（机会仓）规划器：单独标记、不进主策略回测。
规则:
  - 主仓仍是月频 top10（见 plan_rebalance.py）
  - 卫星仓是对"近期爆发但未进 top10"标的的试探仓
  - 预算默认 = 主策略预算的 15% (默认 3000/20000)
  - 分 2 批限价单; 接近60日高点的先等回踩
  - 输出: backtest_output/satellite_plan_YYYYMMDD.csv + satellite_targets.json
用法:
  python satellite_plan.py --tickers "MRNA LLY" --budget 3000
  python satellite_plan.py --csv 候选.csv --budget 3000 --tranches 2
"""
import os, argparse, json
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
    for cand in [Path(os.environ.get("STOCK_DATA_DIR") or "")/ "prices.csv",
                 DATA / "prices.csv",
                 Path(__file__).resolve().parent / "data" / "prices.csv"]:
        if cand.exists():
            return pd.read_csv(cand, index_col=0, parse_dates=True).apply(pd.to_numeric, errors="coerce").sort_index()
    raise FileNotFoundError("找不到 prices.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--budget", type=float, default=3000.0)
    ap.add_argument("--tranches", type=int, default=2)
    ap.add_argument("--buffer", type=float, default=0.02)
    ap.add_argument("--high-win", type=int, default=60)
    ap.add_argument("--chase-th", type=float, default=0.96)
    a = ap.parse_args()

    if a.csv:
        cand = pd.read_csv(a.csv)
        tickers = [str(t) for t in cand["ticker"].astype(str).tolist() if str(t) not in ("", "nan")]
    else:
        tickers = [t for t in (a.tickers.replace(" ", ",").split(",")) if t]

    if not tickers:
        raise SystemExit("请通过 --tickers \"AAA BBB\" 或 --csv 提供候选标的")

    p = load_prices()
    close = p.iloc[-1]
    today = date.today()
    per_batch = round(a.budget / a.tranches, 2)

    print("="*78)
    print(f"卫星仓(机会仓)计划   生成:{today}   预算:${a.budget:,.0f}(=主仓15%)   分{a.tranches}批 x ${per_batch:,.2f}/批/标的")
    print("="*78)
    print(f"{'代码':<6}{'现价':>9}{'60日高':>9}{'距高点':>8}  {'提示':<18} {'批次限价':>10}")
    plan = []
    approved = []
    for t in tickers:
        px = float(close[t]) if t in close.index and pd.notna(close[t]) else np.nan
        if pd.isna(px):
            print(f"{t:<6}{'--':>9}  数据缺失，跳过"); continue
        hi = float(p[t].tail(a.high_win).max()) if t in p.columns else px
        dist = px / hi
        if dist >= a.chase_th:
            note = "接近高点!先等回踩"
        else:
            note = "可分批买"
        last_limit = None
        for tr in range(1, a.tranches+1):
            limit = round(px * (1 - a.buffer*tr), 2)
            if dist >= a.chase_th and tr > 1:
                limit = min(limit, round(hi*0.95, 2))
            last_limit = limit
            plan.append(("SAT", t, tr, (today + timedelta(days=tr)).isoformat(), limit, per_batch))
        print(f"{t:<6}{px:>9.2f}{hi:>9.2f}{dist*100:>7.1f}%  {note:<18} {last_limit:>10.2f}")
        approved.append(t)

    out_csv = OUT / f"satellite_plan_{today:%Y%m%d}.csv"
    pd.DataFrame(plan, columns=["tag","ticker","batch","date","limit_price","notional"]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    # 记录卫星目标（供 paper_tracker 单独跟踪）
    st = {"note": "卫星仓(机会仓)-不进主策略回测", "start_date": today.isoformat(),
          "budget": a.budget, "tickers": approved}
    (OUT / "satellite_targets.json").write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    print("-"*78)
    print("说明: 卫星仓≤主仓预算15%; 每批限价单; 接近高点先等回踩; 单独标记、不计入策略回测。")
    print(f"\n计划已保存: {out_csv}")
    print(f"卫星目标已记录: {OUT / 'satellite_targets.json'}")

if __name__ == "__main__":
    main()