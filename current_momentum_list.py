# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path(r"F:\even-codex\lianghua2\backtest_output")
DATA = Path(r"F:\even-codex\us-stock-data")
stk = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
stk = stk.loc[:, stk.count() >= 2400]
def ml(px): return px.resample("ME").last()
m = ml(stk)
sc_mom = m / m.shift(6) - 1.0
sc_accel = 0.5*sc_mom + 0.5*m.pct_change(1)
sc_drop = sc_mom.dropna(how="all")
# last COMPLETED month-end = second-to-last label (data ends 2026-08-14, mid-month)
last_full_month_end = sc_drop.index[-2]

def order_for(score, label, outname):
    s = score.loc[last_full_month_end].dropna().sort_values(ascending=False)
    top20 = s.index[:20]
    last_date = stk.index[-1]
    px_last = stk.loc[last_date, top20]
    capital = 20000.0; w = capital / 20.0
    order = pd.DataFrame({"symbol": top20, "score": s[top20].values, "price": px_last.values})
    order["target_usd"] = round(w,2)
    order["shares_fractional"] = (w/px_last.values).round(4)
    order["shares_whole"] = np.floor(w/px_last.values).astype(int)
    order = order.sort_values("score", ascending=False).reset_index(drop=True); order.index = order.index + 1
    print("="*80); print(label)
    print("Signal date (rebalance basis):", last_full_month_end.date())
    print("Latest price date:", last_date.date())
    print("Target equal-weight USD per position: $", round(w,2))
    print(order.to_string())
    order.to_csv(OUT/outname, index_label="rank", encoding="utf-8-sig")
    print("\nTotal planned:", round(order["target_usd"].sum(),2))

order_for(sc_mom, "纯动量 top20", "current_buy_list_mom6_top20.csv")
order_for(sc_accel, "动量+近1月加速 top20", "current_buy_list_mom6_accel_top20.csv")
