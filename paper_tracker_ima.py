# -*- coding: utf-8 -*-
"""每日 paper 日报 —— IMA 策略（动量+IMA词频），独立于主策略，单独 $20k 跟踪。"""
import os, sys, json
from pathlib import Path
from datetime import date
import requests, pandas as pd, numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

from _paths import OUT
TARGET  = OUT / "ima_final_top10.csv"
STATE   = OUT / "paper_state_ima.json"
LOG     = OUT / "paper_log_ima.csv"
TARGET_EQ = float(os.environ.get("PAPER_TARGET_EQUITY", "20000"))
SPY_FILE = Path(os.environ.get("ETFS_REF_FILE") or r"F:\even-codex\panda\backtest\prices_2016.csv")

def _from_env_file(p):
    out = {}
    if not Path(p).exists(): return out
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s: continue
        k, v = [x.strip() for x in s.split("=", 1)]
        out[k] = v
    return out

KEY = (os.environ.get("ALPACA_KEY_ID") or os.environ.get("ALPACA_API_KEY") or "").strip()
SEC = (os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_SECRET") or "").strip()
EP  = (os.environ.get("ALPACA_ENDPOINT") or "https://paper-api.alpaca.markets").strip()
if not (KEY and SEC):
    ef = _from_env_file(r"F:\even-codex\panda\backtest\alpaca.env")
    KEY = KEY or ef.get("ALPACA_API_KEY", "")
    SEC = SEC or ef.get("ALPACA_SECRET_KEY", "")
if not (KEY and SEC):
    print("ERROR: 未找到 Alpaca 凭据。"); sys.exit(1)
hdr = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC}

_acc_r = requests.get(f"{EP}/v2/account", headers=hdr, timeout=15)
try: acc = _acc_r.json()
except Exception: acc = {"raw": _acc_r.text[:300]}
if not (isinstance(acc, dict) and "equity" in acc and "cash" in acc):
    print(f"ERROR: Alpaca 账户接口异常 (HTTP {_acc_r.status_code}): {str(acc)[:300]}"); sys.exit(1)
_pos_r = requests.get(f"{EP}/v2/positions", headers=hdr, timeout=15)
try: pos = _pos_r.json()
except Exception: pos = {"raw": _pos_r.text[:300]}
if not isinstance(pos, list):
    print(f"ERROR: Alpaca 持仓接口异常 (HTTP {_pos_r.status_code}): {str(pos)[:300]}"); sys.exit(1)
total_equity = float(acc["equity"]); cash = float(acc["cash"])

if not TARGET.exists():
    print("ERROR: 未找到 ima_final_top10.csv，无法跟踪 IMA 策略目标名单。"); sys.exit(1)
target = pd.read_csv(TARGET)
tickers = sorted(target["ticker"].astype(str).tolist())
target_set = set(tickers)
# ---- IMA 仓位：优先按已成交的 IMA 订单份额跟踪（避免与主策略重叠股重复计值） ----
ima_orders = []
orders_file = OUT / "ima_orders.json"
if orders_file.exists():
    try: ima_orders = json.loads(orders_file.read_text(encoding="utf-8"))
    except Exception: ima_orders = []
order_qty = {x["symbol"]: float(x["qty"]) for x in ima_orders if x.get("qty")}
price_map = {pp["symbol"]: float(pp.get("current_price") or pp.get("lastday_price") or 0) for pp in pos}
avail = {}
for pp in pos:
    avail[pp["symbol"]] = pp
limit_pt = {pp["symbol"]: float(pp.get("unrealized_plpc", 0)) * 100 for pp in pos}
ima_share_lines = []
use_orders = bool(order_qty)
if use_orders:
    nav = 0.0
    for sym, qty in order_qty.items():
        qty = float(qty); px = price_map.get(sym, 0.0)
        mv = qty * px
        avg = next((x["avg_price"] for x in ima_orders if x["symbol"]==sym), 0.0) or 0.0
        cost = qty * avg
        upl = (px - avg) / avg * 100 if avg else 0.0
        ima_share_lines.append({"symbol": sym, "qty": qty, "price": px, "cost": cost, "mkt": mv, "upl": upl})
        nav += mv
    # 用 IMA 目标名单补齐尚未有成交的（兜底）——正常应全部有
    for t in target_set:
        if t not in order_qty:
            ppos = avail.get(t)
            mv = float(ppos["market_value"]) if ppos else 0.0
            ima_share_lines.append({"symbol": t, "qty": 0.0, "price": price_map.get(t,0), "cost": 0.0, "mkt": mv, "upl": float(ppos.get("unrealized_plpc",0))*100 if ppos else 0.0})
            nav += mv
else:
    ima_pos = [p for p in pos if p["symbol"] in target_set]
    nav = sum(float(p["market_value"]) for p in ima_pos)
    for p in ima_pos:
        ima_share_lines.append({"symbol": p["symbol"], "qty": float(p["qty"]), "price": float(p.get("current_price") or 0),
                                "cost": float(p["cost_basis"] or 0), "mkt": float(p["market_value"]),
                                "upl": float(p.get("unrealized_plpc",0))*100})

# ---- 参照指数（缺失就跳过对比） ----
spy = qqq = None
if SPY_FILE.exists():
    try:
        ref = pd.read_csv(SPY_FILE, parse_dates=["date"]).set_index("date").sort_index()
        if "SPY" in ref.columns: spy = float(ref["SPY"].iloc[-1])
        if "QQQ" in ref.columns: qqq = float(ref["QQQ"].iloc[-1])
    except Exception:
        pass

# ---- state ----
with_state = {}
if STATE.exists():
    try: with_state = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception: with_state = {}
if with_state.get("start_equity", 0) > TARGET_EQ * 1.2:
    with_state["start_equity"] = TARGET_EQ
if not with_state.get("start_equity"):
    with_state.update({
        "start_date": str(date.today()),
        "start_equity": TARGET_EQ,
        "start_spy": spy,
        "start_qqq": qqq,
        "target_tickers": tickers,
    })
    STATE.write_text(json.dumps(with_state, indent=2), encoding="utf-8")

ret_total = nav / float(with_state["start_equity"]) - 1
ret_spy = spy / float(with_state["start_spy"]) - 1 if (spy and with_state.get("start_spy")) else None
ret_qqq = qqq / float(with_state["start_qqq"]) - 1 if (qqq and with_state.get("start_qqq")) else None
alpha_spy = ret_total - ret_spy if ret_spy is not None else None
alpha_qqq = ret_total - ret_qqq if ret_qqq is not None else None

# ---- 追加日志 ----
log_row = {
    "date": str(date.today()),
    "nav": round(nav, 2),
    "total_equity": round(total_equity, 2),
    "cash": round(cash, 2),
    "n_pos": len(ima_share_lines),
    "spy": round(spy, 4) if spy else "",
    "qqq": round(qqq, 4) if qqq else "",
    "ret_total": round(ret_total * 100, 4),
    "ret_spy": round(ret_spy * 100, 4) if ret_spy is not None else "",
    "ret_qqq": round(ret_qqq * 100, 4) if ret_qqq is not None else "",
    "alpha_spy": round(alpha_spy * 100, 4) if alpha_spy is not None else "",
    "alpha_qqq": round(alpha_qqq * 100, 4) if alpha_qqq is not None else "",
}
if LOG.exists():
    log = pd.read_csv(LOG, parse_dates=["date"])
    missing = [c for c in log_row if c not in log.columns]
    for c in missing: log[c] = np.nan
else:
    log = pd.DataFrame(columns=list(log_row.keys()))
log = log[log["date"] != str(pd.Timestamp(date.today()))]
log = pd.concat([log, pd.DataFrame([log_row])], ignore_index=True)
log.to_csv(LOG, index=False, encoding="utf-8-sig")

# ---- 指标 ----
prev_nav = float(log.iloc[-2]["nav"]) if len(log) >= 2 else None
day_ret = (nav / prev_nav - 1) if prev_nav else None
daily = log["nav"].diff().dropna() / log["nav"].shift(1).dropna()
daily = daily.replace([np.inf, -np.inf], np.nan).dropna()

def sharpe_n(n):
    s = daily.tail(n)
    if len(s) < 2 or s.std() == 0: return None
    return float(s.mean() / s.std() * np.sqrt(252))
def ret_n(n):
    s = daily.tail(n)
    return float((1 + s).prod() - 1) if len(s) else None
def vol_n(n):
    s = daily.tail(n)
    return float(s.std() * np.sqrt(252)) if len(s) >= 2 else None

hwm = log["nav"].cummax()
dd = (log["nav"] / hwm - 1)
max_dd = float(dd.min()); cur_dd = float(dd.iloc[-1])

target_w = 100.0 / len(tickers) if tickers else 0
holdings = []
for rec in ima_share_lines:
    mv = float(rec["mkt"])
    w = mv / nav if nav > 0 else 0
    holdings.append((rec["symbol"], mv, w * 100, (w - target_w/100.0) * 100, float(rec["upl"])))
holdings.sort(key=lambda x: x[2], reverse=True)
max_drift = max((abs(h[3]) for h in holdings), default=0)

days_held = (pd.Timestamp(date.today()) - pd.Timestamp(with_state["start_date"])).days + 1

print("=" * 64)
print(f"【IMA 策略日报】{date.today()}    (运行 {days_held} 天)")
print("=" * 64)
print("【策略概况】")
print(f"  策略预算 / 起始权益 : ${with_state['start_equity']:>12,.2f}  (动量 + IMA 词频, 独立 $20k)")
print(f"  当前策略部分 NAV    : ${nav:>12,.2f}    ({len(ima_share_lines)} 只持仓)")
print(f"  当日盈亏            : {f'${day_ret:+.2%}' if day_ret is not None else '(无前一日数据)'}")
pnl = nav - float(with_state["start_equity"])
print(f"  累计盈亏            : ${pnl:>+12,.2f}    ({ret_total*100:+.2f}%)")
print(f"  距最高点回撤        : {cur_dd*100:>6.2f}%     历史最大回撤 {max_dd*100:>6.2f}%")
print()
print("【账户总览(非 IMA 部分)】")
print(f"  账户总权益          : ${total_equity:>12,.2f}")
print(f"  账户现金            : ${cash:>12,.2f}")
print(f"  IMA 已投入          : ${nav:>12,.2f}    占比 {nav/total_equity*100 if total_equity else 0:>5.1f}%")
print()
print(f"【收益对比(自 {with_state['start_date']} 至今)】")
print(f"  IMA 策略            : {ret_total*100:+7.2f}%")
if ret_spy is not None:
    print(f"  SPY                 : {ret_spy*100:+7.2f}%   alpha vs SPY  {alpha_spy*100:+6.2f}%")
if ret_qqq is not None:
    print(f"  QQQ                 : {ret_qqq*100:+7.2f}%   alpha vs QQQ  {alpha_qqq*100:+6.2f}%")
print()
print("【滚动指标】")
def _fmt(x, kind="pct"):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "  (数据不足)"
    if kind == "pct": return f"{x*100:+6.2f}%"
    if kind == "sharpe": return f"{x:5.2f}"
    if kind == "vol": return f"{x*100:5.1f}%"
    return str(x)
print(f"  1日  收益/夏普      : {_fmt(ret_n(1))} / {_fmt(sharpe_n(1),'sharpe')}")
print(f"  5日  收益/夏普/波动 : {_fmt(ret_n(5))} / {_fmt(sharpe_n(5),'sharpe')} / {_fmt(vol_n(5),'vol')}")
print(f"  21日 收益/夏普/波动 : {_fmt(ret_n(21))} / {_fmt(sharpe_n(21),'sharpe')} / {_fmt(vol_n(21),'vol')}")
print(f"  全部 收益/夏普/波动 : {_fmt(ret_total)} / {_fmt(sharpe_n(len(daily)),'sharpe')} / {_fmt(vol_n(len(daily)),'vol')}")
print()
if holdings:
    print(f"【持仓明细(等权目标 {target_w:.1f}%)】")
    print(f"  {'代码':<6} {'股数':>10} {'市值':>10} {'权重':>6} {'偏离':>7} {'浮盈%':>8}")
    qty_map = {r["symbol"]: r["qty"] for r in ima_share_lines}
    for sym, mv, w, drift, upl in holdings:
        flag = " ⚠" if abs(drift) > 1.0 else ""
        print(f"  {sym:<6} {qty_map.get(sym,0):>10.3f} ${mv:>9,.2f} {w:>5.2f}% {drift:>+6.2f}pp {upl:>+7.2f}%{flag}")
    print(f"  持仓最大偏离等权: {max_drift:.2f}pp  (>{5.0}pp 建议调仓)")
else:
    print("【持仓明细】  (尚未建仓)")
print()
print(f"日志: {LOG}")
