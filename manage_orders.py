# -*- coding: utf-8 -*-
"""
账户下单 + 止盈止损管理（Paper）。
模式:
  python manage_orders.py --status                 # 只读: 打印持仓+现有挂单
  python manage_orders.py --protect                # dry-run: 打印将挂的止盈/止损 OCO
  python manage_orders.py --protect --execute      # 实际挂止盈止损
  python manage_orders.py --entry plan.csv --execute  # 按计划挂限价加仓单(SAT/主干)
  python manage_orders.py --portfolio --execute    # 组合级 -25% 触发清仓
  python manage_orders.py --tpsl [--execute]       # 每日收盘后止盈/止损检查(软件级,兼容碎股)
安全: 默认 dry-run; 需 --execute 才真正下单。
止盈/止损默认: 止盈 +20% / 止损 -30% / 组合 -25% 清仓 (可用参数覆盖)
"""
import os, sys, json, argparse, datetime
from pathlib import Path
import requests

try:
    from _paths import OUT
except Exception:
    OUT = Path(__file__).resolve().parent / "backtest_output"

def _env(p):
    out = {}
    if Path(p).exists():
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = [x.strip() for x in s.split("=", 1)]; out[k] = v
    return out

def creds():
    ef = _env(r"F:\even-codex\panda\backtest\alpaca.env")
    mel = _env(Path(__file__).resolve().parent / "mail.env")
    KEY = os.environ.get("ALPACA_API_KEY") or ef.get("ALPACA_API_KEY") or ef.get("ALPACA_KEY_ID") or mel.get("ALPACA_API_KEY")
    SEC = os.environ.get("ALPACA_SECRET_KEY") or ef.get("ALPACA_SECRET_KEY") or mel.get("ALPACA_SECRET_KEY")
    EP  = os.environ.get("ALPACA_ENDPOINT") or ef.get("ALPACA_ENDPOINT") or "https://paper-api.alpaca.markets"
    if not (KEY and SEC):
        print("ERROR: 缺少 Alpaca 凭据"); sys.exit(1)
    return {"hdr": {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC}, "ep": EP}

C, EP = creds(), None

def get(path):
    return requests.get(f"{C['ep']}{path}", headers=C["hdr"], timeout=20).json()
def post(path, body):
    return requests.post(f"{C['ep']}{path}", headers=C["hdr"], data=json.dumps(body), timeout=20)
def _cancel_all(sym=None):
    orders = get("/v2/orders?status=open&limit=200")
    for o in orders:
        if sym and o.get("symbol") != sym: continue
        try: requests.delete(f"{C['ep']}/v2/orders/{o['id']}", headers=C["hdr"], timeout=15)
        except Exception as e: print("  cancel err", e)

def place_oco_sell(sym, qty, tp, sl, dry):
    body = {"symbol": sym, "qty": str(round(qty, 4)), "side": "sell",
            "type": "limit", "limit_price": str(round(tp,2)),
            "order_class": "oco", "time_in_force": "gtc",
            "take_profit": {"limit_price": str(round(tp,2))},
            "stop_loss": {"stop_price": str(round(sl,2))}}
    print(f"  OCO SELL {sym:<6} qty={qty:8.3f}  止盈@{tp:9.2f}  止损@{sl:9.2f}")
    if not dry:
        r = post("/v2/orders", body)
        print(f"      -> {r.status_code} {r.text[:120]}")

def satellite_set():
    f = OUT / "satellite_targets.json"
    if f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8")); return set(d.get("tickers", []))
        except Exception: return set()
    return set()

def core_set():
    f = next(OUT.glob("current_holdings_*.csv"), None)
    if not f: return set()
    import pandas as pd
    return set(pd.read_csv(f)["ticker"].astype(str))

def protect(dry, tp_pct, sl_pct, watch_sym=None):
    print(f"止盈/止损: 止盈+{tp_pct*100:.0f}%  止损-{sl_pct*100:.0f}%  (dry-run={dry})")
    pos = get("/v2/positions")
    orders = get("/v2/orders?status=open&limit=200")
    have = {o["symbol"] for o in orders if o.get("order_class") == "oco"}
    for p in pos:
        sym = p["symbol"]; qty = float(p["qty"])
        avg = float(p["avg_entry_price"])
        if watch_sym and sym != watch_sym: continue
        if sym in have:
            print(f"  {sym:<6} 已有止盈止损挂单,跳过"); continue
        tp = avg * (1 + tp_pct); sl = avg * (1 - sl_pct)
        place_oco_sell(sym, qty, tp, sl, dry)

def entry(plan_csv, dry, notional_col="notional"):
    import pandas as pd
    plan = pd.read_csv(plan_csv)
    today = datetime.date.today().strftime("%Y-%m-%d")
    for _, r in plan.iterrows():
        sym, limit = r["ticker"], float(r["limit_price"])
        notional = float(r.get(notional_col, 0))
        qty_est = notional / limit if notional else 0
        print(f"  BUY LIMIT {sym:<6} x ~{qty_est:8.3f} @ {limit:9.2f}  (金额${notional:,.0f})")
        if not dry:
            body = {"symbol": sym, "notional": str(round(notional,2)), "side": "buy",
                    "type": "limit", "limit_price": str(round(limit,2)), "time_in_force": "day"}
            r2 = post("/v2/orders", body)
            print(f"      -> {r2.status_code} {r2.text[:120]}")

def portfolio(dry, liq_pct):
    start = 20000.0
    pos = get("/v2/positions")
    nav = sum(float(p["market_value"]) for p in pos if p["symbol"] in (core_set() | satellite_set()))
    lvl = start * (1 - liq_pct)
    print(f"组合NAV=${nav:,.2f}  (-{(1-nav/start)*100:.1f}%)  清仓线=${lvl:,.0f} (-{liq_pct*100:.0f}%)")
    if nav <= lvl:
        print("!! 达清仓线,触发全清仓")
        if not dry:
            for p in pos:
                if p["symbol"] in (core_set() | satellite_set()):
                    body={"symbol":p["symbol"],"qty":str(round(float(p["qty"]),4)),"side":"sell","type":"market","time_in_force":"day"}
                    r2=post("/v2/orders",body); print(f"  SELL {p['symbol']} -> {r2.status_code}")
    else:
        print(f"  未触及清仓线,继续持有。")

def tpsl(dry, tp_pct, sl_pct, warn_pct=0.20, liq_pct=0.25):
    """软件级止盈止损：每日检查现价，触发即市价卖出（兼容碎股，碎股不支持OCO/GTC）。
    参数填 0/正数都按 0 处理->关闭该项。默认 dry-run；--execute 才真实卖出。"""
    import pandas as pd
    today = datetime.date.today().isoformat()
    tp_on = tp_pct is not None and tp_pct > 0
    sl_on = sl_pct is not None and sl_pct > 0
    print(f"止盈/止损监控: 止盈+{tp_pct*100 if tp_on else 0:.0f}%  止损-{sl_pct*100 if sl_on else 0:.0f}%  (dry-run={dry})  日期={today}")
    pos = get("/v2/positions")
    rows, triggered = [], 0
    for p in pos:
        sym = p["symbol"]
        try:
            qty = float(p["qty"]); avg = float(p["avg_entry_price"]); cur = float(p["current_price"])
        except Exception:
            continue
        if qty <= 0 or avg <= 0 or cur <= 0:
            print(f"  {sym:<6} 跳过(无有效价格)"); continue
        tp = avg * (1 + tp_pct) if tp_on else float("inf")
        sl = avg * (1 - sl_pct) if sl_on else 0.0
        action = "SELL_STOP" if sl_on and cur <= sl else ("SELL_TP" if tp_on and cur >= tp else None)
        print(f"  {sym:<6} 均价={avg:9.2f} 现价={cur:9.2f} 止盈={tp:9.2f} 止损={sl:9.2f}  {action or '持有'}")
        if action and not dry:
            body = {"symbol": sym, "qty": str(round(qty, 4)), "side": "sell",
                    "type": "market", "time_in_force": "day"}
            r = post("/v2/orders", body)
            print(f"      -> 触发{action} {sym}  qty={qty:.3f}  {r.status_code} {r.text[:160]}")
            if r.status_code in (200, 201, 202):
                rows.append({"date": today, "symbol": sym, "action": action,
                             "avg": round(avg,2), "exit_price": round(cur,2), "qty": round(qty,4)})
                triggered += 1
    # ---- 组合级 NAV 预警 / 清仓（按关键解读：组合级管控回撤，不由单只止损承担）----
    start = float(os.environ.get("PAPER_TARGET_EQUITY", "20000"))
    pos2 = get("/v2/positions")
    nav = sum(float(x["market_value"]) for x in pos2)
    dd = nav / start - 1.0
    warn_lvl = start * (1 - warn_pct if warn_pct and warn_pct > 0 else 0)
    liq_lvl  = start * (1 - liq_pct  if liq_pct  and liq_pct  > 0 else 0)
    print(f"组合NAV=${nav:,.2f}  相对起始 -{dd*-1*100:.1f}%  预警线=${warn_lvl:,.0f}(-{warn_pct*100:.0f}%)  清仓线=${liq_lvl:,.0f}(-{liq_pct*100:.0f}%)")
    if nav <= warn_lvl:
        print(f"  ⚠ 已达组合预警线 ({dd*100:.1f}%)，建议降仓/暂停追高；")
    if nav <= liq_lvl:
        print(f"  !! 已达组合清仓线 ({dd*100:.1f}%)，触发全清仓")
        if not dry:
            sold = 0
            for x in pos2:
                sym = x["symbol"]; qty = float(x["qty"])
                if qty <= 0: continue
                body = {"symbol": sym, "qty": str(round(qty, 4)), "side": "sell",
                        "type": "market", "time_in_force": "day"}
                r = post("/v2/orders", body)
                print(f"      -> 清仓 {sym} qty={qty:.3f}  {r.status_code} {r.text[:120]}")
                sold += 1
            print(f"  已提交清仓 {sold} 笔")
        else:
            print("  [dry-run] 加 --execute 才真正清仓")
    else:
        print(f"  组合未触清仓线，继续执行个股止盈/止损。")

    if not dry:
        if rows:
            log = OUT / "tpsl_log.csv"
            new = pd.DataFrame(rows)
            if log.exists():
                old = pd.read_csv(log)
                new = pd.concat([old, new], ignore_index=True)
            new.to_csv(log, index=False, encoding="utf-8-sig")
            print(f"  今日个股触发 {triggered} 笔, 已追加日志: {log}")
        else:
            print("  今日个股无触发，保持不变。")
    else:
        print("  以上为预计动作；加 --execute 才会真实执行。")

def status():
    print("== 持仓 ==")
    for p in get("/v2/positions"):
        print(f"  {p['symbol']:<6} qty={float(p['qty']):8.3f} 均价={float(p['avg_entry_price']):>9.2f} 现价={float(p['current_price']):>9.2f} 市值={float(p['market_value']):>10.2f} 浮盈%={float(p.get('unrealized_plpc',0))*100:+.2f}")
    print("== 现有挂单 ==")
    for o in get("/v2/orders?status=open&limit=200"):
        print(f"  {o['symbol']:<6} {o['side']:>4} {o.get('order_class','simple'):<9} simplex= {o.get('take_profit','')} stop={o.get('stop_loss','')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--protect", action="store_true")
    ap.add_argument("--entry", default=None)
    ap.add_argument("--portfolio", action="store_true")
    ap.add_argument("--tpsl", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--tp", type=float, default=0.0)   # 关键解读：止盈截断动量收益，默认关闭
    ap.add_argument("--sl", type=float, default=0.30)  # 止损保留做极端保护
    ap.add_argument("--warn", type=float, default=0.20)
    ap.add_argument("--liq", type=float, default=0.25)
    ap.add_argument("--watch", default=None)
    a = ap.parse_args()
    dry = not a.execute
    if a.status: status()
    elif a.protect: protect(dry, a.tp, a.sl, a.watch)
    elif a.entry: entry(a.entry, dry)
    elif a.portfolio: portfolio(dry, a.liq)
    elif a.tpsl: tpsl(dry, a.tp, a.sl, a.warn, a.liq)
    else: ap.print_help()

if __name__ == "__main__":
    main()
