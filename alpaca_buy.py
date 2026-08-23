# -*- coding: utf-8 -*-
"""
Alpaca 下单脚本（已支持 rebalance）。
  python alpaca_buy.py              # dry-run，按 CSV 买 10 只各 2000
  python alpaca_buy.py --execute    # 实际下单
  python alpaca_buy.py --rebalance --dry-run   # 模拟盘月末调仓
  python alpaca_buy.py --rebalance --execute   # 实际调仓
"""
import os, sys, json
from pathlib import Path
import requests
import pandas as pd

from _paths import OUT
CSV = str(OUT / "current_holdings_6m_skip1_accel_top10.csv")
PER = 2000.0
DRY = "--execute" not in sys.argv
REBAL = "--rebalance" in sys.argv
DEFAULT_EP = "https://paper-api.alpaca.markets"

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
EP  = (os.environ.get("ALPACA_ENDPOINT") or DEFAULT_EP).strip()

if not (KEY and SEC):
    ef = _from_env_file(r"F:\even-codex\panda\backtest\alpaca.env")
    KEY = KEY or ef.get("ALPACA_API_KEY", "")
    SEC = SEC or ef.get("ALPACA_SECRET_KEY", "")
if not (KEY and SEC):
    ef = _from_env_file(str(Path(__file__).parent / "alpaca.env"))
    KEY = KEY or ef.get("ALPACA_API_KEY", "") or ef.get("ALPACA_KEY_ID", "")
    SEC = SEC or ef.get("ALPACA_SECRET_KEY", "")
if not (KEY and SEC):
    print("ERROR: 未找到凭据。检查 alpaca.env 或环境变量。"); sys.exit(1)

hdr = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC, "Content-Type":"application/json"}
acc = requests.get(f"{EP}/v2/account", headers=hdr, timeout=15).json()
print("== Account =="); print("  status :", acc.get("status"))
print("  equity :", acc.get("equity")); print("  cash   :", acc.get("cash"))
print("  endpoint:", EP); print("  mode   :", "rebalance" if REBAL else "buy", "| dry-run:", DRY); print()

df = pd.read_csv(CSV); tickers = df["ticker"].tolist()

def submit(body):
    if DRY:
        print(f"[DRY] {body}"); return None
    r = requests.post(f"{EP}/v2/orders", headers=hdr, data=json.dumps(body), timeout=15)
    print(f"  {body['side']:<4} {body['symbol']:<6} {r.status_code}  {r.text[:140]}")
    return r

if REBAL:
    equity = float(acc['equity'])
    per_name = round(equity / max(len(tickers), 1), 2)
    pos = requests.get(f"{EP}/v2/positions", headers=hdr, timeout=15).json()
    print(f"== Rebalance plan ==")
    print(f"  current equity : ${equity:,.2f}")
    print(f"  per-name target: ${per_name:,.2f}")
    print(f"  current positions: {[p['symbol'] for p in pos]}")
    print(f"  target list    : {tickers}")
    print()
    # 1) sell non-target
    for p in pos:
        if p['symbol'] not in tickers:
            submit({"symbol": p['symbol'], "qty": p['qty'], "side": "sell", "type": "market", "time_in_force": "day"})
    # 2) buy target
    for t in tickers:
        submit({"symbol": t, "notional": per_name, "side": "buy", "type": "market", "time_in_force": "day"})
else:
    print("== Buy plan (initial) ==")
    for t in tickers:
        submit({"symbol": t, "notional": PER, "side": "buy", "type": "market", "time_in_force": "day"})
print("Done.")
