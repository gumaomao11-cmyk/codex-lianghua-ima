# -*- coding: utf-8 -*-
"""
Alpaca 下单：IMA 策略（动量 + IMA 词频），独立于主策略的 $20,000 仓位。
  python alpaca_buy_ima.py                # dry-run，按 ima_final_top10.csv 买 10 只各 $2000
  python alpaca_buy_ima.py --execute      # 实际下单（碎股 notional）
"""
import os, sys, json
from pathlib import Path
import requests
import pandas as pd

from _paths import OUT
CSV = str(OUT / "ima_final_top10.csv")
PER = 2000.0
DRY = "--execute" not in sys.argv
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
    _ef = _from_env_file(str(Path(__file__).parent / "alpaca.env"))
    KEY = KEY or _ef.get("ALPACA_API_KEY", "") or _ef.get("ALPACA_KEY_ID", "")
    SEC = SEC or _ef.get("ALPACA_SECRET_KEY", "")
if not (KEY and SEC):
    print("ERROR: 未找到 Alpaca 凭据。检查 alpaca.env 或环境变量。"); sys.exit(1)

hdr = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC, "Content-Type": "application/json"}
acc = requests.get(f"{EP}/v2/account", headers=hdr, timeout=15).json()
print("== Account ==")
print("  status :", acc.get("status"))
print("  equity :", acc.get("equity"))
print("  cash   :", acc.get("cash"))
print("  endpoint:", EP)
print("  mode   : IMA 策略单独买入 | dry-run:", DRY)
print()

if not Path(CSV).exists():
    print(f"ERROR: 找不到 {CSV}，请先生成 ima_final_top10.csv"); sys.exit(1)
df = pd.read_csv(CSV)
tickers = df["ticker"].astype(str).tolist()

print("== Buy plan (IMA $20k, 10 只等权 = 每只 $2,000 notional) ==")
total = 0.0
for t in tickers:
    total += PER
    print(f"  BUY  {t:<6} notional ${PER:,.2f}")
print(f"  TOTAL: ${total:,.2f}")

def submit(body):
    if DRY:
        print(f"  [DRY] {body['side']:<4} {body['symbol']:<6} {body.get('type','market')} notional={body.get('notional')}")
        return None
    r = requests.post(f"{EP}/v2/orders", headers=hdr, data=json.dumps(body), timeout=15)
    print(f"  {body['side']:<4} {body['symbol']:<6} {r.status_code}  {r.text[:140]}")
    return r

for t in tickers:
    submit({"symbol": t, "notional": PER, "side": "buy", "type": "market", "time_in_force": "day"})

if not DRY:
    # 保存已成交份额，供 paper_tracker_ima 精确跟踪（每个 symbol 取最新一笔 buy）
    try:
        r = requests.get(f"{EP}/v2/orders", headers=hdr, timeout=15,
                         params={"status":"filled","symbols":",".join(tickers),"limit":50,"direction":"desc"})
        od = r.json()
        latest = {}
        if isinstance(od, list):
            for o in od:
                if o.get("side")=="buy" and o.get("status")=="filled" and o.get("symbol") in tickers:
                    ts = o.get("created_at","")
                    try: qty=float(o.get("filled_qty") or 0); avg=float(o.get("filled_avg_price") or 0)
                    except Exception: continue
                    if o["symbol"] not in latest or ts > latest[o["symbol"]]["created_at"]:
                        latest[o["symbol"]] = {"symbol":o["symbol"],"created_at":ts,"qty":qty,"avg_price":avg}
        recs = [latest[s] for s in tickers if s in latest]
        (OUT/"ima_orders.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
        print("saved ima_orders.json:", len(recs), "orders")
    except Exception as e:
        print("警告: 保存成交份额失败:", e)
print("Done.")
