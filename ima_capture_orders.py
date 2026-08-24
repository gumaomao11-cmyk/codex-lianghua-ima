# -*- coding: utf-8 -*-
"""把 IMA 已成交订单写入 ima_orders.json，供 paper_tracker_ima 精确跟踪 IMA 仓位。"""
import os, sys, json
from pathlib import Path
import requests
import pandas as pd
from _paths import OUT
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGET = OUT / "ima_final_top10.csv"
STORE = OUT / "ima_orders.json"

def _from_env_file(p):
    out={}
    if not Path(p).exists(): return out
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        s=line.strip()
        if not s or s.startswith("#") or "=" not in s: continue
        k,v=[x.strip() for x in s.split("=",1)]; out[k]=v
    return out
KEY=(os.environ.get("ALPACA_API_KEY") or "").strip()
SEC=(os.environ.get("ALPACA_SECRET_KEY") or "").strip()
if not (KEY and SEC):
    ef=_from_env_file(r"F:\even-codex\panda\backtest\alpaca.env"); KEY=KEY or ef.get("ALPACA_API_KEY",""); SEC=SEC or ef.get("ALPACA_SECRET_KEY","")
if not (KEY and SEC):
    print("ERROR: 未找到凭据"); sys.exit(1)
hdr={"APCA-API-KEY-ID":KEY,"APCA-API-SECRET-KEY":SEC}
EP=os.environ.get("ALPACA_ENDPOINT") or "https://paper-api.alpaca.markets"

tickers = sorted(pd.read_csv(TARGET)["ticker"].astype(str).tolist())
r = requests.get(f"{EP}/v2/orders", headers=hdr, timeout=15, params={"status":"filled","symbols":",".join(tickers),"limit":50,"direction":"desc"})
orders = r.json()
if not isinstance(orders, list):
    print("ERROR orders:", str(orders)[:300]); sys.exit(1)

# 每个 symbol 取最新一笔 buy 成交（当前 IMA 建仓）
latest = {}
for o in orders:
    if o.get("symbol") in tickers and o.get("side")=="buy" and o.get("status")=="filled":
        sym = o["symbol"]; ts = o.get("created_at","")
        try: qty = float(o.get("filled_qty") or 0); avg = float(o.get("filled_avg_price") or 0)
        except Exception: continue
        if sym not in latest or ts > latest[sym]["created_at"]:
            latest[sym] = {"symbol":sym,"created_at":ts,"qty":qty,"avg_price":avg}
recs = [latest[s] for s in tickers if s in latest]
STORE.write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
print("saved", STORE)
for x in recs:
    print(f"  {x['symbol']:<6} qty {x['qty']:>10.4f}  avg {x['avg_price']:>10.4f}  {x['created_at'][:19]}")
