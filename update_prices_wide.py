# -*- coding: utf-8 -*-
"""云端用：直接更新宽表 prices.csv。并发拉取以适配 GitHub Actions 时间预算。"""
import os, time
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, pandas as pd

DATA_DIR = Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data")
PRICES   = DATA_DIR / "prices.csv"
MASTER   = DATA_DIR / "master_tickers.csv"
LOG      = DATA_DIR / "update_prices_wide.log"
UA       = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
TODAY    = date.today().strftime("%Y-%m-%d")
WORKERS  = int(os.environ.get("WORKERS", "12"))

def log(msg):
    s = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")

def fetch(sym, frm, to, session):
    try:
        url = f"https://api.nasdaq.com/api/quote/{sym}/historical?assetclass=stocks&fromdate={frm}&todate={to}&limit=9999"
        r = session.get(url, headers={"User-Agent": UA}, timeout=20)
        if r.status_code != 200: return sym, []
        return sym, ((r.json().get("data") or {}).get("tradesTable") or {}).get("rows") or []
    except Exception as e:
        return sym, ("err", str(e))

def to_float(x):
    if x is None: return None
    s = str(x).replace("$","").replace(",","").strip()
    try: return float(s)
    except: return None

def main():
    if not PRICES.exists():
        log(f"ERROR: 找不到 {PRICES}"); return
    prices = pd.read_csv(PRICES, index_col=0, parse_dates=True).sort_index()
    log(f"existing: {prices.shape[1]} cols x {prices.shape[0]} rows, last={prices.index.max().date()}")
    tickers = pd.read_csv(MASTER)["symbol"].astype(str).tolist()
    frm = (prices.index.max() + timedelta(days=1)).strftime("%Y-%m-%d")
    if frm > TODAY:
        log("no new dates needed."); return
    log(f"fetching {frm} -> {TODAY} (workers={WORKERS})")
    s = requests.Session()
    new_data = {}
    done = 0; errs = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, sym, frm, TODAY, s): sym for sym in tickers}
        for fu in as_completed(futs):
            sym, rows = fu.result()
            done += 1
            if isinstance(rows, tuple) and rows and rows[0] == "err":
                errs += 1
            else:
                for r in rows:
                    d = pd.to_datetime(r["date"], format="%m/%d/%Y")
                    new_data.setdefault(d, {})[sym] = to_float(r.get("close"))
            if done % 50 == 0:
                log(f"  progress {done}/{len(tickers)}  new_dates={len(new_data)}  errs={errs}")
    if not new_data:
        log("no new data returned by NASDAQ."); return
    new_df = pd.DataFrame(new_data).T
    new_df = new_df.reindex(columns=prices.columns)
    combined = pd.concat([prices, new_df]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.to_csv(PRICES)
    log(f"updated: added {len(new_data)} new dates; new last={combined.index.max().date()}")

if __name__ == "__main__":
    main()
