# -*- coding: utf-8 -*-
"""Incremental daily update for F:\\even-codex\\us-stock-data
For each ticker, fetch only dates AFTER last row in raw/<SYM>.csv,
append (dedup), then rebuild prices.csv + summary.csv.
Resumable / idempotent: 每天跑都安全。
"""
import os, csv, time
from datetime import date, datetime, timedelta
import requests, pandas as pd

BASE  = r"F:\even-codex\us-stock-data"
RAW   = os.path.join(BASE, "raw")
LOG   = os.path.join(BASE, "logs", "daily_update.log")
MASTER= os.path.join(BASE, "master_tickers.csv")
UA    = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
TODAY = date.today().strftime("%Y-%m-%d")

def log(msg):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
    print(msg, flush=True)

def variant(sym):
    return sym.replace(".", "-") if "." in sym else sym.replace("-", ".")

def fetch(sym, frm, to, session):
    url = f"https://api.nasdaq.com/api/quote/{sym}/historical?assetclass=stocks&fromdate={frm}&todate={to}&limit=9999"
    r = session.get(url, headers={"User-Agent": UA}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    j = r.json()
    return ((j.get("data") or {}).get("tradesTable") or {}).get("rows") or []

def last_date_in(sym):
    p = os.path.join(RAW, sym + ".csv")
    if not os.path.exists(p): return None
    try:
        df = pd.read_csv(p, usecols=["Date"])
    except Exception:
        return None
    if df.empty: return None
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    d = df["Date"].dropna().max()
    return None if pd.isna(d) else d.date()

def append_rows(sym, rows):
    if not rows: return 0
    p = os.path.join(RAW, sym + ".csv")
    existing = set()
    if os.path.exists(p):
        try:
            d = pd.read_csv(p, usecols=["Date"])
            existing = set(d["Date"].astype(str))
        except Exception:
            existing = set()
    added = 0
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in rows:
            d_str = row["date"]
            if d_str in existing: continue
            w.writerow([d_str, row["open"], row["high"], row["low"], row["close"], row["volume"]])
            added += 1
    return added

def main():
    with open(MASTER, encoding="utf-8") as f:
        tickers = [r["symbol"].strip() for r in csv.DictReader(f) if r.get("symbol","").strip()]
    log(f"start: {len(tickers)} tickers, target window to {TODAY}")
    s = requests.Session()
    n_new=n_done=n_skip=0
    for idx, sym in enumerate(tickers, 1):
        ld = last_date_in(sym)
        frm = (ld + timedelta(days=1)).strftime("%Y-%m-%d") if ld else "2016-08-01"
        to  = TODAY
        if frm > to:
            n_skip += 1; continue
        got = 0
        for s_sym in {sym, variant(sym)}:
            for attempt in range(3):
                try:
                    rows = fetch(s_sym, frm, to, s)
                    if rows:
                        got = append_rows(sym, rows); break
                    else:
                        break
                except Exception as e:
                    if attempt == 2: log(f"  FAIL {sym}: {e}")
                    time.sleep(2 + attempt*2)
            if got: break
        if got: n_new += 1
        else:   n_done += 1
        if idx % 50 == 0 or got:
            log(f"[{idx}/{len(tickers)}] {sym} frm={frm} added={got}")
        time.sleep(0.3)
    log(f"summary: updated={n_new} no_new={n_done} skipped={n_skip}")
    log("rebuilding prices.csv / summary.csv ...")
    os.system(f'python "{os.path.join(BASE, "scripts", "build_dataset.py")}"')
    log("done.")

if __name__ == "__main__":
    main()
