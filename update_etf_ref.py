# -*- coding: utf-8 -*-
"""Incremental update for F:\\even-codex\\panda\\backtest\\prices_2016.csv
Appends new rows for SPY/QQQ/IWM/TLT/SPMO/QUAL/USMV/VLUE from NASDAQ.
"""
import os, sys, time
from datetime import date, datetime, timedelta
import requests, pandas as pd

PATH  = os.environ.get("ETFS_REF_FILE") or r"F:\even-codex\panda\backtest\prices_2016.csv"
LOG   = os.path.join(os.path.dirname(PATH), "etf_ref_update.log")
ETFS  = ["SPY","QQQ","IWM","TLT","SPMO","QUAL","USMV","VLUE"]
UA    = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
TODAY = date.today().strftime("%Y-%m-%d")

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")

def fetch(sym, frm, to, s):
    url = f"https://api.nasdaq.com/api/quote/{sym}/historical?assetclass=etf&fromdate={frm}&todate={to}&limit=9999"
    r = s.get(url, headers={"User-Agent": UA}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    j = r.json()
    return ((j.get("data") or {}).get("tradesTable") or {}).get("rows") or []

def to_float(x):
    if x is None: return None
    s = str(x).replace("$","").replace(",","").strip()
    try: return float(s)
    except: return None

def main():
    s = requests.Session()
    # 读已有宽表
    df = pd.read_csv(PATH, parse_dates=["date"]).set_index("date").sort_index()
    log(f"start: existing {df.shape[1]} cols, {df.shape[0]} rows, ends {df.index[-1].date()}")

    # 1) 先扫一遍 8 个 ETF 各自最后一天
    last_per = {t: df[t].dropna().index.max() if df[t].notna().any() else None for t in ETFS}
    # 2) 全局最后一天（决定起算点）
    frm_global = max([d for d in last_per.values() if d is not None]) + timedelta(days=1)
    frm_str = frm_global.strftime("%Y-%m-%d")
    if frm_str > TODAY:
        log("no new dates to fetch."); return
    log(f"fetch window: {frm_str} -> {TODAY}")

    # 3) 准备日期索引(union of all returned dates)
    new_rows = {}
    for t in ETFS:
        if last_per[t] is not None and (last_per[t] + timedelta(days=1)) > pd.Timestamp(TODAY):
            continue
        try:
            rows = fetch(t, frm_str, TODAY, s)
        except Exception as e:
            log(f"  FAIL {t}: {e}"); continue
        for row in rows:
            d = pd.to_datetime(row["date"], format="%m/%d/%Y")
            c = to_float(row.get("close"))
            if d not in new_rows: new_rows[d] = {}
            new_rows[d][t] = c
        time.sleep(0.3)
    if not new_rows:
        log("no new rows returned."); return

    # 4) 把新行写回宽表
    for d, kv in new_rows.items():
        if d not in df.index:
            df.loc[d] = [None] * df.shape[1]
        for t, v in kv.items():
            df.at[d, t] = v
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(PATH, encoding="utf-8")
    log(f"added {len(new_rows)} new dates; new tail:")
    print(df.tail(len(new_rows)).round(4).to_string())

if __name__ == "__main__":
    main()
