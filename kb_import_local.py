# -*- coding: utf-8 -*-
"""Local KB exporter importer: read md/txt exported from the ima client into a factor.

Place exported files under the kb_export folder in the stock data directory, then run.
Output: backtest_output/kb_local_signals.csv
"""
import sys, os, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import pandas as pd

DATA = Path(os.environ.get("STOCK_DATA_DIR", r"F:\\even-codex\\us-stock-data"))
EXPORT = DATA / "kb_export"
OUT = Path(__file__).resolve().parent / "backtest_output"
EXPORT.mkdir(parents=True, exist_ok=True)

POS = ["\u4e0a\u8c03","\u8d85\u9884\u671f","\u589e\u6301","\u4e70\u5165","\u8d85\u914d","\u5f3a\u52b2","\u63d0\u5347","\u4e0a\u4fee","\u79ef\u6781","\u5411\u597d","\u666f\u6c14","\u7ffb\u500d","\u6539\u5584","\u5229\u597d","\u7a81\u7834","\u52a0\u901f"]
NEG = ["\u4e0b\u8c03","\u51cf\u6301","\u4f4e\u914d","\u4e0d\u53ca\u9884\u671f","\u6076\u5316","\u98ce\u9669","\u62d6\u7d2f","\u627f\u538b","\u538b\u5236","\u4e0b\u6ed1","\u8d70\u5f31","\u9006\u98ce","\u524a\u51cf","\u62c5\u5fe7","\u4f4e\u4e8e","\u5229\u7a7a","\u56de\u843d","\u4e8f\u635f"]

def parse_date(s):
    m = re.search(r"(20\\d{2})(\\d{2})(\\d{2})", s)
    if m:
        y, mo, da = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= da <= 31:
            return f"{y:04d}-{mo:02d}-{da:02d}"
    m2 = re.search(r"(\\d{1,2})\u6708(\\d{1,2})\u65e5", s)
    if m2:
        mo, da = int(m2.group(1)), int(m2.group(2))
        if 1 <= mo <= 12 and 1 <= da <= 31:
            yy = 2025 if mo > 8 else 2026
            return f"{yy}-{mo:02d}-{da:02d}"
    return None

def main():
    files = list(EXPORT.rglob("*.md")) + list(EXPORT.rglob("*.txt"))
    rows = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        d = parse_date(f.name) or parse_date(text[:500])
        tickers = set(x.upper() for x in re.findall(r"([A-Z][A-Z0-9]{1,5})\\.US", text, re.I))
        tickers |= set(x.upper() for x in re.findall(r"[\uff08(]([A-Z][A-Z0-9]{1,5})[\uff09)]", text))
        for t in tickers:
            pos = [k for k in POS if k in text]
            neg = [k for k in NEG if k in text]
            sgn = 1 if pos and not neg else (-1 if neg and not pos else 0)
            rows.append(dict(signal_date=d, ticker=t, file=str(f), n_pos=len(pos), n_neg=len(neg), sign=sgn))
    df = pd.DataFrame(rows).drop_duplicates(subset=["signal_date", "ticker", "file"])
    df.to_csv(OUT / "kb_local_signals.csv", index=False, encoding="utf-8-sig")
    print(f"Imported {len(files)} file(s), parsed {len(df)} signals, covered {df['signal_date'].nunique() if len(df) else 0} trading days")
    if len(df):
        print(df.groupby("ticker")["sign"].sum().sort_values(ascending=False).to_string())

if __name__ == "__main__":
    main()
