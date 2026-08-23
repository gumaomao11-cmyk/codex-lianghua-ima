# -*- coding: utf-8 -*-
"""KB cache PDF extraction and factor import (improved quiet version).

Reads PDFs from the local ima.copilot browser cache, saves the readable PDF
and extracted text under the private data directory, then builds local
ticker mention / sentiment factor rows from that text.

This only reads files already cached by the IMA client on this machine. Nothing
from the subscribed library is uploaded to any remote repository.
"""
import sys, os, re, logging, contextlib, io
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
logging.getLogger("pypdf").setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)
import pypdf

CACHE_DIR = Path(os.environ.get("IMA_CACHE_DIR", r"C:\Users\ASUS\AppData\Local\ima.copilot\User Data\Default\Cache\Cache_Data"))
DATA = Path(os.environ.get("STOCK_DATA_DIR", r"F:\even-codex\us-stock-data"))
_KNOWN_CSV = DATA / "prices.csv"
KNOWN_TICKERS = set()
if _KNOWN_CSV.exists():
    import csv as _kcsv
    with _KNOWN_CSV.open(newline="", encoding="utf-8", errors="ignore") as _kf:
        _kreader = _kcsv.reader(_kf)
        _khead = next(_kreader, [])
        KNOWN_TICKERS = {_c.strip() for _c in _khead[1:]}
PDF_DIR = DATA / "kb_cache_pdfs"
TEXT_DIR = DATA / "kb_cache_text"
OUT = Path(__file__).resolve().parent / "backtest_output"
for p in (CACHE_DIR, PDF_DIR, TEXT_DIR, OUT):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

POS = ["\u4e0a\u8c03","\u8d85\u9884\u671f","\u589e\u6301","\u4e70\u5165","\u8d85\u914d","\u5f3a\u52b2","\u63d0\u5347","\u4e0a\u4fee","\u79ef\u6781","\u5411\u597d","\u666f\u6c14","\u7ffb\u500d","\u6539\u5584","\u5229\u597d","\u7a81\u7834","\u52a0\u901f"]
NEG = ["\u4e0b\u8c03","\u51cf\u6301","\u4f4e\u914d","\u4e0d\u53ca\u9884\u671f","\u6076\u5316","\u98ce\u9669","\u62d6\u7d2f","\u627f\u538b","\u538b\u5236","\u4e0b\u6ed1","\u8d70\u5f31","\u9006\u98ce","\u524a\u51cf","\u62c5\u5fe7","\u4f4e\u4e8e","\u5229\u7a7a","\u56de\u843d","\u4e8f\u635f"]

def complete_pdf(data):
    if not data.startswith(b"%PDF"):
        return None
    e = data.find(b"%%EOF")
    if e < 0:
        return None
    return data[: e + 5]

def parse_date(text):
    m = re.search(r"(20\\d{2})[-/]?(\\d{1,2})[-/](\\d{1,2})", text)
    if m:
        try:
            y, mo, da = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= mo <= 12 and 1 <= da <= 31:
                return f"{y:04d}-{mo:02d}-{da:02d}"
        except Exception:
            pass
    m2 = re.search(r"(\\d{1,2})\u6708(\\d{1,2})\u65e5", text)
    if m2:
        try:
            mo, da = int(m2.group(1)), int(m2.group(2))
            yy = 2025 if mo > 8 else 2026
            return f"{yy}-{mo:02d}-{da:02d}"
        except Exception:
            pass
    return None

def extract_pdf_text(data):
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        text = "".join((p.extract_text() or "") for p in reader.pages)
        title = (reader.metadata.title if reader.metadata else None) or ""
        return len(reader.pages), text, str(title)
    except Exception:
        return None

def scan_factors(source, text, date):
    tickers = set(x.upper() for x in re.findall(r"([A-Z][A-Z0-9]{1,5})\\.US", text, re.I))
    tickers |= set(x.upper() for x in re.findall(r"[\uff08(]([A-Z][A-Z0-9]{1,5})[\uff09)]", text))
    rows = []
    for t in tickers:
        if len(t) < 2 or (KNOWN_TICKERS and t not in KNOWN_TICKERS):
            continue
        pos = sum(text.count(k) for k in POS)
        neg = sum(text.count(k) for k in NEG)
        sign = 1 if pos and not neg else (-1 if neg and not pos else 0)
        rows.append({"source": source, "pdf_date": date, "ticker": t, "n_pos": pos, "n_neg": neg, "sign": sign})
    return rows

def extract_new_from_cache():
    if not CACHE_DIR.is_dir():
        print("IMA cache dir not found:", CACHE_DIR)
        return 0, 0, 0
    files = sorted(CACHE_DIR.glob("f_*"))
    new_pdfs = 0
    new_texts = 0
    seen = set()
    for f in files:
        try:
            data = f.read_bytes()
        except Exception:
            continue
        payload = complete_pdf(data)
        if not payload:
            continue
        seen.add(f.name)
        pdf_path = PDF_DIR / (f.name + ".pdf")
        if not pdf_path.exists() or pdf_path.stat().st_size != len(payload):
            pdf_path.write_bytes(payload)
            new_pdfs += 1
        txt_path = TEXT_DIR / (f.name + ".txt")
        if txt_path.exists():
            continue
        parsed = extract_pdf_text(payload)
        if not parsed:
            continue
        n_pages, text, title = parsed
        if not text.strip():
            continue
        header = f"# source={f.name}\n# title={title}\n# pages={n_pages}\n# date_file={f.name}\n\n"
        txt_path.write_text(header + text, encoding="utf-8")
        new_texts += 1
    return len(files), new_pdfs, new_texts

def build_factors_from_text():
    rows = []
    for txt in sorted(TEXT_DIR.glob("*.txt")):
        try:
            text = txt.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        d = parse_date(text[:4000])
        rows.extend(scan_factors(txt.stem, text, d))
    return rows

def write_factors(rows):
    import csv
    with open(OUT / "kb_cache_factors.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "pdf_date", "ticker", "n_pos", "n_neg", "sign"])
        for r in rows:
            w.writerow([r["source"], r["pdf_date"], r["ticker"], r["n_pos"], r["n_neg"], r["sign"]])

def main():
    total, new_pdfs, new_texts = extract_new_from_cache()
    rows = build_factors_from_text()
    write_factors(rows)
    print(f"cache_files_scanned={total} new_pdfs={new_pdfs} new_texts={new_texts} text_rows_total={len(rows)}")
    if rows:
        import collections
        cnt = collections.Counter(r["ticker"] for r in rows)
        print("top_tickers:", cnt.most_common(20))
        daily = collections.Counter((r["pdf_date"] or "unknown") for r in rows)
        print("daily_dates_with_rows:", len(daily))

if __name__ == "__main__":
    main()
