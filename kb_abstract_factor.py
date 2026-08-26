# -*- coding: utf-8 -*-
import json, re, sys, csv, datetime
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path(r"F:\even-codex\us-stock-data")
LIST = DATA / "ima_all_meta.json"
OUT = Path(__file__).resolve().parent / "backtest_output"
OUT.mkdir(parents=True, exist_ok=True)

KNOWN = set()
prices = DATA / "prices.csv"
if prices.exists():
    with prices.open(newline="", encoding="utf-8", errors="ignore") as f:
        r = csv.reader(f); head = next(r, None)
        if head: KNOWN = {c.strip().upper() for c in head[1:]}

POS = ["上调","超预期","增持","买入","超配","强劲","提升","积极","向好","景气","改善","利好","突破","加速","看多","乐观","推荐","加持","低估","布局","优于预期","低估","超预期"]
NEG = ["下调","减持","低配","不及预期","恶化","风险","拖累","承压","压制","下滑","走弱","逆风","削减","担忧","低于","利空","回落","亏损","看空","谨慎","预警","低于预期","估值过高"]

STOP = set("""A I AN IN ON AT TO OF FOR THE AND IS ARE HAS HAVE BE NEW NEXT THIS THAT AS BY WITH IT ITS ALL SO IF OR WE YOU OUR CAN MAY NOT NO MORE ALSO US DR MR CEO CFO CTO EVP VP PM AM ET GMT BOX DAY YEAR LOW OPEN TOP ADD CUT BUY SELL HOLD DATA RISK ONE TWO THREE VOL
""".split())
STOP |= {"TECH","NEWS","REPORT","DAILY","MARKET","TIME","GPU","CPU","AI","SAAS","CBS"}

def from_create_time(it):
    ts = str(it.get("create_time") or "0").strip()
    if not ts.isdigit():
        return ""
    ms = int(ts)
    sec = ms/1000 if ms > 100000000000 else ms
    try:
        return datetime.datetime.fromtimestamp(sec).strftime("%Y-%m-%d")
    except Exception:
        return ""

def parse_date(it):
    title = it.get("title","")
    m = re.search(r"(20\d{2})-(\d{1,2})-(\d{1,2})", title)
    if m:
        y,mo,da=map(int,m.groups()); return f"{y:04d}-{mo:02d}-{da:02d}"
    m = re.search(r"(20\d{2})[年\\-](\d{1,2})[月\\-](\d{1,2})[日]?", title)
    if m:
        y,mo,da=map(int,m.groups()); return f"{y:04d}-{mo:02d}-{da:02d}"
    m = re.search(r"(?:^|[^0-9])(\d{2})[\/\-_]?(\d{2})(?:[^0-9]|$)", title)
    if m:
        mo,da=map(int,m.groups())
        if 1<=mo<=12 and 1<=da<=31:
            y = 2025 if mo > 8 else 2026
            return f"{y}-{mo:02d}-{da:02d}"
    m = re.search(r"(\d{1,2})月(\d{1,2})日", title)
    if m:
        mo,da=map(int,m.groups()); y = 2025 if mo > 8 else 2026
        return f"{y}-{mo:02d}-{da:02d}"
    return from_create_time(it)

def scan(text):
    toks=set()
    for t in re.finditer(r"(?<![\w.])([A-Z][A-Z0-9]{1,5})\.US", text, re.I):
        tok=t.group(1).upper()
        if len(tok)>=2: toks.add(tok)
    for t in re.finditer(r"(?<![\w.])([A-Z]{2,6})(?![\w])", text):
        tok=t.group(1)
        if tok in STOP or len(tok)>6: continue
        toks.add(tok)
    for t in re.finditer(r"[\uff08(]([A-Z]{2,6})[\uff09)]", text):
        tok=t.group(1)
        if tok not in STOP: toks.add(tok)
    if KNOWN:
        toks={t for t in toks if t in KNOWN}
    return toks

def main():
    items=json.loads(LIST.read_text(encoding="utf-8"))
    rows=[]
    for it in items:
        title=it.get("title","")
        text=(it.get("abstract","") or "") + " " + (it.get("introduction","") or "")
        d=parse_date(it)
        src=it.get("source_folder","")
        for t in sorted(scan(text)):
            pos=sum(text.count(k) for k in POS); neg=sum(text.count(k) for k in NEG)
            sign=1 if pos and not neg else (-1 if neg and not pos else 0)
            rows.append({"source_folder":src,"title":title,"pdf_date":d,"media_id":it.get("media_id",""),"ticker":t,"n_pos":pos,"n_neg":neg,"sign":sign})
    out=OUT/"kb_abstract_factors.csv"
    with out.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["source_folder","title","pdf_date","media_id","ticker","n_pos","n_neg","sign"]); w.writeheader(); w.writerows(rows)
    import collections
    cnt=collections.Counter(r["ticker"] for r in rows); daily=collections.Counter(r["pdf_date"] for r in rows); src=collections.Counter(r["source_folder"] for r in rows)
    ds=sorted(d for d in daily if d)
    print("items",len(items),"factor_rows",len(rows),"unique_tickers",len(cnt))
    print("by_source",dict(src))
    print("top_mentions",cnt.most_common(25))
    print("distinct_dates",len(daily),"range",(ds[0],ds[-1]) if ds else None,"no_date",daily.get("",0))
    print("saved",out)
if __name__=="__main__": main()
