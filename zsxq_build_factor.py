# -*- coding: utf-8 -*-
"""从知识星球主题构建 ZSXQ 词频/提及因子（schema 对齐 kb_abstract_factors.csv）。
输出 backtest_output/zsxq_factors.csv（本地产出，不进 GitHub）。
"""
import sys, re, json, csv, datetime
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path(r"F:\even-codex\us-stock-data")
SRC  = Path(r"F:\even-codex\lianghua+IMA\backtest_output\zsxq_topics.json")
OUT  = Path(r"F:\even-codex\lianghua+IMA\backtest_output\zsxq_factors.csv")

KNOWN = set()
prices = DATA / "prices.csv"
if prices.exists():
    with prices.open(newline="", encoding="utf-8", errors="ignore") as f:
        r = csv.reader(f); head = next(r, None)
        if head: KNOWN = {c.strip().upper() for c in head[1:]}

POS = ["上调","超预期","增持","买入","超配","强劲","提升","积极","向好","景气","改善","利好","突破","加速","看多","乐观","推荐","加持","低估","布局","优于预期","低估","超预期","爆发","大涨","涨停","新高","龙头"]
NEG = ["下调","减持","低配","不及预期","恶化","风险","拖累","承压","压制","下滑","走弱","逆风","削减","担忧","低于","利空","回落","亏损","看空","谨慎","预警","低于预期","估值过高","做空","造假","爆雷","大跌","套现","风险提示"]
STOP = set("""A I AN IN ON AT TO OF FOR THE AND IS ARE HAS HAVE BE NEW NEXT THIS THAT AS BY WITH IT ITS ALL SO IF OR WE YOU OUR CAN MAY NOT NO MORE ALSO US DR MR CEO CFO CTO EVP VP PM AM ET GMT BOX DAY YEAR LOW OPEN TOP ADD CUT BUY SELL HOLD DATA RISK ONE TWO THREE VOL
""".split())
STOP |= {"TECH","NEWS","REPORT","DAILY","MARKET","TIME","GPU","CPU","AI","SAAS","CBS","SIGN","QUANT","MACRO"}

def date_of(s):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s or "")
    return m.group(1) if m else ""

def parse_ts(ts):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", ts or "")
    return m.group(1) if m else ""

def scan(text):
    toks = set()
    for t in re.finditer(r"(?<![\w.])([A-Z][A-Z0-9]{1,5})\.US", text, re.I):
        tok = t.group(1).upper()
        if len(tok) >= 2: toks.add(tok)
    for t in re.finditer(r"(?<![\w.])([A-Z]{2,6})(?![\w])", text):
        tok = t.group(1)
        if tok in STOP or len(tok) > 6: continue
        toks.add(tok)
    for t in re.finditer(r"[\uff08(]([A-Z]{2,6})[\uff09)]", text):
        tok = t.group(1)
        if tok not in STOP: toks.add(tok)
    if KNOWN: toks = {t for t in toks if t in KNOWN}
    return toks

items = json.loads(SRC.read_text(encoding="utf-8"))
rows = []
for it in items:
    title = it.get("title","") or ""
    content = it.get("content","") or ""
    text = title + " " + content
    d = date_of(it.get("create_time","")) or parse_ts(it.get("create_time",""))
    gname = ((it.get("group") or {}).get("name") or "").strip()
    src = gname or "知识星球"
    pos_c = sum(text.count(k) for k in POS)
    neg_c = sum(text.count(k) for k in NEG)
    sign = 1 if (pos_c and not neg_c) else (-1 if (neg_c and not pos_c) else 0)
    for t in sorted(scan(text)):
        rows.append({"source_folder":src,"title":title[:120],"pdf_date":d,
                     "media_id":it.get("topic_id",""),"ticker":t,
                     "n_pos":pos_c,"n_neg":neg_c,"sign":sign})
with OUT.open("w",encoding="utf-8-sig",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["source_folder","title","pdf_date","media_id","ticker","n_pos","n_neg","sign"])
    w.writeheader(); w.writerows(rows)
import collections
cnt = collections.Counter(r["ticker"] for r in rows); daily = collections.Counter(r["pdf_date"] for r in rows); src = collections.Counter(r["source_folder"] for r in rows)
ds = sorted(d for d in daily if d)
print("topics",len(items),"rows",len(rows),"tickers",len(cnt))
print("by_source",dict(src))
print("top",cnt.most_common(25))
print("dates",len(daily),"range",(ds[0],ds[-1]) if ds else None)
print("saved",OUT)
