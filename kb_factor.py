# -*- coding: utf-8 -*-
"""把 kb_raw_items.json 解析为：日期/ticker/机构/情绪 的信息因子，并输出汇总"""
import sys, json, re
from pathlib import Path
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(r"F:\even-codex\lianghua2\backtest_output")
items = json.loads((OUT/"kb_raw_items.json").read_text(encoding="utf-8"))

FIRMS = ["摩根大通","大摩","摩根士丹利","花旗","高盛","德银","德意志","汇丰","瑞银","美银","巴克莱","野村","伯恩斯坦","中金","国金","高至"]
POS = ["上调","超预期","增持","买入","超配","强劲","提升","上修","积极","向好","景气","超配","跑赢","升级","翻倍","改善"]
NEG = ["下调","减持","低配","不及预期","恶化","风险","拖累","承压","压制","下滑","走弱","逆风","削减","担忧","低于","利空","回落"]

def parse_date(t):
    y=y0=y1=None
    m1=re.search(r'(20\d{2})(\d{2})(\d{2})', t)
    if m1: return f"{m1.group(1)}-{m1.group(2)}-{m1.group(3)}"
    m2=re.search(r'(\d{1,2})月(\d{1,2})日', t)
    if m2:
        mo,da=int(m2.group(1)),int(m2.group(2))
        yy=2025 if mo>8 else 2026
        return f"{yy}-{mo:02d}-{da:02d}"
    m3=re.search(r'(?<!\d)(\d{2})(\d{2})(?=美股|盘前|盘后|晚|早|日|电话会|_原文|_|\.|\s|20\d{2})', t)
    if m3:
        mo,da=int(m3.group(1)),int(m3.group(2))
        yy=2025 if mo>8 else 2026
        return f"{yy}-{mo:02d}-{da:02d}"
    return None

def extract_tickers(t):
    res=set()
    for m in re.finditer(r'([A-Z][A-Z0-9]{0,5})\.US', t, re.I):
        res.add(m.group(1).upper())
    return sorted(res)

def firm(t):
    for f in FIRMS:
        if f in t: return f
    return ""

def sentiment(t):
    pos=[k for k in POS if k in t]; neg=[k for k in NEG if k in t]
    sgn = 1 if pos and not neg else (-1 if neg and not pos else 0)
    return sgn, ",".join(pos), ",".join(neg)

rows=[]
for it in items:
    t=it["title"]; d=parse_date(t); tk=extract_tickers(t)
    sgn,posk,negk=sentiment(t); f=firm(t)
    if not tk: continue
    for tick in tk:
        rows.append(dict(signal_date=d, ticker=tick, source=it["folder"], firm=f, sign=sgn,
                         pos=posk, neg=negk, title=t, media_id=it["media_id"]))
df=pd.DataFrame(rows)
df.to_csv(OUT/"kb_signals.csv", index=False, encoding="utf-8-sig")
print(f"解析到 {len(df)} 条 ticker 级信号（{df["signal_date"].nunique()} 个交易日，{df["ticker"].nunique()} 个美股 ticker）")
print("\n按月份覆盖（有信号的月）:")
tmp=df.copy(); tmp["ym"]=tmp["signal_date"].str[:7]
print(tmp[tmp["signal_date"].notna()]["ym"].value_counts().sort_index().to_string())
print("\n各 ticker 净信号（正向-负向, 只看有信号）:")
net=df.groupby("ticker")["sign"].sum().sort_values(ascending=False)
print(net[net.abs()>0].to_string())
print("\n当前策略 10 只的信号热度:")
cur=["MU","WDC","INTC","STX","MRVL","NBIS","AMD","AMAT","GLW","FLEX"]
for c in cur:
    if c in net.index:
        print(f"  {c:<6} 净信号={net[c]:+d}  (正{ (df[(df["ticker"]==c)&(df["sign"]>0)].shape[0]) }, 负{ (df[(df["ticker"]==c)&(df["sign"]<0)].shape[0]) })")
