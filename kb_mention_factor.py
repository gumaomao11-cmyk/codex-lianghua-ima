# -*- coding: utf-8 -*-
"""基于 search_knowledge 命中构建【提及频率因子】：不需要下载PDF，标题带日期即可按日聚合。
输入: backtest_output/kb_search_signals.csv (已检索的 825 条命中)
输出: backtest_output/kb_mention_daily.csv (date,ticker,n_mention,n_pos,n_neg,net)
"""
import sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import pandas as pd
OUT=Path(r"F:\even-codex\lianghua2\backtest_output")
df=pd.read_csv(OUT/"kb_search_signals.csv")

def parse_date(t):
    m=re.search(r'(20\d{2})(\d{2})(\d{2})', str(t))
    if m:
        y,mo,da=int(m.group(1)),int(m.group(2)),int(m.group(3))
        if 1<=mo<=12 and 1<=da<=31: return f"{y:04d}-{mo:02d}-{da:02d}"
    m2=re.search(r'(\d{1,2})月(\d{1,2})日', str(t))
    if m2:
        mo,da=int(m2.group(1)),int(m2.group(2))
        if 1<=mo<=12 and 1<=da<=31:
            yy=2025 if mo>8 else 2026; return f"{yy}-{mo:02d}-{da:02d}"
    return None

df["d"]=df["title"].map(parse_date)
df=df[df["d"].notna()].copy().drop_duplicates(subset=["d","ticker","media_id"])
df["ym"]=df["d"].str[:7]

agg=df.groupby(["d","ticker"]).agg(n=("sign","size"), pos=("sign",lambda s:(s>0).sum()), neg=("sign",lambda s:(s<0).sum()))
agg["net"]=agg["pos"]-agg["neg"]
agg.reset_index().sort_values(["d","ticker"]).to_csv(OUT/"kb_mention_daily.csv", index=False, encoding="utf-8-sig")

print("mention 日度因子:", len(agg), "行,", df["d"].nunique(), "个交易日,", df["ticker"].nunique(), "个ticker")
print("\n月份覆盖:")
print(df["ym"].value_counts().sort_index().to_string())
print("\n当前策略10只的提及情况（去重媒体后按日）:")
cur=["MU","WDC","INTC","STX","MRVL","NBIS","AMD","AMAT","GLW","FLEX"]
for c in cur:
    sub=df[df["ticker"]==c]
    if len(sub):
        days=len(sub["d"].unique())
        avg=len(sub)/days
        print(f"  {c:<6} 命中{len(sub):>3}  覆盖{days:>3}日  均提及/日={avg:.2f}  (+{int(sub.sign.sum())})")
