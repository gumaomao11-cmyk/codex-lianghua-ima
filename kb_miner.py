# -*- coding: utf-8 -*-
"""更深的 IMA 检索挖掘：股票池+主题词，标题和文件夹名都补日期 → 日度提及热度"""
import sys, subprocess, json, time, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import pandas as pd
IMA_API = r"C:\Users\ASUS\.codex\skills\ima-skill\ima_api.cjs"
KB = "MUb6MX2SCTN5Xi2EjCPBsHHuWODJ-fHkL7lSAXe_BdE="
OUT = Path(r"F:\even-codex\lianghua2\backtest_output")

STOCKS = ["MU","INTC","WDC","MRVL","STX","NBIS","GLW","AMAT","AMD","FLEX",
          "NVDA","AVGO","SMCI","ORCL","CRWV","MSFT","LITE","HPE","FTNT","PANW",
          "VRT","CIEN","TER","TDY","ONTO","LRCX","CCL","GDS","DVA","AXON"]
THEMES = ["存储","HBM","AI服务器","光模块","数据中心","AIDC","ASIC","CPO","半导体设备","铜缆","液冷","电源","PCB"]
QUERIES = STOCKS + THEMES

FOLDER_DATE = {
 "folder_7495854348857629":"2026-08-19",
 "folder_7496216044637152":"2026-08-20",
 "folder_7496954556717169":"2026-08-21",
}

def api(path, body, tries=6):
    for k in range(tries):
        try:
            p=subprocess.run(["node", IMA_API, "openapi/wiki/v1/"+path, json.dumps(body, ensure_ascii=False)],
                             capture_output=True, text=True, encoding="utf-8", timeout=40)
            r=json.loads(p.stdout.strip() or "{}")
        except Exception as e:
            r={"code":-1,"msg":str(e)}
        if r.get("code") in (200001,110021):
            time.sleep(1.5+k*2.0); continue
        time.sleep(0.8)
        return r
    return r

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

def search(q):
    rows=[]; cur=""
    for _ in range(6):
        r=api("search_knowledge", {"query":q, "knowledge_base_id":KB, "cursor":cur, "limit":50})
        if r.get("code")!=0:
            print(f"  [{q}] err {r.get('code')} {r.get('msg')}"); break
        data=r.get("data") or {}; rows.extend(data.get("info_list") or [])
        if data.get("is_end") is True or not data.get("next_cursor"): break
        cur=data.get("next_cursor")
    return rows

allhit=[]
for q in QUERIES:
    hits=search(q)
    for h in hits:
        allhit.append(dict(q=q, title=h.get("title"), media_id=h.get("media_id"), folder=h.get("parent_folder_id")))
    print(f"{q:<10} -> {len(hits)} 命中")
    time.sleep(0.4)

# 去重并写原始命中
uniq={}
for h in allhit:
    key=(h["q"], h["title"])
    if key not in uniq: uniq[key]=h
raw=list(uniq.values())
(OUT/"kb_mention_raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")

# 聚合
rows=[]
for h in raw:
    d=parse_date(h["title"]) or FOLDER_DATE.get(h.get("folder"))
    if not d: continue
    kind="ticker" if h["q"].isalpha() and h["q"] in STOCKS else "theme"
    rows.append(dict(date=d, q=h["q"], kind=kind, title=h["title"]))

df=pd.DataFrame(rows)
agg=df.groupby(["date","q"]).size().rename("n").reset_index()
agg.to_csv(OUT/"kb_mention_miner_daily.csv", index=False, encoding="utf-8-sig")
print("\n保存:", OUT/"kb_mention_raw.json", "条", len(raw))
print("可定日命中:", len(df), "行,", df["date"].nunique(), "个交易日")
print("\n月份覆盖:")
print(df["date"].str[:7].value_counts().sort_index().to_string())
print("\n覆盖最多的 ticker:")
if len(agg):
    t=agg[agg["q"].isin(STOCKS)].groupby("q")["n"].sum().sort_values(ascending=False)
    print(t.head(15).to_string())
