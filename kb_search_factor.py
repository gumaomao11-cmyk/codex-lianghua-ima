# -*- coding: utf-8 -*-
"""基于 search_knowledge 检索构建 KB 信息因子：按 ticker 检索→解析日期/机构/情绪→输出日度信号"""
import sys, subprocess, json, time, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import pandas as pd
IMA_API = r"C:\Users\ASUS\.codex\skills\ima-skill\ima_api.cjs"
KB = "MUb6MX2SCTN5Xi2EjCPBsHHuWODJ-fHkL7lSAXe_BdE="
OUT = Path(r"F:\even-codex\lianghua2\backtest_output")
UNIV = ["MU","INTC","WDC","MRVL","STX","NBIS","GLW","AMAT","AMD","FLEX",
        "NVDA","AVGO","SMCI","ORCL","CRWV","MSFT","LITE","HPE","FTNT","PANW",
        "DVA","HBM","CIEN","TER","VRT"]

FIRMS=["摩根大通","大摩","摩根士丹利","花旗","高盛","德银","德意志","汇丰","瑞银","美银","巴克莱","野村","伯恩斯坦","中金","国金"]
POS=["上调","超预期","增持","买入","超配","强劲","提升","上修","积极","向好","景气","跑赢","升级","翻倍","改善","重申"]
NEG=["下调","减持","低配","不及预期","恶化","风险","拖累","承压","压制","下滑","走弱","逆风","削减","担忧","低于","利空","回落"]

def api(path, body, tries=6):
    for k in range(tries):
        try:
            p=subprocess.run(["node", IMA_API, "openapi/wiki/v1/"+path, json.dumps(body, ensure_ascii=False)],
                             capture_output=True, text=True, encoding="utf-8", timeout=40)
            r=json.loads(p.stdout.strip() or "{}")
        except Exception as e:
            r={"code":-1,"msg":str(e)}
        if r.get("code") in (200001,110021):
            time.sleep(1.5+k*1.5); continue
        time.sleep(0.7)
        return r
    return r

def parse_date(t):
    m=re.search(r'(20\d{2})(\d{2})(\d{2})', t)
    if m: return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m2=re.search(r'(\d{1,2})月(\d{1,2})日', t)
    if m2:
        mo,da=int(m2.group(1)),int(m2.group(2)); yy=2025 if mo>8 else 2026
        return f"{yy}-{mo:02d}-{da:02d}"
    m3=re.search(r'(?<!\d)(\d{2})(\d{2})(?=美股|盘前|盘后|晚|早|日|电话会|_原文|_|\.|\s)', t)
    if m3:
        mo,da=int(m3.group(1)),int(m3.group(2)); yy=2025 if mo>8 else 2026
        return f"{yy}-{mo:02d}-{da:02d}"
    return None

def firm(t):
    for f in FIRMS:
        if f in t: return f
    return ""

def sentiment(t):
    pos=[k for k in POS if k in t]; neg=[k for k in NEG if k in t]
    sgn=1 if pos and not neg else (-1 if neg and not pos else 0)
    return sgn

def search_ticker(q):
    rows=[]; cur=""
    for _ in range(8):
        r=api("search_knowledge", {"query":q, "knowledge_base_id":KB, "cursor":cur, "limit":50})
        if r.get("code")!=0:
            print(f"  [search {q}] err {r.get('code')} {r.get('msg')}"); break
        items=(r.get("data") or {}).get("info_list") or []
        for it in items:
            rows.append(it)
        data=r.get("data") or {}
        if data.get("is_end") is True or not data.get("next_cursor"): break
        cur=data.get("next_cursor")
    return rows

allrows=[]
for q in UNIV:
    hits=search_ticker(q)
    found=0
    for it in hits:
        t=it["title"]; d=parse_date(t); st=sentiment(t)
        allrows.append(dict(query=q, signal_date=d, ticker=q, firm=firm(t), sign=st,
                            title=t, media_id=it.get("media_id"), folder=it.get("parent_folder_id")))
        found+=1
    print(f"{q:<6} -> {found} 条命中")
    time.sleep(0.5)

df=pd.DataFrame(allrows)
df.to_csv(OUT/"kb_search_signals.csv", index=False, encoding="utf-8-sig")
print("\nsaved", OUT/"kb_search_signals.csv", "共", len(df), "条")
if len(df):
    dd=df[df["signal_date"].notna()].copy()
    print("带日期的命中:", len(dd), "个交易日", dd["signal_date"].nunique(), "，月份分布:")
    print(dd["signal_date"].str[:7].value_counts().sort_index().to_string())
