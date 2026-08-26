# -*- coding: utf-8 -*-
"""遍历【浑水调研】共享库的外资研报文件夹，导出标题清单(带日期)到 data/。
"""
import json, re, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import ima_openapi as io

KB = "RJxmvo3aEkYpjzFlq7Z_wcU8GFQQXidQx5h7KETo-c4="
FOLDER = "folder_7495389305403551"
OUT = Path("data/ima_waizi_folder.csv")

def dump(folder_id, name):
    rows, cursor, guard = [], "", 0
    while True and guard < 2000:
        r = io.api("get_knowledge_list",
                   {"knowledge_base_id": KB, "folder_id": folder_id, "cursor": cursor, "limit": 50})
        d = r.get("data", {})
        if r.get("code") != 0:
            print(json.dumps(r, ensure_ascii=False)[:300]); break
        kl = d.get("knowledge_list", []) or []
        for it in kl:
            rows.append(it)
        if d.get("is_end"):
            break
        cursor = d.get("next_cursor", "")
        if not cursor:
            break
        guard += 1
        time.sleep(0.15)
    return rows

rows = dump(FOLDER, "外资研报")
OUT.parent.mkdir(exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    f.write("title\tmedia_id\tmedia_type\tparent_folder_id\n")
    for r in rows:
        f.write("\t".join([str(r.get("title","")), str(r.get("media_id","")),
                            str(r.get("media_type","")), str(r.get("parent_folder_id",""))])+"\n")
print("外资研报条目数:", len(rows))
dates = sorted(set(re.findall(r"2\d{7}", " ".join(r.get("title","") for r in rows))))
yrs = sorted(set(re.findall(r"(20\d{2})", " ".join(r.get("title","") for r in rows))))
print("标题中含 8 位日期 数量:", len(dates), "范围:", (dates[0], dates[-1]) if dates else None)
print("标题中含年份:", yrs)
types = {}
for r in rows: types[r.get("media_type")] = types.get(r.get("media_type"),0)+1
print("media_type分布:", types)
print("已保存:", OUT.resolve())
