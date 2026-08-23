# -*- coding: utf-8 -*-
"""扫描浑水调研知识库的两个日更归档文件夹，分页拉全，缓存到 JSON"""
import sys, subprocess, json, time, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
IMA_API = r"C:\Users\ASUS\.codex\skills\ima-skill\ima_api.cjs"
KB = "MUb6MX2SCTN5Xi2EjCPBsHHuWODJ-fHkL7lSAXe_BdE="
OUT = Path(r"F:\even-codex\lianghua2\backtest_output")
folders = [
    ("root", "root", "根目录"),
    ("US_tech_daily", "folder_7480035300630799", "美国科技日报（日更）"),
    ("daily_research_summary", "folder_7478540115130568", "每天投研资料AI总结（日更）"),
    ("date_0821_23", "folder_7496954556717169", "8月21-23日"),
    ("date_0820", "folder_7496216044637152", "8月20日"),
    ("date_0819", "folder_7495854348857629", "8月19日"),
]

def api(path, body, tries=4):
    for k in range(tries):
        p = subprocess.run(["node", IMA_API, "openapi/wiki/v1/" + path, json.dumps(body, ensure_ascii=False)],
                           capture_output=True, text=True, encoding="utf-8", timeout=40)
        raw = p.stdout.strip()
        try:
            r = json.loads(raw)
        except Exception:
            r = {"code": -1, "msg": raw[:200]}
        if r.get("code") == 110021:
            time.sleep(2 + k * 2); continue
        time.sleep(0.15)
        return r, p.stderr.strip()[:300]
    return r, "retry exhausted"

def collect(folder_id, label):
    cur = ""; items = []
    while True:
        params = {"knowledge_base_id": KB, "cursor": cur, "limit": 50}
        if folder_id != "root": params["folder_id"] = folder_id
        r, err = api("get_knowledge_list", params)
        if r.get("code") != 0:
            print(f"[{label}] err {r.get('code')} {r.get('msg')} stderr={err}")
            break
        data = r.get("data") or {}
        arr = data.get("knowledge_list") or []
        for it in arr:
            items.append({"folder": label, "media_id": it.get("media_id"), "title": it.get("title"),
                          "parent_folder_id": it.get("parent_folder_id"), "media_type": it.get("media_type")})
        if data.get("is_end") is True or not data.get("next_cursor"):
            break
        cur = data.get("next_cursor")
        if len(items) > 20000:
            print(f"[{label}] cap 20000 reached, stop"); break
    return items

all_items = []
for label, fid, name in folders:
    its = collect(fid, label)
    print(f"{label} ({name}): {len(its)} items")
    all_items.extend(its)

out = OUT / "kb_raw_items.json"
out.write_text(json.dumps(all_items, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved", out, len(all_items), "records")
