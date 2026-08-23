# -*- coding: utf-8 -*-
"""扫描 IMA 订阅知识库目录：分页拉取根目录+一层文件夹，统计归档覆盖与文件数"""
import sys, subprocess, json, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
IMA_API = r"C:\Users\ASUS\.codex\skills\ima-skill\ima_api.cjs"
KB = "MUb6MX2SCTN5Xi2EjCPBsHHuWODJ-fHkL7lSAXe_BdE="

def api(path, body):
    p = subprocess.run(["node", IMA_API, "openapi/wiki/v1/" + path, json.dumps(body, ensure_ascii=False)],
                       capture_output=True, text=True, encoding="utf-8", timeout=30)
    raw = p.stdout.strip()
    try:
        return json.loads(raw)
    except Exception:
        return {}
    finally:
        time.sleep(0.25)

def paginate(path, params, key):
    cur = ""
    items = []
    for _ in range(6):
        r = api(path, dict(params, cursor=cur))
        if r.get("code") != 0:
            print("  err:", r.get("code"), r.get("msg")); break
        arr = (r.get("data") or {}).get(key) or []
        items.extend(arr)
        data = r.get("data") or {}
        if data.get("is_end") is True: break
        cur = data.get("next_cursor") or ""
        if not cur: break
    return items

top = paginate("get_knowledge_list", {"knowledge_base_id": KB, "limit": 50}, "knowledge_list")
files = [x for x in top if int(x.get("media_type", 99)) != 99]
folders = [x for x in top if int(x.get("media_type", 99)) == 99]
print(f"根目录: {len(folders)} 个文件夹, {len(files)} 个文件")
print("\n=== 根目录文件夹 ===")
for f in folders: print(f"  {f['title']}  (id={f['media_id'][-8:]})")
print("\n=== 根目录文件(前40) ===")
for f in files[:40]: print("  ", f["title"])
