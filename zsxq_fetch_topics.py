# -*- coding: utf-8 -*-
"""抓取知识星球主题到本地（不进 GitHub）。"""
import sys, json, subprocess, datetime, shutil, os
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 定位 zsxq-cli（Windows npm 全局不是默认 PATH）
_CLI = shutil.which("zsxq-cli") or str(Path.home()/ "AppData/Roaming/npm/zsxq-cli.cmd")
if not Path(_CLI).exists():
    print("ERR: 找不到 zsxq-cli", _CLI); sys.exit(1)
print("zsxq-cli:", _CLI)

OUT = Path(r"F:\even-codex\lianghua+IMA\backtest_output")
OUT.mkdir(parents=True, exist_ok=True)
GROUPS = [("28512858211281","浑水调研Plus"), ("15552822254242","短评&信息")]
CUTOFF = "2025-05-01T00:00:00.000+0800"
os.environ["PYTHONIOENCODING"]="utf-8"

def fetch_group(gid, name):
    items = {}; seen = set(); end = None; pages = 0
    while pages < 100:
        cmd = [_CLI,"group","+topics","--group-id",gid,"--json","--limit","30"]
        if end: cmd += ["--end-time", end]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=120)
        if r.returncode != 0:
            print(f"[{name}] ERR {r.returncode}: {r.stderr[:300]}"); break
        try: j = json.loads(r.stdout)
        except Exception as e:
            print(f"[{name}] JSON err page {pages}: {e}\nstdout500: {r.stdout[:500]}"); break
        lst = j.get("topics_brief") or []
        if not lst:
            print(f"[{name}] no rows page {pages}"); break
        oldest = None; added = 0
        for it in lst:
            tid = it.get("topic_id"); ct = it.get("create_time","")
            if tid in seen: continue
            seen.add(tid); items[tid] = it; added += 1
            if not oldest or ct < oldest: oldest = ct
        print(f"[{name}] page {pages}: got {len(lst)} added {added} total {len(items)} oldest {oldest} has_more {j.get('has_more')}")
        pages += 1
        if not j.get("has_more"): break
        end = j.get("next_end_time")
        if not end: break
        if oldest and oldest < CUTOFF:
            print(f"[{name}] reached cutoff {oldest}"); break
        if len(items) > 20000:
            print(f"[{name}] hit cap 20000"); break
    outp = OUT / f"zsxq_{name}.json"
    outp.write_text(json.dumps(list(items.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}] saved {outp} items={len(items)}")
    return list(items.values())

alldata = []
for gid, name in GROUPS:
    alldata += fetch_group(gid, name)
(out2 := OUT/"zsxq_topics.json").write_text(json.dumps(alldata, ensure_ascii=False, indent=2), encoding="utf-8")
print("TOTAL", len(alldata), "saved", out2)
