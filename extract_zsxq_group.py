# -*- coding: utf-8 -*-
"""分页拉取知识星球 短评&信息 group 的全部 topics_brief，保存为 JSON。
遇到创建时间早于 STOP_DATE 时停止；每页追加保存，避免中断丢失。
"""
import json, subprocess, sys, time
from pathlib import Path

GROUP_ID = "15552822254242"
STOP_DATE = "2025-01-01T00:00:00.000+0800"
OUT = Path(r"F:\even-codex\lianghua+IMA\backtest_output") / f"zsxq_group_{GROUP_ID}.json"
CLI = r"C:\Users\ASUS\AppData\Roaming\npm\zsxq-cli.cmd"

def fetch(end_time=""):
    params = {"group_id": GROUP_ID, "limit": 30}
    if end_time:
        params["end_time"] = end_time
    cmd = [CLI, "api", "call", "get_group_topics", "--params", json.dumps(params, ensure_ascii=False)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return json.loads(r.stdout)

def save(all_topics):
    OUT.write_text(json.dumps(all_topics, ensure_ascii=False, indent=2), encoding="utf-8")

all_topics = []
if OUT.exists():
    all_topics = json.loads(OUT.read_text(encoding="utf-8"))
    print(f"loaded existing {len(all_topics)} topics")

seen = {t["topic_id"] for t in all_topics}
end_time = ""
page = 0
stopped = False
while True:
    page += 1
    resp = fetch(end_time)
    topics = resp.get("topics_brief", [])
    if not topics:
        break
    new = [t for t in topics if t["topic_id"] not in seen]
    if not new:
        # 整页都重复，说明卡在同一时间戳，跳过这页
        end_time = topics[-1]["create_time"]
        if page > 2000:
            break
        continue
    all_topics.extend(new)
    seen.update(t["topic_id"] for t in new)
    print(f"page {page}: +{len(new)} new topics, total={len(all_topics)}, oldest={topics[-1]['create_time']}")
    save(all_topics)
    # 检查是否到达停止日期
    if topics[-1]["create_time"] <= STOP_DATE:
        print("reached stop date")
        break
    if not resp.get("has_more"):
        break
    end_time = topics[-1]["create_time"]
    time.sleep(0.2)

print(f"saved {OUT}: {len(all_topics)} topics")
