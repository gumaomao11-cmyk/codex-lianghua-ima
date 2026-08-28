# -*- coding: utf-8 -*-
"""通过知识星球官方 API 抓取历史主题，支持从指定时间点向前翻页。

用法示例：
    python scrape_zsxq_api.py --group-id 48418411254128 --target-date 2025-01-01 --max-pages 5000
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

CWD = Path(__file__).resolve().parent
OUT_DIR = CWD / "backtest_output"
OUT_DIR.mkdir(exist_ok=True)

STORAGE_STATE = OUT_DIR / "zsxq_web_storage_state.json"
CHECKPOINT = OUT_DIR / "zsxq_api_checkpoint.json"
DEFAULT_GROUP = "48418411254128"
API_BASE = "https://api.zsxq.com"


def load_storage():
    if not STORAGE_STATE.exists():
        print(f"[error] 未找到登录状态文件: {STORAGE_STATE}")
        print("请先运行 Playwright 登录脚本或手动登录后保存 storage state。")
        sys.exit(2)
    data = json.loads(STORAGE_STATE.read_text(encoding="utf-8"))
    return {c["name"]: c["value"] for c in data.get("cookies", [])}


def load_checkpoint(group_id):
    if CHECKPOINT.exists():
        data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        if data.get("group_id") == group_id:
            data["seen_ids"] = set(data.get("seen_ids", []))
            return data
    return {"group_id": group_id, "topics": [], "seen_ids": set(), "oldest_time": None, "page": 0}


def save_checkpoint(state):
    CHECKPOINT.write_text(
        json.dumps(
            {
                "group_id": state["group_id"],
                "topics": state["topics"],
                "seen_ids": list(state["seen_ids"]),
                "oldest_time": state.get("oldest_time"),
                "page": state.get("page", 0),
                "updated_at": datetime.now().isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_final(topics, group_id, suffix="api"):
    out = OUT_DIR / f"zsxq_group_{group_id}_{suffix}.json"
    out.write_text(json.dumps(topics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out}: {len(topics)} topics")
    return out


def normalize_topic(raw):
    """把 API 返回的 topic 转成统一格式。"""
    talk = raw.get("talk") or {}
    owner = talk.get("owner") or {}
    text = talk.get("text") or ""
    if not text:
        # 有些帖子只有 title 或 files
        text = raw.get("title") or ""
    author = owner.get("name") or owner.get("alias") or ""
    images = [img.get("thumbnail") or img.get("url") for img in talk.get("images", []) if img]
    files = []
    for f in talk.get("files", []):
        url = f.get("url") or f.get("download_url")
        if url:
            files.append(url)
    return {
        "topic_id": str(raw.get("topic_id") or raw.get("topic_uid") or ""),
        "create_time": raw.get("create_time"),
        "author": author,
        "text": text.strip(),
        "images": images,
        "files": files,
        "raw_type": raw.get("type"),
    }


def fetch_page(group_id, cookies, end_time=None, count=20, retries=3):
    url = f"{API_BASE}/v2/groups/{group_id}/topics"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": "https://wx.zsxq.com/",
        "Origin": "https://wx.zsxq.com",
    }
    params = {"count": count, "scope": "all", "sort": "time"}
    if end_time:
        params["end_time"] = end_time
    for attempt in range(retries):
        resp = requests.get(url, params=params, cookies=cookies, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"[error] HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        try:
            data = resp.json()
        except Exception as e:
            print(f"[error] JSON parse failed: {e}")
            return None
        # API 偶尔会返回空 resp_data，等一下重试
        if (data.get("resp_data") or {}).get("topics") or attempt == retries - 1:
            return data
        time.sleep(2 ** attempt)
    return None


def parse_time(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-id", default=DEFAULT_GROUP)
    ap.add_argument("--target-date", default="2025-01-01", help="爬到该日期为止，格式 2025-01-01")
    ap.add_argument("--max-pages", type=int, default=10000, help="最大翻页数")
    ap.add_argument("--count", type=int, default=30, help="每页条数，最大 30")
    ap.add_argument("--sleep", type=float, default=0.5, help="每页间隔秒数")
    ap.add_argument("--merge", action="store_true", help="与已有的 web json 合并去重")
    args = ap.parse_args()

    cookies = load_storage()
    state = load_checkpoint(args.group_id)
    target_dt = datetime.strptime(args.target_date, "%Y-%m-%d")
    target_dt = target_dt.replace(tzinfo=None)

    end_time = state.get("oldest_time")
    seen = state["seen_ids"]
    topics = state["topics"]
    start_page = state.get("page", 0)

    print(f"[start] group={args.group_id} target={args.target_date} existing={len(topics)}")

    empty_streak = 0
    for page in range(start_page, args.max_pages):
        data = fetch_page(args.group_id, cookies, end_time=end_time, count=args.count)
        if data is None:
            empty_streak += 1
            if empty_streak >= 5:
                print("[warn] fetch failed 5 times in a row, save checkpoint and exit")
                break
            print(f"[warn] fetch failed ({empty_streak}/5), retry after 10s")
            time.sleep(10)
            continue

        # 有些页面会返回 error 空 resp_data
        resp_data = data.get("resp_data") or {}
        raw_topics = resp_data.get("topics") or []
        if not raw_topics:
            empty_streak += 1
            if empty_streak >= 5:
                print("[done] no more topics (5 consecutive empty pages)")
                break
            print(f"[warn] empty page ({empty_streak}/5), sleep 10s and retry")
            time.sleep(10)
            continue
        empty_streak = 0

        added = 0
        for raw in raw_topics:
            tid = str(raw.get("topic_id") or raw.get("topic_uid") or "")
            # 翻页边界去重：end_time 是上一页最后一条的 create_time，下一页首条会重复
            if tid in seen:
                continue
            seen.add(tid)
            topics.append(normalize_topic(raw))
            added += 1

        oldest = raw_topics[-1].get("create_time")
        oldest_dt = parse_time(oldest)
        state["oldest_time"] = oldest
        state["page"] = page + 1

        if page % 10 == 0 or added > 0:
            print(f"[page {page}] added={added} total={len(topics)} oldest={oldest}")
            save_checkpoint(state)

        if oldest_dt and oldest_dt.replace(tzinfo=None) < target_dt:
            print(f"[done] reached target date: {oldest}")
            break

        end_time = oldest
        time.sleep(args.sleep)

    save_checkpoint(state)

    out_topics = topics
    if args.merge:
        web_path = OUT_DIR / f"zsxq_group_{args.group_id}_web.json"
        if web_path.exists():
            web_topics = json.loads(web_path.read_text(encoding="utf-8"))
            merged = {t["topic_id"]: t for t in web_topics}
            for t in topics:
                merged[t["topic_id"]] = t
            out_topics = list(merged.values())
            out_topics.sort(key=lambda x: x.get("create_time") or "", reverse=True)
            print(f"[merge] web={len(web_topics)} + api={len(topics)} -> {len(out_topics)}")
        save_final(out_topics, args.group_id, suffix="web")
    else:
        save_final(out_topics, args.group_id, suffix="api")


if __name__ == "__main__":
    main()
