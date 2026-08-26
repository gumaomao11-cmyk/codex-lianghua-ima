# -*- coding: utf-8 -*-
"""明天配额恢复后运行：补齐浑水调研库的词频/高亮语料。
- 遍历共享库根目录 + 电话会总结 + 上市公司模型 + AI总结(markdown)
- 对美股候选代码做 search_knowledge，抓 highlight_content 片段
- 遇到 220021 配额上限自动停止
输出: data/ima_sweep_*.json
"""
import json, re, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import ima_openapi as io

KB = "RJxmvo3aEkYpjzFlq7Z_wcU8GFQQXidQx5h7KETo-c4="
OUT = Path("data")
CAND_TICKERS = ["NVDA","AMD","INTC","MU","WDC","STX","MRVL","AMAT","GLW","FLEX",
                "DELL","NBIS","ARM","LITE","AVGO","SNPS","ADI","KLAC","LRCX",
                "MCHP","ON","NXPI","TXN","QCOM","SMCI","META","AAPL","MSFT",
                "GOOGL","GOOG","AMZN","TSLA","LLY","MRNA","MRK","COIN","PLTR"]
QUOTA_MSG = "资料获取次数已达上限"

def safe_call(ep, body):
    r = io.api(ep, body)
    if isinstance(r, dict) and r.get("code") == 220021:
        print("[QUOTA] 已达今日上限，停止。"); return None
    return r

def page_list(folder_id, name):
    rows, cursor, guard = [], "", 0
    while guard < 2000:
        r = safe_call("get_knowledge_list",
                      {"knowledge_base_id": KB, "folder_id": folder_id, "cursor": cursor, "limit": 50})
        if r is None: break
        d = r.get("data", {})
        rows += d.get("knowledge_list", []) or []
        if d.get("is_end"): break
        cursor = (d.get("next_cursor") or "")
        if not cursor: break
        guard += 1; time.sleep(0.12)
    print(f"[folder] {name} -> {len(rows)} items")
    return rows

def search_for(ticker):
    r = safe_call("search_knowledge", {"knowledge_base_id": KB, "query": ticker, "cursor": ""})
    if r is None: return None
    return r.get("data", {}).get("info_list", []) or []

def main():
    OUT.mkdir(exist_ok=True)
    # 1) 外资研报已导过，这里补根目录/AI总结/电话会/模型
    allrows = []
    allrows += page_list("folder_7495389305403551", "外资研报(增量)")
    allrows += page_list("folder_7495387891903752", "电话会总结(日更)")
    allrows += page_list("folder_7495387745103155", "上市公司模型")
    # 根目录顶层(含AI总结markdown)
    allrows += page_list("7495378999998757", "库根目录")
    (OUT / "ima_sweep_items.json").write_text(json.dumps(allrows, ensure_ascii=False, indent=2), encoding="utf-8")
    # 2) 针对美股代码做高亮检索（受配额限制，省着打）
    hits = {}
    for i, tk in enumerate(CAND_TICKERS):
        r = search_for(tk)
        if r is None: break
        hits[tk] = r
        print(f"[search] {tk} -> {len(r)}")
        time.sleep(0.3)
    (OUT / "ima_sweep_search.json").write_text(json.dumps(hits, ensure_ascii=False, indent=2), encoding="utf-8")
    print("完成。输出:", list((OUT / "ima_sweep_items.json").resolve()), list((OUT / "ima_sweep_search.json").resolve()))

if __name__ == "__main__":
    main()
