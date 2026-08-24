# -*- coding: utf-8 -*-
"""本地一键同步 IMA 结果表到仓库 data/ima + push。
用法：
  python sync_local_ima.py           # 只把本地产出的因子/情感表复制到 data/ima（不 push）
  python sync_local_ima.py --push    # 复制 + commit + push 到 GitHub

注意：原始 ima 摘要/论文停留在本机 us-stock-data，不进公网仓库。
复制的是提取后的结果表（词频因子、LLM 情绪信号、目标名单）。
"""
import sys, subprocess
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WS = Path(__file__).resolve().parent
SRC = WS / "backtest_output"
DST = WS / "data" / "ima"
FILES = ["kb_abstract_factors.csv", "kb_llm_sentiment.csv", "ima_final_top10.csv"]
PUSH = "--push" in sys.argv

copied = []
for name in FILES:
    s = SRC / name
    if not s.exists():
        print(f"[跳过] 本地没有 {name}")
        continue
    d = DST / name
    d.write_bytes(s.read_bytes())
    copied.append(name)
print("已复制到 data/ima:", copied or "(无)")

if PUSH and copied:
    for name in copied:
        subprocess.run(["git", "add", f"data/ima/{name}"], cwd=str(WS), check=True)
    r = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=str(WS))
    if r.returncode != 0:
        subprocess.run(["git", "commit", "-m", "chore(ima): sync local IMA factor tables"], cwd=str(WS), check=True)
        subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=str(WS), check=True)
        subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=str(WS), check=True)
        print("已 commit + push")
    else:
        print("无变化，未 push")
