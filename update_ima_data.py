# -*- coding: utf-8 -*-
"""一键更新 IMA 数据（本机执行）：
  1) 用本机最新 ima_all_meta.json 重建词频因子 kb_abstract_factors.csv
  2) 复制结果表到仓库 data/ima
  3) 重选目标名单 data/ima/ima_final_top10.csv（动量+IMA win60/λ1.2）
  4) 可选 --push：commit + push 到 GitHub

用法：
  python update_ima_data.py           # 本地重建 + 更新 data/ima（不 push）
  python update_ima_data.py --push    # 额外 commit + push
前置：请先在 ima 客户端刷新知识库，并确保本机 us-stock-data/ima_all_meta.json 是最新合并结果。
"""
import os, sys, subprocess, shutil
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WS = Path(__file__).resolve().parent
SRC = WS / "backtest_output"
DST = WS / "data" / "ima"
PY  = sys.executable
PUSH = "--push" in sys.argv

def step(msg): print(f"\n== {msg} ==", flush=True)

step("1/4 重建词频因子 kb_abstract_factors.csv")
subprocess.run([PY, "kb_abstract_factor.py"], cwd=str(WS), check=True)

step("2/4 复制结果表到 data/ima")
s = SRC / "kb_abstract_factors.csv"
if s.exists():
    d = DST / "kb_abstract_factors.csv"; d.write_bytes(s.read_bytes())
    print(" 已复制 →", d)
else:
    print(" [警告] 本地没有 kb_abstract_factors.csv，跳过"); sys.exit(1)
for extra in ["kb_llm_sentiment.csv", "ima_final_top10.csv"]:
    se = SRC / extra
    if se.exists():
        de = DST / extra; de.write_bytes(se.read_bytes()); print(" 已复制 →", de)

step("3/4 重选目标名单 select_ima_final.py")
subprocess.run([PY, "select_ima_final.py"], cwd=str(WS), check=True)

step("4/4 汇总")
print("更新完成。data/ima 现包含：")
for f in sorted(DST.iterdir()):
    print("  ", f.name, f.stat().st_size)

if PUSH:
    step("push 到 GitHub")
    subprocess.run(["git", "add", "data/ima"], cwd=str(WS), check=True)
    r = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=str(WS))
    if r.returncode != 0:
        subprocess.run(["git", "commit", "-m", "chore(ima): update IMA data"], cwd=str(WS), check=True)
        subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=str(WS), check=True)
        subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=str(WS), check=True)
        print("已 commit + push")
    else:
        print("无数据变化，未 push")
