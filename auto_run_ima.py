# -*- coding: utf-8 -*-
"""
IMA 策略每日编排：跑 paper_tracker_ima 并把日报邮件发到 QQ 邮箱。
  python auto_run_ima.py
行为：
  - 每天跑 paper_tracker_ima.py 生成日报
  - 把日报正文 + ima_final_top10.csv 发到 869357594@qq.com
"""
import os, subprocess, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from datetime import date
from pathlib import Path
import mailer

WS = Path(__file__).resolve().parent
OUT = WS / "backtest_output"
LOG = WS / "logs"; LOG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

def run(label, *args):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"; env["PYTHONIOUTF8"] = "1"
    full = [sys.executable, *map(str, args)]
    log_path = LOG / f"auto_ima_{label}.log"
    try:
        r = subprocess.run(full, capture_output=True, text=True, encoding="utf-8",
                           env=env, cwd=str(WS), timeout=300)
        out = r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr else "")
        log_path.write_text(out, encoding="utf-8")
        return out, r.returncode
    except Exception as e:
        msg = f"!! 异常: {e}"
        log_path.write_text(msg, encoding="utf-8")
        return msg, -1

def main():
    today = date.today()
    out, code = run("paper", WS / "paper_tracker_ima.py")
    subject = f"[IMA策略 日报] {today}"
    body = out if code == 0 else ("IMA 日报生成失败（详见日志）。\n\n" + out)
    # 附件
    att = []
    csv = OUT / "paper_log_ima.csv"
    if csv.exists(): att.append(csv)
    tgt = OUT / "ima_final_top10.csv"
    if tgt.exists(): att.append(tgt)
    print(subject)
    print(body[:1200], "...")
    ok = mailer.send(subject, body, attachments=att or None)
    print("Mail delivered:", ok)

if __name__ == "__main__":
    main()

