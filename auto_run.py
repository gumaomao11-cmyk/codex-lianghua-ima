# -*- coding: utf-8 -*-
"""
总编排器：把原来要手动做的全部接进来，跑完把结果整理成邮件发到你的 QQ 邮箱。
行为：
  每天必做：跑 paper_tracker
  每周日  ：跑 shadow_compare
  月末     ：跑 current_holdings + 准备调仓计划（默认 dry-run；若 ALPACA_AUTO_EXECUTE=1 则真下）
  季末     ：跑 walkforward + 成本敏感度
"""
import os, subprocess, sys
from datetime import date, timedelta
from pathlib import Path
import mailer  # 同目录

WS = Path(__file__).resolve().parent
LOG = WS / "logs"; LOG.mkdir(exist_ok=True)
OUT = WS / "backtest_output"; OUT.mkdir(exist_ok=True)

def run(label, *args, timeout=600):
    """跑子脚本，捕获输出"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"; env["PYTHONIOUTF8"] = "1"
    full = [sys.executable, *map(str, args)]
    log_path = LOG / f"auto_{label}.log"
    try:
        r = subprocess.run(full, capture_output=True, text=True, encoding="utf-8",
                           env=env, cwd=str(WS), timeout=timeout)
        out = r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr else "")
        log_path.write_text(out, encoding="utf-8")
        return out, r.returncode
    except subprocess.TimeoutExpired:
        msg = f"!! 超时 {timeout}s"
        log_path.write_text(msg, encoding="utf-8")
        return msg, -1
    except Exception as e:
        msg = f"!! 异常: {e}"
        log_path.write_text(msg, encoding="utf-8")
        return msg, -1

def is_last_business_day():
    t = date.today()
    nxt = (t.replace(day=28) + timedelta(days=4)).replace(day=1)  # 下个月第一天
    last = nxt - timedelta(days=1)  # 本月最后一天
    while last.weekday() >= 5:       # 周末往前推
        last -= timedelta(days=1)
    return t == last

def is_quarter_end():
    return date.today().month in (3, 6, 9, 12) and is_last_business_day()

def is_sunday():
    return date.today().weekday() == 6

def main():
    today = date.today()
    sections = []   # (title, content)
    def add(title, body):
        sections.append((title, body.strip() if isinstance(body, str) else str(body)))

    # === 每日：paper_tracker ===
    out, code = run("paper", WS / "paper_tracker.py")
    add("【每日 paper 日报】", out)

    # === 交易日（美股收盘后）：软件级止盈止损 ===
    if today.weekday() < 5:
        out_tp, _ = run("tpsl", WS / "manage_orders.py", "--tpsl", "--execute", "--tp", "0", "--sl", "0.30")
        add("【止盈止损（自动执行，兼容碎股）】", out_tp)

    # === 周末：shadow_compare ===
    if is_sunday():
        out, _ = run("shadow", WS / "shadow_compare.py")
        add("【每周影子策略对比】", out)
        wout, _ = run("weekly_strategy", WS / "weekly_strategy.py")
        add("【周频动量策略（影子，和月频同时跑）】", wout)

    # === 月末：调仓 ===
    auto_exe = os.environ.get("ALPACA_AUTO_EXECUTE") == "1"
    if is_last_business_day():
        out, _ = run("holdings", WS / "current_holdings.py")
        add("【月末新持仓清单】", out)
        try:
            bout, bcode = run("backtest_current", WS / "current_backtest_report.py")
            add("【用最新数据复算当前策略回测】", bout)
        except Exception as e:
            add("【用最新数据复算当前策略回测】", f"生成失败: {e}")
        # 关键解读：板块分散版 top10（相关性聚类，最多3只/板块）——只出方案，不自动下单
        try:
            dout, _ = run("diversified_holdings", WS / "diversified_holdings.py")
            add("【下轮调仓建议：板块分散版 top10（相关性聚类版，max 3/板块）】", dout)
            dplan = OUT / f"rebalance_plan_{date.today():%Y%m%d}_div.csv"
            pout2, _ = run("plan_div", WS / "plan_rebalance.py", "--csv", str(OUT / "current_holdings_6m_skip1_top10_div.csv"), "--budget", "20000", "--out", str(dplan))
            add("【板块分散版分批限价执行计划】", pout2)
        except Exception as e:
            add("【板块分散版 top10】", f"生成失败: {e}")
        if auto_exe:
            out2, _ = run("rebalance_exec", WS / "alpaca_buy.py", "--rebalance", "--execute")
            add("【月末调仓（已自动执行）】", out2)
        else:
            out2, _ = run("rebalance_dry", WS / "alpaca_buy.py", "--rebalance")
            add("【月末调仓计划（dry-run，未自动执行）】", out2)
            sections[-1] = (sections[-1][0],
                sections[-1][1] + "\n\n> 若要本月自动执行：把环境变量 ALPACA_AUTO_EXECUTE=1 设上后重跑一次。")
            # 分批限价执行计划（避开买在局部高点）
            try:
                pout, _ = run("plan", WS / "plan_rebalance.py", "--csv", str(OUT / "current_holdings_6m_skip1_accel_top10.csv"))
                add("【分批限价执行计划（按SOP执行，防追高）】", pout)
            except Exception as e:
                add("【分批限价执行计划】", f"生成失败: {e}")

    # === 周日：PDF 周报 ===
    if is_sunday():
        out, _ = run("weekly_pdf", WS / "weekly_report_pdf.py")
        add("【周日 PDF 周报生成】", out)
    if is_quarter_end():
        out, _ = run("walkforward", WS / "walkforward_v6.py")
        add("【季度滚动验证】", out)
        out2, _ = run("cost", WS / "cost_sensitivity.py")
        add("【季度成本敏感度】", out2)

    # === 组装邮件 ===
    body = "\n\n" + "="*60 + "\n".join(f"\n{title}\n{'-'*len(title)}\n{c}\n" for title, c in sections)
    tags = []
    if is_sunday(): tags.append("周报")
    if is_last_business_day(): tags.append("月报")
    if is_quarter_end(): tags.append("季报")
    if not tags: tags.append("日报")
    subject = f"[策略 {'+'.join(tags)}] {today}"

    # 附件
    attachments = []
    cand = OUT / "current_holdings_6m_skip1_accel_top10.csv"
    if cand.exists(): attachments.append(cand)
    log_csv = OUT / "paper_log.csv"
    if log_csv.exists(): attachments.append(log_csv)
    if is_last_business_day():
        bmd = OUT / "current_backtest_report.md"
        if bmd.exists(): attachments.append(bmd)
    wf = OUT / "current_holdings_6m_skip1_top10_weekly.csv"
    if wf.exists(): attachments.append(wf)
    # 周日附加 PDF 周报
    if is_sunday():
        pdf = OUT / f"weekly_report_{today.strftime('%Y%m%d')}.pdf"
        if pdf.exists(): attachments.append(pdf)

    print(subject)
    print(body[:1500], "...")
    mailer.send(subject, body, attachments=attachments or None)
    print("Done.")

if __name__ == "__main__":
    main()

