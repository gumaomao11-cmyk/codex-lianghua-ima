# -*- coding: utf-8 -*-
"""
周报 PDF：每周日跑，把这一周的 paper 表现生成可分享的 PDF，附到邮件里。
依赖：reportlab + matplotlib + pandas
用法：python weekly_report_pdf.py
输出：backtest_output/weekly_report_YYYYMMDD.pdf
"""
import os, sys
from pathlib import Path
from datetime import date, timedelta
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，server 友好
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd, numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# --- 路径 ---
WS   = Path(__file__).resolve().parent
OUT  = WS / "backtest_output"
LOG  = OUT / "paper_log.csv"
STATE= OUT / "paper_state.json"
CHART= OUT / "_weekly_chart_tmp.png"
PDF  = OUT / f"weekly_report_{date.today().strftime('%Y%m%d')}.pdf"

# --- 字体配置（CJK）---
def _setup_fonts():
    # 1) matplotlib 找中文字体
    cjk_kw = ["Microsoft YaHei", "SimHei", "SimSun", "NSimSun", "KaiTi", "Noto Sans CJK SC", "Noto Sans CJK", "Noto Sans SC", "Source Han Sans CN", "Source Han Sans SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
    all_names = {f.name for f in fm.fontManager.ttflist}
    hit = next((n for n in all_names for k in cjk_kw if k in n), None)
    if hit:
        plt.rcParams["font.sans-serif"] = [hit, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return hit
    return None

CJK_FONT = _setup_fonts()

# 2) reportlab CJK 字体（不依赖系统字体）
try:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    RL_CJK = "STSong-Light"
except Exception:
    RL_CJK = "Helvetica"

# --- 读数据 ---
if not LOG.exists():
    print(f"[weekly_pdf] 找不到 {LOG}，先生成几天的 paper_log 再跑")
    sys.exit(0)
df = pd.read_csv(LOG)
if df.empty or len(df) < 1:
    print("[weekly_pdf] paper_log.csv 还没数据，跳过")
    sys.exit(0)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 最近 7 天（不足就全取）
last7 = df.tail(7).copy()
if len(df) >= 2:
    prev_eq = float(df.iloc[-2]["total_equity"]) if len(df) >= 2 else float(df.iloc[-1]["total_equity"])
else:
    prev_eq = float(df.iloc[-1]["total_equity"])
cur_eq   = float(last7.iloc[-1]["total_equity"])
start_eq = float(last7.iloc[0]["total_equity"]) if len(last7) > 0 else cur_eq

start_state_eq = float(pd.read_json(STATE)["start_equity"].iloc[0]) if STATE.exists() else 20000.0

def pct(x):  return f"{x*100:+.2f}%"

# --- 画图：净值曲线（策略 vs SPY vs QQQ）---
fig, ax = plt.subplots(figsize=(7, 3.2), dpi=140)
if "strategy_nav" in df.columns and "spy" in df.columns:
    # 归一化到 100
    base_s = df["strategy_nav"].iloc[0] or 1
    base_q = df["spy"].iloc[0] or 1
    base_k = df["qqq"].iloc[0] or 1
    ax.plot(df["date"], df["strategy_nav"]/base_s*100, label="Strategy", linewidth=2, color="#1f77b4")
    ax.plot(df["date"], df["spy"]/base_q*100, label="SPY", linewidth=1.5, color="#2ca02c")
    ax.plot(df["date"], df["qqq"]/base_k*100, label="QQQ", linewidth=1.5, color="#ff7f0e")
    ax.axhline(100, color="grey", linewidth=0.5, linestyle="--")
    ax.set_title("累计净值（起点=100）", fontsize=12)
    ax.set_ylabel("净值")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(CHART, dpi=140, bbox_inches="tight")
plt.close(fig)

# --- 组装 PDF ---
styles = getSampleStyleSheet()
zh = ParagraphStyle("zh", parent=styles["BodyText"], fontName=RL_CJK, fontSize=10, leading=14)
title = ParagraphStyle("title", parent=styles["Title"], fontName=RL_CJK, fontSize=18, leading=22, spaceAfter=8)
h2    = ParagraphStyle("h2", parent=styles["Heading2"], fontName=RL_CJK, fontSize=13, leading=16, spaceBefore=10, spaceAfter=6)

doc = SimpleDocTemplate(str(PDF), pagesize=A4,
                        leftMargin=1.8*cm, rightMargin=1.8*cm,
                        topMargin=1.5*cm, bottomMargin=1.5*cm)
story = []

# 标题
period = f"{last7['date'].min().strftime('%Y-%m-%d')} ~ {last7['date'].max().strftime('%Y-%m-%d')}" if len(last7) else "—"
story.append(Paragraph(f"动量策略 · 周报", title))
story.append(Paragraph(f"区间：{period}　|　生成：{date.today().isoformat()}", zh))
story.append(Spacer(1, 6))

# 关键指标卡片
total_ret = (cur_eq - start_state_eq) / start_state_eq if start_state_eq else 0
week_ret  = (cur_eq - start_eq) / start_eq if start_eq and len(last7) > 1 else 0
alpha_spy = float(last7["alpha_spy"].iloc[-1]) if "alpha_spy" in last7.columns and len(last7) else 0
alpha_qqq = float(last7["alpha_qqq"].iloc[-1]) if "alpha_qqq" in last7.columns and len(last7) else 0

metrics_data = [
    ["当前权益", f"${cur_eq:,.2f}", "总收益", pct(total_ret)],
    ["本周涨跌", pct(week_ret), "vs SPY", pct(alpha_spy)],
    ["现金",     f"${float(last7['cash'].iloc[-1]):,.2f}", "vs QQQ", pct(alpha_qqq)],
    ["持仓数",   f"{int(last7['n_strategy_pos'].iloc[-1])}", "起点权益", f"${start_state_eq:,.2f}"],
]
t = Table(metrics_data, colWidths=[2.6*cm, 3.2*cm, 2.6*cm, 3.2*cm])
t.setStyle(TableStyle([
    ("FONTNAME", (0,0), (-1,-1), RL_CJK),
    ("FONTSIZE", (0,0), (-1,-1), 10),
    ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f0f4f8")),
    ("BACKGROUND", (2,0), (2,-1), colors.HexColor("#f0f4f8")),
    ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ("ALIGN", (1,0), (1,-1), "RIGHT"),
    ("ALIGN", (3,0), (3,-1), "RIGHT"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("LEFTPADDING", (0,0), (-1,-1), 6),
    ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
]))
story.append(t)
story.append(Spacer(1, 10))

# 净值曲线
if CHART.exists():
    story.append(Paragraph("累计净值曲线", h2))
    story.append(Image(str(CHART), width=16*cm, height=7.3*cm))
    story.append(Spacer(1, 8))

# 每日明细表
story.append(Paragraph("最近交易日明细", h2))
detail_cols = ["date", "total_equity", "cash", "strat_pct", "spy_pct", "alpha_spy"]
detail_cols = [c for c in detail_cols if c in df.columns]
disp = last7[detail_cols].copy()
if "date" in disp.columns:
    disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
for c in disp.columns:
    if c in ("strat_pct", "spy_pct", "alpha_spy"):
        disp[c] = disp[c].astype(float).map(lambda x: f"{x*100:+.2f}%")
    elif c in ("total_equity", "cash"):
        disp[c] = disp[c].astype(float).map(lambda x: f"${x:,.2f}")
header = [c for c in detail_cols]
rows = [header] + disp.values.tolist()
t2 = Table(rows, colWidths=[3.2*cm, 3.4*cm, 3.2*cm, 2.6*cm, 2.6*cm, 2.6*cm])
t2.setStyle(TableStyle([
    ("FONTNAME", (0,0), (-1,-1), RL_CJK),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f77b4")),
    ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
    ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
    ("ALIGN", (1,1), (-1,-1), "RIGHT"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
]))
story.append(t2)
story.append(Spacer(1, 10))

# 持仓清单
target_csv = OUT / "current_holdings_6m_skip1_accel_top10.csv"
if target_csv.exists():
    story.append(PageBreak())
    story.append(Paragraph("当前持仓清单（top10）", h2))
    h = pd.read_csv(target_csv)
    if "ticker" in h.columns and "weight" in h.columns:
        h = h[["ticker", "weight"]].copy()
        h["weight"] = h["weight"].astype(float).map(lambda x: f"{x*100:.1f}%")
    rows = [list(h.columns)] + h.values.tolist()
    t3 = Table(rows, colWidths=[4*cm, 4*cm])
    t3.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), RL_CJK),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2ca02c")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
    ]))
    story.append(t3)

# 风险提示
story.append(Spacer(1, 14))
story.append(Paragraph("风险提示", h2))
story.append(Paragraph(
    "本报告为算法自动生成的模拟盘跟踪数据，<b>不构成投资建议</b>。"
    "动量策略在大波段中盈利但会承受较长时间的小幅回撤，3 个月内若回撤超过 -40% 请暂停调仓并参考 backtest_output/manual_checklist.md。",
    zh))

doc.build(story)

# 清理临时图
try: CHART.unlink()
except: pass

print(f"[weekly_pdf] 写出: {PDF}  ({PDF.stat().st_size//1024} KB)")
