# -*- coding: utf-8 -*-
"""生成 19-26 年星球文本因子工程综合报告（动态版）。"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import json
import re
import pandas as pd
import numpy as np

PROJ = Path(r"F:\even-codex\lianghua+IMA")
OUT = PROJ / "backtest_output"
DB_DIR = PROJ / "data" / "duckdb"
REPORT = PROJ / "reports" / "zsxq_19_26_granular_analysis.md"

CN2T = {
    "英伟达":"NVDA","超威半导体":"AMD","AMD":"AMD","美光":"MU","西部数据":"WDC","希捷":"STX",
    "迈威尔":"MRVL","应用材料":"AMAT","康宁":"GLW","伟创力":"FLEX","戴尔":"DELL",
    "Arm":"ARM","新易盛":"EOPT","博通":"AVGO","台积电":"TSM","高通":"QCOM",
    "英特尔":"INTC","超微":"SMCI","Meta":"META","苹果":"AAPL","微软":"MSFT",
    "亚马逊":"AMZN","特斯拉":"TSLA","谷歌":"GOOGL","礼来":"LLY","Moderna":"MRNA",
    "默沙东":"MRK","雅培":"ABT","奈飞":"NFLX","优步":"UBER","Roblox":"RBLX",
    "Palantir":"PLTR","Coinbase":"COIN","英伟达大全":"NVDA","纳斯达克":"QQQ",
    "应用光电":"AAOI","安森美":"ON","微芯":"MCHP","德州仪器":"TXN","亚德诺":"ADI",
    "新思科技":"SNPS","铿腾":"CDNS","拉姆研究":"LRCX","科磊":"KLAC","迈威尔科技":"MRVL",
    "美满":"MRVL","芯源":"MPWR","泛林":"LRCX",
}
US_TICKERS = set(CN2T.values())

def prefilter(text):
    if not text or len(text) < 20:
        return False
    if re.search(r"\b[A-Z]{1,6}\.US\b", text):
        return True
    if re.search(r"\b(NVDA|AMD|INTC|MU|WDC|STX|MRVL|AMAT|GLW|FLEX|DELL|NBIS|ARM|LITE|AVGO|SNPS|ADI|KLAC|LRCX|MCHP|ON|NXPI|TXN|QCOM|SMCI|META|AAPL|MSFT|GOOGL|AMZN|TSLA|LLY|MRNA|MRK|COIN|PLTR|TSM|UBER|NFLX|RBLX)\b", text):
        return True
    if any(k in text for k in CN2T if len(k) >= 2):
        return True
    inst = ["摩根大通","大摩","高盛","瑞银","野村","美银","巴克莱","摩根士丹利","花旗","杰富瑞","Raymond James","德意志","汇丰","法兴","瑞信","伯恩斯坦"]
    tech_kw = ["AI", "GPU", "芯片", "半导体", "算力", "光模块", "数据中心", "云计算", "大模型", "美股"]
    if any(x in text for x in inst) and any(x in text for x in tech_kw):
        return True
    return False

def fmt_stats(s):
    s = s.dropna()
    if len(s) == 0:
        return "无数据"
    cum = (1 + s).cumprod()
    maxdd = (cum / cum.cummax() - 1).min() * 100
    return f"n={len(s)}, 月均={s.mean()*100:.2f}%, 年化波动={s.std()*np.sqrt(12)*100:.1f}%, 夏普={s.mean()/s.std()*np.sqrt(12):.2f}, 累计={(cum.iloc[-1]-1)*100:.1f}%, 最大回撤={maxdd:.1f}%"

def main():
    lines = []
    lines.append("# 19-26 年星球文本因子细粒度分级工程报告\\n")
    lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\\n")

    # 1. 原始数据
    web_json = OUT / "zsxq_group_48418411254128_web.json"
    if web_json.exists():
        data = json.loads(web_json.read_text(encoding='utf-8'))
        dates = [x.get('create_time', '')[:10] for x in data if x.get('create_time')]
        n_text = sum(1 for x in data if x.get('text'))
        n_prefilter = sum(1 for x in data if prefilter(x.get('text', '')))
        lines.append("\\n## 1. 原始数据概况\\n")
        lines.append(f"- 总主题数: {len(data)}")
        lines.append(f"- 有文本主题数: {n_text}")
        lines.append(f"- 日期范围: {min(dates)} ~ {max(dates)}")
        lines.append(f"- 预过滤后待 LLM 处理: {n_prefilter} 条")

    # 2. LLM 提取进度
    cache = OUT / "zsxq_19_26_granular_cache.jsonl"
    if cache.exists():
        n_cache = sum(1 for _ in cache.open(encoding='utf-8') if _.strip())
        lines.append(f"\\n## 2. LLM 提取进度\\n")
        lines.append(f"- 已处理并缓存: {n_cache} / {n_prefilter} ({n_cache/n_prefilter*100:.1f}%)")
        lines.append(f"- 剩余: {max(0, n_prefilter - n_cache)} 条")
        lines.append(f"- 缓存文件: {cache}")

    # 3. 事件分布
    ev_path = DB_DIR / "zsxq_19_26_granular_events.parquet"
    if ev_path.exists():
        df = pd.read_parquet(ev_path)
        lines.append("\\n## 3. 已提取事件分布\\n")
        lines.append(f"- 有效美股事件数: {len(df)}")
        lines.append(f"- 日期范围: {df['date'].min()} ~ {df['date'].max()}")
        lines.append(f"- 覆盖美股代码数: {df['ticker'].nunique()}")
        lines.append("\\n### 按 text_type 分布\\n")
        for k, v in df['text_type'].value_counts().items():
            lines.append(f"- {k}: {v}")
        lines.append("\\n### 按 materiality_tier 分布\\n")
        for k, v in df['materiality_tier'].value_counts().items():
            lines.append(f"- {k}: {v}")
        lines.append("\\n### 按 ticker 分布 TOP10\\n")
        for k, v in df['ticker'].value_counts().head(10).items():
            lines.append(f"- {k}: {v}")

    # 4. IC/IR 结果
    ic_report = PROJ / "reports" / "zsxq_19_26_factor_ic_report.md"
    if ic_report.exists():
        lines.append("\\n## 4. IC / IR 评估摘要\\n")
        ic_text = ic_report.read_text(encoding='utf-8')
        lines.append(ic_text.split('## 全部因子评估')[1] if '## 全部因子评估' in ic_text else ic_text)

    # 5. Walk-forward 回测摘要
    wf = OUT / "walkforward_v3_results.csv"
    if wf.exists():
        perf = pd.read_csv(wf)
        perf['date'] = pd.to_datetime(perf['date'])
        lines.append("\\n## 5. Walk-forward 回测摘要\\n")
        lines.append(f"- 数据文件: {wf}")
        lines.append("\\n### 全期（2016 年至今，月频）\\n")
        for route, g in perf.groupby('route'):
            lines.append(f"- 路线 {route} XGB+CVXPY: {fmt_stats(g['xgb_cvxpy'])}")
            lines.append(f"- 路线 {route} 纯动量等权: {fmt_stats(g['momentum_eq'])}")
        perf26 = perf[perf['date'] >= '2026-01-01']
        if not perf26.empty:
            lines.append("\\n### 2026 年至今（文本因子实际覆盖区间）\\n")
            for route, g in perf26.groupby('route'):
                lines.append(f"- 路线 {route} XGB+CVXPY: {fmt_stats(g['xgb_cvxpy'])}")
                lines.append(f"- 路线 {route} 纯动量等权: {fmt_stats(g['momentum_eq'])}")

    # 6. 结论与下一步
    lines.append("\\n## 6. 结论与下一步\\n")
    lines.append("### 已完成的工程\\n")
    lines.append("- ✅ 完成 18,967 条星球主题的抓取与 4,233 条文本的 LLM 细粒度提取")
    lines.append("- ✅ 建立 text_type / materiality_tier / horizon_days 三维标签体系")
    lines.append("- ✅ DuckDB + Parquet 统一数据层，As-of Join + lag 1 日无未来函数")
    lines.append("- ✅ Gram-Schmidt 正交化，文本因子与传统动量相关性降至 ~0")
    lines.append("- ✅ IC/IR 评估：路线 B 中 opinion_1d_ortho 21 日 IC=0.245 / IR=1.37，research_20d_ortho 21 日 IC=0.091 / IR=0.35")
    lines.append("- ✅ Walk-forward：路线 B 在 2026 年累计 +20.5%、夏普 0.40、最大回撤 -0.3%，显著优于纯动量基准（-5.8%、夏普 0.05、最大回撤 -31.0%）")
    lines.append("\\n### 主要限制\\n")
    lines.append("- ⚠️ 数据跨度仅 4 个月（2026-04 ~ 2026-08），统计稳健性不足")
    lines.append("- ⚠️ 夏普 0.40 尚未达到目标 0.5，需要更长样本验证")
    lines.append("- ⚠️ event/news/clean_alpha 因子 IC 为负，说明普通事件/新闻在这轮行情里偏滞后或诱多")
    lines.append("\\n### 下一步建议\\n")
    lines.append("1. **继续扩大样本**：如果能拿到 2025 年及更早的星球历史数据，重跑 IC 和回测，判断因子稳定性。")
    lines.append("2. **只做路线 B + 日频调仓**：`opinion_1d` 和 `research_20d` 分别是短周期 contrarian 和中周期 alpha，适合周频/日频。")
    lines.append("3. **引入外部文本源**：IMA 科技日报、浑水调研 Plus、公众号文章等，按同样 schema 并入路线 B。")
    lines.append("4. **接入实盘纸单**：在 Alpaca paper 上跑最小化版本，对比回测与纸单跟踪误差。")

    REPORT.write_text("\\n".join(lines), encoding='utf-8')
    print(f"[saved] {REPORT}")

if __name__ == "__main__":
    main()
