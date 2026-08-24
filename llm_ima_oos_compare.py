# -*- coding: utf-8 -*-
"""IS/OOS 对比：momentum vs momentum+IMA词频 vs momentum+LLM。
IS=前5次调仓选参，OOS=后6次验证，统一5bp成本、10只等权、月度。
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path(r"F:\even-codex\us-stock-data")
OUT  = Path(r"F:\even-codex\lianghua+IMA\backtest_output")
OUT.mkdir(parents=True, exist_ok=True)

px = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
recent = px.loc[px.index >= pd.Timestamp("2025-01-01")]
px = px.loc[:, recent.notna().sum() >= 150]
ml = px.resample("ME").last()
mom = (ml.shift(1) / ml.shift(7) - 1.0).replace([np.inf, -np.inf], np.nan)
daily_ret = px.pct_change().fillna(0.0)
cols = list(px.columns)
rebal = list(ml.truncate("2025-09-30", px.index[-1]).index)

# 因子数据
wfac = pd.read_csv(OUT/"kb_abstract_factors.csv", encoding="utf-8-sig")
wfac["pdf_date"] = pd.to_datetime(wfac["pdf_date"], errors="coerce"); wfac = wfac.dropna(subset=["pdf_date"])
lfac = pd.read_csv(OUT/"kb_llm_sentiment.csv", encoding="utf-8-sig")
lfac["pdf_date"] = pd.to_datetime(lfac["pdf_date"], errors="coerce"); lfac = lfac.dropna(subset=["pdf_date"])

def wscore(d, window):
    s = wfac[wfac.pdf_date < d]; s = s[s.pdf_date >= d - pd.Timedelta(days=window)]
    if s.empty: return pd.Series(dtype=float)
    g = s.groupby("ticker").agg(n=("n_pos","sum"), d=("n_neg","sum"), sig=("sign","sum"))
    sc = g["n"] - g["d"] + g["sig"]*0.5
    return (sc - sc.mean()) / (sc.std() if sc.std()>0 else 1.0)

def lscore(d, window, strong):
    s = lfac[lfac.pdf_date < d]; s = s[s.pdf_date >= d - pd.Timedelta(days=window)]
    if strong:
        s = s[(s["direction"]!=0) & (s["strength"]>=0.5)]
    if s.empty: return pd.Series(dtype=float)
    sc = (s["direction"].astype(float) * s["strength"].astype(float)).groupby(s["ticker"]).sum()
    return (sc - sc.mean()) / (sc.std() if sc.std()>0 else 1.0)

def monthly_rets(kind, window, lam, strong=None, cost_bps=5):
    rets = {}
    for i, d in enumerate(rebal):
        if d not in mom.index: continue
        m = mom.loc[d].dropna()
        if m.empty: continue
        if kind == "mom":
            score = m.rank(pct=True)
        else:
            z = (wscore(d, window) if kind == "ima" else lscore(d, window, strong)).reindex(m.index)
            score = m.rank(pct=True) + lam * z.fillna(0.0)
        sel = score.sort_values(ascending=False).index[:10]
        if len(sel) == 0: continue
        w = pd.Series(0.0, index=cols); w[sel] = 1.0/len(sel)
        end = rebal[i+1] if i+1 < len(rebal) else px.index[-1]
        days = px.index[(px.index > d) & (px.index <= end)]
        if len(days)==0: continue
        r = (w.reindex(cols).values * daily_ret.loc[days, :].values).sum(axis=1)
        r = r - ((w - pd.Series(0.0, index=cols)).abs().sum()/2.0) * cost_bps/10000.0
        rets[d] = float(r.sum())
    return pd.Series(rets)

def summ(s):
    s = s.dropna()
    if len(s)==0: return {"n":0,"mean":np.nan,"vol":np.nan,"sharpe":np.nan}
    if len(s)==1: return {"n":1,"mean":float(s.mean()),"vol":np.nan,"sharpe":np.nan}
    mr=float(s.mean()); v=float(s.std(ddof=0))
    return {"n":len(s),"mean":mr*100,"vol":v*100,"sharpe":float(mr/v) if v>0 else np.nan}

def rep(prefix):
    rows=[]
    for w in [30,60,90]:
        for lam in [0.3,0.7,1.2]:
            r = monthly_rets("ima", w, lam)
            is_ = summ(r.iloc[:5]); oos = summ(r.iloc[5:])
            rows.append({"kind":"ima","window":w,"lam":lam,"is_sharpe":is_["sharpe"],
                         "oos_sharpe":oos["sharpe"],"oos_mean":oos["mean"],"oos_vol":oos["vol"]})
    return pd.DataFrame(rows)

mom_ret = monthly_rets("mom", 0, 0.0)
mom_full = summ(mom_ret); mom_is = summ(mom_ret.iloc[:5]); mom_oos = summ(mom_ret.iloc[5:])

def pick_best(rows, col="is_sharpe"):
    rows = rows.replace({col: {np.nan: -999}})
    return rows.loc[rows[col].idxmax()]

ima_grid = rep("ima")
llm_rows = []
for w in [15,30,60]:
    for strong in [True,False]:
        for lam in [0.7,1.0,1.5]:
            r = monthly_rets("llm", w, lam, strong)
            is_ = summ(r.iloc[:5]); oos = summ(r.iloc[5:])
            llm_rows.append({"kind":"llm","window":w,"strong":strong,"lam":lam,
                             "is_sharpe":is_["sharpe"],"oos_sharpe":oos["sharpe"],
                             "oos_mean":oos["mean"],"oos_vol":oos["vol"]})
llm_grid = pd.DataFrame(llm_rows)

best_ima = pick_best(ima_grid); best_llm = pick_best(llm_grid)

def fmt_v(x):
    return f"{x:.2f}" if x is not None and not (isinstance(x,float) and np.isnan(x)) else "n/a"

lines = []
lines.append("# IS/OOS 对比报告：momentum vs 动量+IMA词频 vs 动量+LLM")
lines.append("")
lines.append(f"- 数据区间：{rebal[0].date()} ~ {rebal[-1].date()}")
lines.append(f"- 调仓次数：{len(rebal)}（IS=前5，OOS=后{len(rebal)-5}）")
lines.append(f"- 成本：5bp/次，10只等权，月度")
lines.append("")
lines.append("## 基准 momentum")
lines.append(f"- 全期 月均 {mom_full['mean']:.2f}% / 月波动 {mom_full['vol']:.2f}% / 月度夏普 {fmt_v(mom_full['sharpe'])}")
lines.append(f"- IS 月均 {mom_is['mean']:.2f}% / 夏普 {fmt_v(mom_is['sharpe'])}")
lines.append(f"- OOS 月均 {mom_oos['mean']:.2f}% / 月波动 {mom_oos['vol']:.2f}% / 夏普 {fmt_v(mom_oos['sharpe'])}")
lines.append("")
lines.append("## momentum + IMA 词频（按 IS 夏普选参，然后看 OOS）")
lines.append(f"- IS 最优参数：window={int(best_ima['window'])} λ={best_ima['lam']}，IS夏普 {fmt_v(best_ima['is_sharpe'])}")
lines.append(f"- OOS：月均 {best_ima['oos_mean']:.2f}% / 月波动 {best_ima['oos_vol']:.2f}% / 夏普 {fmt_v(best_ima['oos_sharpe'])}")
lines.append("")
lines.append("## momentum + LLM 情绪（按 IS 夏普选参，然后看 OOS）")
lines.append(f"- IS 最优参数：window={int(best_llm['window'])} strong={bool(best_llm['strong'])} λ={best_llm['lam']}，IS夏普 {fmt_v(best_llm['is_sharpe'])}")
lines.append(f"- OOS：月均 {best_llm['oos_mean']:.2f}% / 月波动 {best_llm['oos_vol']:.2f}% / 夏普 {fmt_v(best_llm['oos_sharpe'])}")
lines.append("")
lines.append("## 参数网格（IS 夏普排序，供核验是否过拟合）")
lines.append("### IMA 词频")
lines.append("| window | λ | IS夏普 | OOS夏普 | OOS月均% |")
lines.append("|---|---:|---:|---:|---:|")
for _,r in ima_grid.sort_values("is_sharpe",ascending=False).iterrows():
    lines.append(f"| {int(r['window'])} | {r['lam']} | {fmt_v(r['is_sharpe'])} | {fmt_v(r['oos_sharpe'])} | {fmt_v(r['oos_mean'])} |")
lines.append("### LLM 情绪")
lines.append("| window | strong | λ | IS夏普 | OOS夏普 | OOS月均% |")
lines.append("|---|---:|---:|---:|---:|---:|")
for _,r in llm_grid.sort_values("is_sharpe",ascending=False).iterrows():
    lines.append(f"| {int(r['window'])} | {int(r['strong'])} | {r['lam']} | {fmt_v(r['is_sharpe'])} | {fmt_v(r['oos_sharpe'])} | {fmt_v(r['oos_mean'])} |")
lines.append("")
lines.append("## 结论")
iu = best_ima['oos_sharpe']; lu = best_llm['oos_sharpe']; mu = mom_oos['sharpe']
g = "IMA词频" if iu >= lu else "LLM"
lines.append(f"- OOS 夏普：momentum={fmt_v(mu)}，动量+IMA={fmt_v(iu)}，动量+LLM={fmt_v(lu)}")
lines.append(f"- 若 IMA >= LLM 且高于基准，说明 IMA 词频在样本外更稳；反之说明该参数已过拟合。")
report = "\n".join(lines)
(OUT/"llm_ima_oos_compare.md").write_text(report, encoding="utf-8")
ima_grid.to_csv(OUT/"ima_oos_grid_full.csv", index=False, encoding="utf-8-sig")
llm_grid.to_csv(OUT/"llm_oos_grid_full.csv", index=False, encoding="utf-8-sig")
print(report)
print(f"\n  基准 momentum OOS={fmt_v(mu)}  IMA OOS={fmt_v(iu)}  LLM OOS={fmt_v(lu)}")
print("saved llm_ima_oos_compare.md")
