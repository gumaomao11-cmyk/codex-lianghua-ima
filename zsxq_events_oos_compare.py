# -*- coding: utf-8 -*-
"""IS/OOS 对比：momentum vs +星球事件因子 vs +事件+IMA词频。
使用 walk-forward（IS 前段选参，OOS 后段验证）。覆盖不足时输出受限说明。
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA=Path(r"F:\even-codex\us-stock-data"); OUT=Path(r"F:\even-codex\lianghua+IMA\backtest_output")

def loadfactor(name):
    ifac=pd.read_csv(OUT/name,encoding="utf-8-sig")
    ifac["pdf_date"]=pd.to_datetime(ifac["pdf_date"],errors="coerce")
    return ifac.dropna(subset=["pdf_date"])

px=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce")
recent=px.loc[px.index>=pd.Timestamp("2025-01-01")]; px=px.loc[:,recent.notna().sum()>=150]
ml=px.resample("ME").last(); mom=(ml.shift(1)/ml.shift(7)-1.0).replace([np.inf,-np.inf],np.nan)
daily_ret=px.pct_change().fillna(0.0); cols=list(px.columns)
rebal=list(ml.truncate("2025-09-30",px.index[-1]).index)

event=loadfactor("zsxq_events_factors.csv"); ima=loadfactor("kb_abstract_factors.csv") if (OUT/"kb_abstract_factors.csv").exists() else pd.DataFrame(columns=["pdf_date","ticker","n_pos","n_neg","sign"])

def fac_score(fac,d,window):
    s=fac[fac.pdf_date<d]; s=s[s.pdf_date>=d-pd.Timedelta(days=window)]
    if s.empty: return pd.Series(dtype=float)
    g=s.groupby("ticker").agg(n=("n_pos","sum"),d=("n_neg","sum"),sig=("sign","sum"))
    sc=g["n"]-g["d"]+g["sig"]*0.5
    return (sc-sc.mean())/(sc.std() if sc.std()>0 else 1.0)

def monthly(window, lam, fac=event, extra=None, extra_lam=0.0, cost_bps=5):
    rets={}
    for i,d in enumerate(rebal):
        if d not in mom.index: continue
        m=mom.loc[d].dropna()
        if m.empty: continue
        score=m.rank(pct=True)
        score=score+lam*fac_score(fac,d,window).reindex(m.index).fillna(0.0)
        if extra is not None:
            score=score+extra_lam*fac_score(extra,d,window).reindex(m.index).fillna(0.0)
        sel=score.sort_values(ascending=False).index[:10]
        if len(sel)==0: continue
        w=pd.Series(0.0,index=cols); w[sel]=1/len(sel)
        end=rebal[i+1] if i+1<len(rebal) else px.index[-1]
        days=px.index[(px.index>d)&(px.index<=end)]
        if len(days)==0: continue
        r=(w.reindex(cols).values*daily_ret.loc[days,:].values).sum(axis=1)
        r=r-((w-pd.Series(0.0,index=cols)).abs().sum()/2.0)*cost_bps/10000.0
        rets[d]=float(r.sum())
    return pd.Series(rets)
def summ(s):
    s=s.dropna()
    if len(s)==0: return {"n":0,"mean":np.nan,"vol":np.nan,"sharpe":np.nan}
    if len(s)==1: return {"n":1,"mean":float(s.mean()),"vol":np.nan,"sharpe":np.nan}
    mr=float(s.mean()); v=float(s.std(ddof=0)); return {"n":len(s),"mean":mr*100,"vol":v*100,"sharpe":float(mr/v) if v>0 else np.nan}
def fmt(x): return f"{x:.2f}" if x is not None and not (isinstance(x,float) and np.isnan(x)) else "n/a"

mom_ret=monthly(60,0)
# 判断事件因子在哪些 rebal 有实际覆盖
cov=[ (d, int((event.pdf_date>=d-pd.Timedelta(days=90)).sum())) for d in rebal ]
print("rebalance months with event factor coverage(>=1 row in 90d):",
      [(d.date().strftime('%Y-%m'),c) for d,c in cov if c>0])

grid=[]
for w in [30,60,90]:
    for lam in [0.3,0.7,1.2]:
        r=monthly(w,lam)
        # IS = 前5调仓（有数据才算）
        rsub=r.dropna()
        if len(rsub)<=3:
            grid.append({"w":w,"lam":lam,"is":np.nan,"oos":np.nan,"n":len(rsub)})
            continue
        n_is=min(5,max(1,len(rsub)-3))
        grid.append({"w":w,"lam":lam,
                     "is":summ(rsub.iloc[:n_is])["sharpe"],
                     "oos":summ(rsub.iloc[n_is:])["sharpe"],
                     "n":len(rsub)})
g=pd.DataFrame(grid)
def pick(d):
    dd=d.replace({"is":{np.nan:-999}}); return dd.loc[dd["is"].idxmax()]
best=g.loc[g["is"].idxmax()] if len(g) else {}
lines=[]
lines.append("# 星球事件因子 (LLM 结构化) IS/OOS 验证")
lines.append(f"- 区间 {rebal[0].date()}~{rebal[-1].date()}，调仓 {len(rebal)}，10只等权，5bp")
lines.append(f"- 事件因子行数：{len(event)}；有因子覆盖的调仓月（近90天内有数据）见上方")
lines.append(f"- ⚠️ 星球源(浑水调研Plus)仅覆盖 2026-06~08，历史调仓月因子为空，结论仅代表最近段")
lines.append("")
lines.append("## 基准 momentum")
lines.append(f"- 全期月均 {summ(mom_ret)['mean']:.2f}% 夏普 {fmt(summ(mom_ret)['sharpe'])} (n={len(mom_ret)})")
lines.append("")
lines.append("## 动量+星球事件因子（IS 选参）")
if len(g):
    lines.append(f"- 最优 window={int(best['w'])} λ={best['lam']} IS夏普 {fmt(best['is'])} OOS夏普 {fmt(best['oos'])} n={int(best['n'])}")
lines.append("")
lines.append("## 参数网格")
lines.append("| window | λ | IS夏普 | OOS夏普 | n |")
lines.append("|---|---:|---:|---:|---:|")
for _,r in g.sort_values("is",ascending=False).iterrows():
    lines.append(f"| {int(r['w'])} | {r['lam']} | {fmt(r['is'])} | {fmt(r['oos'])} | {int(r['n'])} |")
# 最近3个月直接对照
recent_rebal=[d for d in rebal if d>=pd.Timestamp('2026-05-31')]
if recent_rebal:
    lines.append("")
    lines.append("## 最近段(2026-06~08)直读")
    lines.append("| 调仓日 | momentum 选中 | +事件(60/1.2) 选中 |")
    for d in recent_rebal:
        try:
            m=mom.loc[d].dropna()
            base=m.rank(pct=True).sort_values(ascending=False).index[:10]
            sc=m.rank(pct=True)+1.2*fac_score(event,d,60).reindex(m.index).fillna(0.0)
            ev=sc.sort_values(ascending=False).index[:10]
            lines.append(f"| {d.date()} | {','.join(base)} | {','.join(ev)} |")
        except Exception as e:
            lines.append(f"| {d.date()} | error {str(e)[:40]} | |")
lines.append("")
lines.append("## 结论")
lines.append("- 样本仅3个月(factor有定义的调仓月极少)，下列数字不能作为可靠夏普结论，仅验证管线可用性与方向。")
outp=OUT/"zsxq_events_oos_compare.md"
(outp).write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))
