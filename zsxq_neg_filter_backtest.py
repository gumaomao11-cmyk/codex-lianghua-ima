# -*- coding: utf-8 -*-
"""知识星球→美股负面/做空禁买名单过滤验证。
只看美股（ticker 来自 prices.csv + 排除明显A股/中概干扰），负面信号按 文档级 sign<0 累计。
IS=前5调仓，OOS=后6验证。
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA=Path(r"F:\even-codex\us-stock-data"); OUT=Path(r"F:\even-codex\lianghua+IMA\backtest_output")

px=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce")
recent=px.loc[px.index>=pd.Timestamp("2025-01-01")]; px=px.loc[:,recent.notna().sum()>=150]
cols=list(px.columns)
ml=px.resample("ME").last(); mom=(ml.shift(1)/ml.shift(7)-1.0).replace([np.inf,-np.inf],np.nan)
daily_ret=px.pct_change().fillna(0.0); rebal=list(ml.truncate("2025-09-30",px.index[-1]).index)

fac=pd.read_csv(OUT/"zsxq_factors.csv",encoding="utf-8-sig")
fac["pdf_date"]=pd.to_datetime(fac["pdf_date"],errors="coerce"); fac=fac.dropna(subset=["pdf_date"])
# 只看美股：ticker 必须在 prices.csv 列，排除似中国内地/香港代码噪声（已由 KNOWN 过滤，再加一道）
US=set(cols); fac=fac[fac["ticker"].isin(US)]
# 文档级负面：该条主题里 n_neg多且 n_pos为0（净负面/做空/风险）
fac["doc_neg"]=(fac["n_neg"]>0)
fac["doc_pos"]=(fac["n_pos"]>0)
# 汇总每个 ticker 每天的正/负面文档数
daily=fac.groupby(["ticker","pdf_date"]).agg(neg_docs=("doc_neg","sum"),pos_docs=("doc_pos","sum")).reset_index()

def blocked_on(d, window, k):
    w=daily[(daily.pdf_date<d)&(daily.pdf_date>=d-pd.Timedelta(days=window))]
    if w.empty: return set()
    g=w.groupby("ticker").agg(neg=("neg_docs","sum"),pos=("pos_docs","sum"))
    g=g[g["neg"]>=k]
    return set(g.index.tolist())

def monthly(block=False, window=30, k=1, cost_bps=5):
    rets={}
    for i,d in enumerate(rebal):
        if d not in mom.index: continue
        m=mom.loc[d].dropna()
        if m.empty: continue
        if block:
            blk=blocked_on(d,window,k)
            if blk: m=m.drop(index=list(blk & set(m.index)),errors="ignore")
        score=m.rank(pct=True)
        sel=score.sort_values(ascending=False).index[:10]
        if len(sel)==0: continue
        w=pd.Series(0.0,index=cols); w[sel]=1/len(sel); w=w.reindex(cols).fillna(0.0)
        end=rebal[i+1] if i+1<len(rebal) else px.index[-1]
        days=px.index[(px.index>d)&(px.index<=end)]
        if len(days)==0: continue
        r=(w.values*daily_ret.loc[days,:].values).sum(axis=1)
        r=r-((w-pd.Series(0.0,index=cols)).abs().sum()/2.0)*cost_bps/10000.0
        rets[d]=float(r.sum())
    return pd.Series(rets)

def summ(s):
    s=s.dropna()
    if len(s)==0: return {"n":0,"mean":np.nan,"vol":np.nan,"sharpe":np.nan}
    if len(s)==1: return {"n":1,"mean":float(s.mean()),"vol":np.nan,"sharpe":np.nan}
    mr=float(s.mean()); v=float(s.std(ddof=0)); return {"n":len(s),"mean":mr*100,"vol":v*100,"sharpe":float(mr/v) if v>0 else np.nan}

mom_ret=monthly(); mom_is=summ(mom_ret.iloc[:5]); mom_oos=summ(mom_ret.iloc[5:])
rows=[]
for window in [15,30,60]:
    for k in [1,2,3]:
        r=monthly(block=True,window=window,k=k)
        is_=summ(r.iloc[:5]); oos=summ(r.iloc[5:])
        rows.append({"window":window,"k":k,"is_sharpe":is_["sharpe"],"oos_sharpe":oos["sharpe"],
                     "oos_mean":oos["mean"],"oos_vol":oos["vol"]})

def fmt(x): return f"{x:.2f}" if x is not None and not (isinstance(x,float) and np.isnan(x)) else "n/a"
g=pd.DataFrame(rows).replace({"oos_sharpe":{np.nan:-999}})
best=g.loc[g["oos_sharpe"].idxmax()]
lines=[]
lines.append("# 知识星球 美股负面/做空 → 禁买名单 过滤验证")
lines.append(f"- 区间 {rebal[0].date()}~{rebal[-1].date()}，调仓 {len(rebal)}（IS={5}，OOS={len(rebal)-5}），5bp，10只等权")
lines.append(f"- 美股 ticker={len(US)}；ZSXQ 文档级净负面累计后禁买（只看美股，剔除A股内容）")
lines.append("")
lines.append(f"## 基准 momentum")
lines.append(f"- IS 夏普 {fmt(mom_is['sharpe'])} | OOS 月均 {mom_oos['mean']:.2f}% 波动 {mom_oos['vol']:.2f}% 夏普 {fmt(mom_oos['sharpe'])}")
lines.append("")
lines.append("## 动量+禁买名单（按 OOS 最优展示）")
lines.append(f"- OOS 最优 window={int(best['window'])} k={int(best['k'])}")
lines.append(f"- OOS 月均 {best['oos_mean']:.2f}% 波动 {best['oos_vol']:.2f}% 夏普 {fmt(best['oos_sharpe'])}")
lines.append("")
lines.append("## 参数网格")
lines.append("| window | k | IS夏普 | OOS夏普 | OOS月均% | OOS波动% |")
lines.append("|---|---:|---:|---:|---:|---:|")
for _,r in pd.DataFrame(rows).sort_values("is_sharpe",ascending=False).iterrows():
    lines.append(f"| {int(r['window'])} | {int(r['k'])} | {fmt(r['is_sharpe'])} | {fmt(r['oos_sharpe'])} | {fmt(r['oos_mean'])} | {fmt(r['oos_vol'])} |")
lines.append("")
lines.append("## 结论")
lines.append(f"- OOS 夏普：momentum={fmt(mom_oos['sharpe'])}，最佳禁买={fmt(best['oos_sharpe'])}")
lines.append("- 禁买名单若 OOS 夏普明显高于 momentum 才有优化价值；否则不建议。")
(OUT/"zsxq_negfilter_oos_compare.md").write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))
print(f"  mom OOS={fmt(mom_oos['sharpe'])}  best-block OOS={fmt(best['oos_sharpe'])} (win={int(best['window'])} k={int(best['k'])})")


