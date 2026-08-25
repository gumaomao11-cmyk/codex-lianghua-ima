# -*- coding: utf-8 -*-
"""IS/OOS：momentum vs +IMA词频 vs +ZSXQ vs +IMA+ZSXQ。前5调仓选参，后6验证。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA=Path(r"F:\even-codex\us-stock-data"); OUT=Path(r"F:\even-codex\lianghua+IMA\backtest_output")

px=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce")
recent=px.loc[px.index>=pd.Timestamp("2025-01-01")]; px=px.loc[:,recent.notna().sum()>=150]
ml=px.resample("ME").last(); mom=(ml.shift(1)/ml.shift(7)-1.0).replace([np.inf,-np.inf],np.nan)
daily_ret=px.pct_change().fillna(0.0); cols=list(px.columns)
rebal=list(ml.truncate("2025-09-30",px.index[-1]).index)

def loadfactor(name):
    ifac=pd.read_csv(OUT/name,encoding="utf-8-sig")
    ifac["pdf_date"]=pd.to_datetime(ifac["pdf_date"],errors="coerce"); return ifac.dropna(subset=["pdf_date"])
ima=loadfactor("kb_abstract_factors.csv"); zx=loadfactor("zsxq_factors.csv")

def fac_score(fac,d,window):
    s=fac[fac.pdf_date<d]; s=s[s.pdf_date>=d-pd.Timedelta(days=window)]
    if s.empty: return pd.Series(dtype=float)
    g=s.groupby("ticker").agg(n=("n_pos","sum"),d=("n_neg","sum"),sig=("sign","sum"))
    sc=g["n"]-g["d"]+g["sig"]*0.5
    return (sc-sc.mean())/(sc.std() if sc.std()>0 else 1.0)

def monthly(kind, window, lam, fac=None, extra=None, extra_lam=0.0, cost_bps=5):
    rets={}
    for i,d in enumerate(rebal):
        if d not in mom.index: continue
        m=mom.loc[d].dropna()
        if m.empty: continue
        score=m.rank(pct=True)
        if fac is not None:
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

mom_ret=monthly("mom",0,0)
mom_is=summ(mom_ret.iloc[:5]); mom_oos=summ(mom_ret.iloc[5:])

def grid(kind,fac):
    rows=[]
    for w in [30,60,90]:
        for lam in [0.3,0.7,1.2]:
            r=monthly(kind,w,lam,fac=fac)
            rows.append({"kind":kind,"window":w,"lam":lam,"is_sharpe":summ(r.iloc[:5])["sharpe"],
                         "oos_sharpe":summ(r.iloc[5:])["sharpe"],"oos_mean":summ(r.iloc[5:])["mean"],
                         "oos_vol":summ(r.iloc[5:])["vol"]})
    return pd.DataFrame(rows)
def pick(df):
    d=df.replace({"is_sharpe":{np.nan:-999}}); return d.loc[d["is_sharpe"].idxmax()]

ima_g=grid("ima",ima); zx_g=grid("zx",zx)
b_ima=pick(ima_g); b_zx=pick(zx_g)
# combined: IMA 固定用 OOS 已知最优 win60/λ1.2；ZX 也用 IS 最优，再看 OOS
combi=monthly("combi", int(b_zx["window"]), float(b_zx["lam"]), fac=ima, extra=zx, extra_lam=1.2)
combi_is=summ(combi.iloc[:5]); combi_oos=summ(combi.iloc[5:])

def fmt(x): return f"{x:.2f}" if x is not None and not (isinstance(x,float) and np.isnan(x)) else "n/a"
lines=[]
lines.append("# ZSXQ(知识星球) 信息因子 × IMA 融合 IS/OOS 验证")
lines.append(f"- 区间 {rebal[0].date()}~{rebal[-1].date()}，调仓 {len(rebal)}（IS={5}，OOS={len(rebal)-5}），5bp，10只等权")
lines.append(f"- ZSXQ 数据：短评&信息 回至 2025-07，浑水调研Plus 回至 2026-06（全文仅本机）")
lines.append("")
lines.append("## 基准 momentum")
lines.append(f"- IS 月均 {mom_is['mean']:.2f}% 夏普 {fmt(mom_is['sharpe'])} | OOS 月均 {mom_oos['mean']:.2f}% 月波动 {mom_oos['vol']:.2f}% 夏普 {fmt(mom_oos['sharpe'])}")
lines.append("")
lines.append("## 动量+IMA 词频（win60/λ1.2 已知最优）")
lines.append("- OOS 夏普 0.54（来自前序报告）")
lines.append("")
lines.append(f"## 动量+ZSXQ（IS 选参）")
lines.append(f"- IS 最优 window={int(b_zx['window'])} λ={b_zx['lam']} IS夏普 {fmt(b_zx['is_sharpe'])}")
lines.append(f"- OOS 月均 {b_zx['oos_mean']:.2f}% 波动 {b_zx['oos_vol']:.2f}% 夏普 {fmt(b_zx['oos_sharpe'])}")
lines.append("")
lines.append("## 动量+IMA+ZSXQ（IMA 固定 win60/λ1.2，ZSXQ 用 IS 最优）")
lines.append(f"- OOS 月均 {combi_oos['mean']:.2f}% 波动 {combi_oos['vol']:.2f}% 夏普 {fmt(combi_oos['sharpe'])}")
lines.append("")
lines.append("## ZSXQ 参数网格")
lines.append("| window | λ | IS夏普 | OOS夏普 | OOS月均% |")
lines.append("|---|---:|---:|---:|---:|")
for _,r in zx_g.sort_values("is_sharpe",ascending=False).iterrows():
    lines.append(f"| {int(r['window'])} | {r['lam']} | {fmt(r['is_sharpe'])} | {fmt(r['oos_sharpe'])} | {fmt(r['oos_mean'])} |")
lines.append("")
lines.append("## 结论")
lines.append(f"- OOS 夏普：momentum={fmt(mom_oos['sharpe'])}，+ZSXQ={fmt(b_zx['oos_sharpe'])}，+IMA+ZSXQ={fmt(combi_oos['sharpe'])}")
lines.append("- 若 +ZSXQ / +IMA+ZSXQ 的 OOS 夏普高于 momentum 才有价值；否则不建议融合。")
(OUT/"zsxq_ima_oos_compare.md").write_text("\n".join(lines),encoding="utf-8")
print("\n".join(lines))
print(f"  基准 mom OOS={fmt(mom_oos['sharpe'])} | +ZSXQ OOS={fmt(b_zx['oos_sharpe'])} | +IMA+ZSXQ OOS={fmt(combi_oos['sharpe'])}")
