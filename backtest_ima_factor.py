# -*- coding: utf-8 -*-
import sys, csv
from pathlib import Path
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path(r"F:\even-codex\us-stock-data")
OUT  = Path(r"F:\even-codex\lianghua+IMA\backtest_output")
OUT.mkdir(parents=True, exist_ok=True)
px = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
recent = px.loc[px.index>=pd.Timestamp("2025-01-01")]
px = px.loc[:, recent.notna().sum() >= 150]
ml = px.resample("ME").last()
mom = ml.shift(1)/ml.shift(7) - 1.0
mom = mom.replace([np.inf,-np.inf], np.nan)
fac = pd.read_csv(OUT/"kb_abstract_factors.csv", encoding="utf-8-sig")
fac["pdf_date"] = pd.to_datetime(fac["pdf_date"], errors="coerce"); fac = fac.dropna(subset=["pdf_date"])

def ima_score_for(d, window=90):
    s = fac[fac.pdf_date < d]
    s = s[s.pdf_date >= d - pd.Timedelta(days=window)]
    if s.empty: return pd.Series(dtype=float)
    g = s.groupby("ticker").agg(n=("n_pos","sum"), d=("n_neg","sum"), sig=("sign","sum"))
    score = g["n"] - g["d"] + g["sig"]*0.5
    z = (score - score.mean()) / (score.std() if score.std() > 0 else 1.0)
    return z

def run_strategy(mode, start="2025-09-30", top=10, window=90, lam=0.7, cost_bps=5):
    rebal = list(ml.truncate(start, px.index[-1]).index)
    if not rebal: return None
    daily_ret = px.pct_change().fillna(0.0)
    cols = list(px.columns)
    weights = pd.DataFrame(0.0, index=px.index, columns=cols)
    prev = pd.Series(0.0, index=cols)
    cost_events = []
    for i, d in enumerate(rebal):
        if d not in mom.index: continue
        m = mom.loc[d].dropna()
        if m.empty: continue
        if mode == "momentum":
            score = m.rank(pct=True)
        elif mode == "momentum_ima":
            z = ima_score_for(d, window).reindex(m.index)
            score = m.rank(pct=True) + lam*z.fillna(0.0)
        elif mode == "ima_only":
            z = ima_score_for(d, window).reindex(m.index)
            if z.notna().sum()==0: continue
            score = z.fillna(-1e9)
        else: raise ValueError(mode)
        sel = score.sort_values(ascending=False).index[:top].tolist()
        if not sel: continue
        w = pd.Series(0.0, index=cols); w[sel] = 1.0/len(sel)
        start_idx = rebal[i+1] if i+1 < len(rebal) else px.index[-1]
        days = px.index[(px.index > d) & (px.index <= start_idx)]
        if len(days):
            weights.loc[days, :] = w.reindex(cols).values
            cost_events.append((days[0], (w-prev).abs().sum()/2.0))
            prev = w
    gross = (weights * daily_ret).sum(axis=1)
    cost = pd.Series(0.0, index=gross.index)
    for day, t in cost_events: cost.loc[day] = t*cost_bps/10000.0
    ret = (gross - cost).fillna(0.0).clip(lower=-0.5)
    nav = (1+ret).cumprod()*20000
    ix = nav.index >= pd.Timestamp("2025-10-01")
    nav2=nav[ix]; rr=ret[ix]
    ann=(nav2.iloc[-1]/nav2.iloc[0])**(252/len(rr))-1
    vol=rr.std()*np.sqrt(252); sharpe=ann/vol if vol>0 else np.nan
    roll=nav2.cummax(); dd=(nav2/roll-1.0); maxdd=dd.min()
    n_reb=len(cost_events); ann_turn=sum(t for _,t in cost_events)/max(n_reb,1)*12
    print(f"[{mode}] final=${nav2.iloc[-1]:,.0f} ann={ann*100:.1f}% sharpe={sharpe:.2f} maxdd={maxdd*100:.1f}% reb={n_reb} ann_turn={ann_turn*100:.0f}%")
    return {"mode":mode,"final_nav":float(nav2.iloc[-1]),"ann_ret":ann,"sharpe":sharpe,"max_dd":maxdd,"n_rebal":n_reb,"ann_turnover":ann_turn}

results=[]
for mode in ["momentum","momentum_ima","ima_only"]:
    r=run_strategy(mode)
    if r: results.append(r)
rows=[{"mode":r["mode"],"final_nav":round(r["final_nav"],2),"ann_ret_pct":round(r["ann_ret"]*100,2),"sharpe":round(r["sharpe"],2),"max_dd_pct":round(r["max_dd"]*100,2),"n_rebal":r["n_rebal"],"ann_turnover_pct":round(r["ann_turnover"]*100,1)} for r in results]
pd.DataFrame(rows).to_csv(OUT/"ima_factor_backtest_summary.csv", index=False, encoding="utf-8-sig")
lines=[f"- **{r['mode']}**: 期末净值 ${r['final_nav']:,.0f} | 年化 {r['ann_ret']*100:.1f}% | 夏普 {r['sharpe']:.2f} | 最大回撤 {r['max_dd']*100:.1f}% | 调仓 {r['n_rebal']}次 | 年化换手 {r['ann_turnover']*100:.0f}%" for r in results]
(OUT/"ima_factor_backtest_report.md").write_text("## IMA 因子对比回测 (2025-09 至 2026-08)\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines)); print("saved")
