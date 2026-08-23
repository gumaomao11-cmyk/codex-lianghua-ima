# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path(r"F:\even-codex\lianghua2\backtest_output")
DATA = Path(r"F:\even-codex\us-stock-data")
IDX = Path(r"F:\even-codex\panda\backtest\prices_2016.csv")
START = 20000.0

etf = pd.read_csv(IDX, index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce").ffill()
stk = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
stk = stk.loc[:, stk.count() >= 2400]

def ml(px): return px.resample("ME").last()
def mom(px, period, skip):
    m=ml(px); return m.shift(skip)/m.shift(skip+period)-1
mret = ml(stk).pct_change().fillna(0.0)
score_idx = mret.index
spy_m = ml(etf[['SPY']])['SPY']
spy_daily = etf['SPY']
f_none = pd.Series(True, index=score_idx)
f_200d = (spy_daily > spy_daily.rolling(200,min_periods=200).mean()).reindex(score_idx).fillna(False)
f_10m  = spy_m >= spy_m.rolling(10,min_periods=10).mean()
f_12_1 = mom(etf[['SPY']],12,1)['SPY'] > 0
filters = {"none":f_none, "SPY>200dSMA":f_200d, "SPY10mSMA":f_10m, "SPY12-1>0":f_12_1}

def run(px, mret, sc, filt, top, cost_bps, sd, ed, cap=START):
    dates = list(mret.index)
    in_range = [d for d in dates if sd <= d <= ed]
    value=cap; rets=[]; rng=[]; jour=[]; w_prev=pd.Series(0.0, index=px.columns)
    for i in range(1, len(in_range)):
        prev_d=in_range[i-1]
        if prev_d not in sc.index: continue
        s=sc.loc[prev_d].dropna(); inv=bool(filt.loc[prev_d]) if prev_d in filt.index else True
        if inv and len(s)>0:
            ta=s.sort_values(ascending=False).index[:top].tolist()
            w=pd.Series(0.0,index=px.columns); w[ta]=1.0/len(ta)
        else:
            w=pd.Series(0.0,index=px.columns)
        to=(w-w_prev).abs().sum()/2; c=to*cost_bps/10000.0
        r=float((w*mret.loc[in_range[i]]).sum())-c
        r=max(r,-0.5); value*=(1+r); rets.append(r); rng.append(in_range[i]); jour.append((w.abs().sum()>1e-6)); w_prev=w.copy()
    if not rets: return None
    ret=pd.Series(rets, index=pd.DatetimeIndex(rng)); nav=(1+ret).cumprod()*cap
    ann=(nav.iloc[-1]/cap)**(12/len(ret))-1; vol=ret.std(ddof=1)*np.sqrt(12); sh=ann/vol if vol>0 else np.nan
    dd=(nav/nav.cummax()-1).min(); cal=ann/abs(dd) if dd<0 else np.nan
    return dict(nav=nav, ret=ret, ann=ann, vol=vol, sharpe=sh, mdd=dd, calmar=cal,
                final=nav.iloc[-1], active=float(np.mean(jour)), cash_months=int(len(jour)-sum(jour)), n=len(rets))

def bh_m(px, t, sd, ed, cap=START):
    p = px[t].dropna()
    years = p.resample('ME').last()
    sub = years[(years.index>=sd)&(years.index<=ed)]
    r = sub.pct_change().dropna()
    nav=(1+r).cumprod()*cap
    ann=(nav.iloc[-1]/cap)**(12/len(r))-1; vol=r.std(ddof=1)*np.sqrt(12); sh=ann/vol if vol>0 else np.nan
    dd=(nav/nav.cummax()-1).min(); cal=ann/abs(dd) if dd<0 else np.nan
    return dict(nav=nav, ret=r, ann=ann, vol=vol, sharpe=sh, mdd=dd, calmar=cal,
                final=nav.iloc[-1], active=1.0, cash_months=0, n=len(r))

IS = (pd.Timestamp("2017-01-31"), pd.Timestamp("2021-12-31"))
OOS = (pd.Timestamp("2022-01-31"), pd.Timestamp("2026-08-31"))

def metrics(r):
    return dict(ann=f"{r['ann']*100:.1f}%", vol=f"{r['vol']*100:.1f}%",
                sharpe=f"{r['sharpe']:.2f}", mdd=f"{r['mdd']*100:.1f}%",
                calmar=f"{r['calmar']:.2f}", final=f"${r['final']:,.0f}",
                active=f"{r['active']*100:.0f}%", cash_months=r['cash_months'])

candidates = [
    ("Mom6_top20", 6, 0, 20, "none"),
    ("Mom6_top20+SPY>200dSMA", 6, 0, 20, "SPY>200dSMA"),
    ("Mom6_top20+SPY10mSMA", 6, 0, 20, "SPY10mSMA"),
    ("Mom9-1_top10+SPY>200dSMA", 9, 1, 10, "SPY>200dSMA"),
    ("Mom12-1_top20+SPY>200dSMA", 12, 1, 20, "SPY>200dSMA"),
]
rows=[]
for cname, period, skip, top, fname in candidates:
    sc = mom(stk, period, skip)
    filt = filters[fname]
    isr = run(stk, mret, sc, filt, top, 10, IS[0], IS[1])
    oosr = run(stk, mret, sc, filt, top, 10, OOS[0], OOS[1])
    rows.append(dict(strategy=cname,
                     IS_sharpe=f"{isr['sharpe']:.2f}", IS_ann=f"{isr['ann']*100:.1f}%",
                     IS_mdd=f"{isr['mdd']*100:.1f}%", IS_active=f"{isr['active']*100:.0f}%",
                     OOS_sharpe=f"{oosr['sharpe']:.2f}", OOS_ann=f"{oosr['ann']*100:.1f}%",
                     OOS_mdd=f"{oosr['mdd']*100:.1f}%", OOS_active=f"{oosr['active']*100:.0f}%",
                     OOS_cash_months=oosr['cash_months'], OOS_final=f"${oosr['final']:,.0f}"))

bench = {}
for t in ['SPY','QQQ','SPMO','USMV']:
    b2 = bh_m(etf, t, OOS[0], OOS[1])
    rows.append(dict(strategy="BH "+t,
                     IS_sharpe="-", IS_ann="-", IS_mdd="-", IS_active="-",
                     OOS_sharpe=f"{b2['sharpe']:.2f}", OOS_ann=f"{b2['ann']*100:.1f}%",
                     OOS_mdd=f"{b2['mdd']*100:.1f}%", OOS_active="100%", OOS_cash_months=0,
                     OOS_final=f"${b2['final']:,.0f}"))

res = pd.DataFrame(rows)
res.to_csv(OUT/"oos_results.csv", index=False, encoding="utf-8-sig")
print("OOS period: 2022-01 .. 2026-08 (stock momentum, cost 10bps/side)")
print(res.to_string(index=False))
