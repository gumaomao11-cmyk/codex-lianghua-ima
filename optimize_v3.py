# -*- coding: utf-8 -*-
"""
Optimization & robustness experiments for monthly momentum strategy.
 - Parameter sweep with IS/OOS split
 - Upgrades: risk-parity weights, low-vol blend, vol targeting, trend filter
"""
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(r"F:\even-codex\us-stock-data")
IDX = Path(r"F:\even-codex\panda\backtest\prices_2016.csv")
OUT = Path(r"F:\even-codex\lianghua2\backtest_output"); OUT.mkdir(parents=True, exist_ok=True)
DAYS=252; START=20000.0

stk = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
stk = stk.loc[:, stk.count()>=2400]
etf = pd.read_csv(IDX, index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce").ffill()

def ml(x): return x.resample("ME").last()
def mom_score(px, period, skip):
    m=ml(px); return m.shift(skip)/m.shift(skip+period)-1
def vol_rank(px, look=60):
    daily=px.pct_change()
    v=daily.rolling(look).std()*np.sqrt(DAYS)
    return v.resample("ME").last()

def run(px, scores, top, cost_bps, risk_weight=False, vol_target=None):
    cols=list(px.columns); dr=px.pct_change().fillna(0.0)
    me=pd.DatetimeIndex(scores.index); W=pd.DataFrame(0.0, index=px.index, columns=cols); rd=[]
    for i in range(1,len(me)):
        d=me[i]; w=pd.Series(0.0,index=cols); s=scores.loc[d].dropna()
        if len(s)>0:
            ta=list(s.sort_values(ascending=False).index[:top])
            w[ta]=1.0/len(ta)
            if risk_weight:
                vv=vol_rank(px,60).loc[d].reindex(ta)
                inv=(1.0/vv.replace(0,np.nan)).fillna(1.0)
                inv=inv.clip(lower=1e-3)
                w[ta] = (inv/inv.sum()).values
        end=me[i+1] if i+1<len(me) else px.index[-1]
        days=px.index[(px.index>d)&(px.index<=end)]
        if len(days):
            for c in cols: W.loc[days,c]=w[c]
            rd.append(days[0])
    ug=(W*dr).sum(axis=1).fillna(0.0)
    scale=pd.Series(1.0,index=ug.index)
    if vol_target:
        sarr={}
        for r0 in rd:
            look=ug.loc[:r0].iloc[-(61):-1].dropna()
            s=min(1.0, vol_target/(look.std(ddof=1)*np.sqrt(DAYS))) if len(look)>=21 and look.std(ddof=1)>0 else 1.0
            sarr[r0]=s
        for i in range(1,len(me)):
            d=me[i]; end=me[i+1] if i+1<len(me) else px.index[-1]
            days=px.index[(px.index>d)&(px.index<=end)]
            if len(days)>0:
                scale.loc[days]=sarr.get(days[0],1.0)
        W=W.mul(scale,axis=0)
    g=(W*dr).sum(axis=1).fillna(0.0)
    cost=pd.Series(0.0,index=g.index); prev=pd.Series(0.0,index=cols)
    for r0 in rd:
        d0=r0-pd.Timedelta(days=1)
        wn=W.loc[d0] if d0 in W.index else W.loc[r0]
        to=(wn-prev).abs().sum()/2; cost.loc[r0]=to*cost_bps/10000.0; prev=wn.copy()
    ss=(g-cost).clip(lower=-0.5); nav=(1+ss).cumprod()*START
    return ss, nav, (W.abs().sum(axis=1)>1e-6).mean()

def stats(ret, nav):
    ann=(nav.iloc[-1]/START)**(DAYS/len(nav))-1 if len(nav)>0 else 0
    vol=ret.std(ddof=1)*np.sqrt(DAYS) if len(ret)>1 else 0
    sharpe=(ann/vol) if vol>0 else np.nan
    dd=(nav/nav.cummax()-1).min()
    return dict(ann=ann, vol=vol, sharpe=sharpe, mdd=dd, final=nav.iloc[-1] if len(nav) else START)

# ---------- Parameter sweep (pure equal-weight momentum) ----------
periods=[3,6,9,12]; skips=[0,1]; tops=[10,20]
rows=[]
for p in periods:
    for k in skips:
        sc=mom_score(stk,p,k)
        for top in tops:
            ret,nav,act=run(stk,sc,top,cost_bps=10)
            is_msk=(ret.index>=pd.Timestamp('2017-02-01'))&(ret.index<=pd.Timestamp('2021-12-31'))
            oo_msk=(ret.index>=pd.Timestamp('2022-01-01'))
            isr=stats(ret[is_msk],nav[is_msk]); oor=stats(ret[oo_msk],nav[oo_msk]); fr=stats(ret,nav)
            rows.append(dict(period=p, skip=k, top=top, is_sharpe=isr['sharpe'], oos_sharpe=oor['sharpe'],
                             full_sharpe=fr['sharpe'], is_ann=isr['ann'], oos_ann=oor['ann'],
                             full_ann=fr['ann'], oos_mdd=oor['mdd'], oos_final=oor['final']))
df=pd.DataFrame(rows).sort_values('oos_sharpe', ascending=False)
df.to_csv(OUT/'optimization_sweep.csv', index=False, encoding='utf-8-sig')
print("=== Parameter sweep (ranked by OOS Sharpe) ===")
print(df.to_string(index=False))

best=df.iloc[0]
print("\nBest OOS combo: lookback=%d skip=%d top=%d OOS.sh=%.2f  IS.sh=%.2f" % (best.period,best.skip,best.top,best.oos_sharpe,best.is_sharpe))

# ---------- Upgrades on a selected base (6m, top20) ----------
base_s=mom_score(stk,6,0)
best_s=mom_score(stk,int(best.period),int(best.skip))

def summarize(name, ret, nav, act):
    st=stats(ret,nav)
    print(f"| {name:<34} | ann={st['ann']*100:5.1f}% | vol={st['vol']*100:5.1f}% | sharpe={st['sharpe']:5.2f} | mdd={st['mdd']*100:6.1f}% | final=${st['final']:>9,.0f} | in={act*100:3.0f}% |")

print("\n=== Upgrade candidates ===")
cases=[
    ('Base 6m top20', base_s, 20, {}, 10),
    ('6m top20 risk-parity', base_s, 20, {'risk_weight':True}, 10),
    ('6m top20 vol25', base_s, 20, {'vol_target':0.25}, 10),
    ('6m top20 rp + vol25', base_s, 20, {'risk_weight':True,'vol_target':0.25}, 10),
    ('BestOOS pure', best_s, int(best.top), {}, 10),
    ('BestOOS + vol25', best_s, int(best.top), {'vol_target':0.25}, 10),
]
for name,sc,top,kw,cost in cases:
    ret,nav,act=run(stk,sc,top,cost_bps=cost, **kw)
    summarize(name,ret,nav,act)
