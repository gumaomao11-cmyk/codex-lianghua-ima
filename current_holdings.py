# -*- coding: utf-8 -*-
import os
"""Generate current top-10 holdings (momentum / momentum+accel) + strategy spec numbers."""
from pathlib import Path
import numpy as np, pandas as pd
DATA=Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data")
from _paths import OUT
stk=pd.read_csv(DATA/"prices.csv",index_col=0,parse_dates=True).sort_index().apply(pd.to_numeric,errors="coerce")
stk_full=stk.loc[:, stk.count()>=2400]
def ml(x): return x.resample("ME").last()
def mom_frame(px,p,k):
    m=ml(px); return m.shift(k)/m.shift(p+k)-1
def accel_frame(px,p,k,am=1,wm=0.5):
    """主策略默认：0.5*动量 + 0.5*近1月加速"""
    m=mom_frame(px,p,k); a=ml(px).pct_change(am)
    return wm*m+(1-wm)*a

def holdings_by_frame(px, sc, top, capital=20000.0):
    last_label = sc.index[sc.index <= px.index[-1].to_period('M').to_timestamp()]
    if len(last_label)==0:
        raise RuntimeError('no month history')
    d=last_label[-1]
    s=sc.loc[d].dropna().sort_values(ascending=False)[:top]
    px_latest=px.loc[px.index[-1], list(s.index)].dropna()
    out=[]
    for t in s.index:
        if t not in px_latest: continue
        w=1.0/top; alloc=capital*w; price=px_latest[t]
        out.append(dict(ticker=t, rank=s.index.tolist().index(t)+1, momentum=float(s[t]),
                        signal_date=str(pd.Timestamp(d).date()), weight=w,
                        price=float(price), alloc_usd=alloc, shares=alloc/price))
    return pd.DataFrame(out).sort_values('rank')

for label, frame in [("6m_skip1_top10", mom_frame(stk_full,6,1)),
                     ("9m_top10", mom_frame(stk_full,9,0)),
                     ("6m_skip1_accel_top10", accel_frame(stk_full,6,1))]:
    dfh=holdings_by_frame(stk_full, frame, 10)
    dfh.to_csv(OUT/f"current_holdings_{label}.csv",index=False,encoding="utf-8-sig")
    print("="*80)
    print("Config:", label, "  signal date:", dfh["signal_date"].iloc[0])
    print(dfh[["rank","ticker","momentum","price","alloc_usd","shares"]].round(4).to_string(index=False))
print("\nLatest data date:", stk.index[-1].date())
