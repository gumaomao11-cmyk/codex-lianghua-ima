# -*- coding: utf-8 -*-
"""
FIX 2 + FIX 3: rebuild aligned dataset with the two bugs fixed.

BUG A (unbounded ffill): old code did factor.ffill() with no limit -> 88.1% of
  factor values were stale copies, one value persisting up to 399 trading days.
  FIX: decay/hold the signal only for its own horizon_days, then let it expire.
       research_report -> 20d, single_event -> 3d, news/opinion -> 1d.
       We use an exponential decay with half-life = horizon/2 and a hard cutoff.

BUG B (turnover placeholder): old code set turnover_20d = 20d mean of CLOSE.
  FIX: real dollar volume from us-stock-data\ohlcv.pkl, plus real ln_mcap
       from fundamentals_daily.pkl.

Also: no-lookahead is enforced by mapping an event at timestamp T to the FIRST
trading day strictly AFTER T (so an event at 2025-03-05 13:08 is usable 03-06).
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")

PROJ=Path(r"F:\even-codex\lianghua+IMA"); DB=PROJ/"data"/"duckdb"
STOCK=Path(r"F:\even-codex\us-stock-data")
HORIZON={"research_report":20,"single_event":3,"news_summary":1,"personal_opinion":1}

px=pd.read_parquet(DB/"prices.parquet"); px["date"]=pd.to_datetime(px["date"])
px=px.sort_values("date")
tdays=np.array(sorted(px["date"].unique()))
long=px.melt(id_vars=["date"],var_name="ticker",value_name="close").dropna(subset=["close"])
print(f"[prices] {long.shape}, {long.ticker.nunique()} tickers, {long.date.min().date()}~{long.date.max().date()}")

ev=pd.read_parquet(DB/"zsxq_events_rebuilt.parquet")
ev["event_time"]=pd.to_datetime(ev["event_time"])
# no-lookahead: first trading day STRICTLY AFTER the event timestamp
i=np.searchsorted(tdays, ev["event_time"].values, side="right")
ok=i<len(tdays); ev=ev[ok].copy(); ev["td"]=tdays[i[ok]]
print(f"[events] {len(ev)} usable, mapped to trading days {pd.Timestamp(ev.td.min()).date()}~{pd.Timestamp(ev.td.max()).date()}")

def decayed_factor(sub, name, horizon):
    """Signal lives for `horizon` trading days with exp decay, then expires (no ffill)."""
    g=sub.groupby(["td","ticker"])["raw_signal"].mean().rename(name).reset_index()
    g=g.rename(columns={"td":"date"})
    base=long[["date","ticker"]].merge(g,on=["date","ticker"],how="left")
    base=base.sort_values(["ticker","date"])
    hl=max(horizon/2.0,0.5); lam=np.log(2)/hl
    out=[]
    for tk,gg in base.groupby("ticker",sort=False):
        v=gg[name].to_numpy(dtype=float); n=len(v); res=np.full(n,np.nan)
        cur=np.nan; age=0
        for k in range(n):
            if not np.isnan(v[k]):
                cur=v[k]; age=0; res[k]=cur
            elif not np.isnan(cur):
                age+=1
                if age<horizon:                # hard cutoff = expiry
                    res[k]=cur*np.exp(-lam*age)
                else:
                    cur=np.nan; res[k]=np.nan
        gg=gg.copy(); gg[name]=res; out.append(gg)
    return pd.concat(out,ignore_index=True)

print("\n[route A] clean alpha = research_report + single_event, tier_1 only")
cleanA=ev[(ev.text_type.isin(["research_report","single_event"])) & (ev.materiality_tier=="tier_1_hard_data")]
print(f"  clean events: {len(cleanA)}")
fa=decayed_factor(cleanA,"factor_clean_alpha",10)

print("[route B] per-pool factors with pool-specific horizons")
fb=long[["date","ticker"]].copy()
for tt,hz in HORIZON.items():
    sub=ev[ev.text_type==tt]
    nm={"research_report":"factor_research_20d","single_event":"factor_event_3d",
        "news_summary":"factor_news_1d","personal_opinion":"factor_opinion_1d"}[tt]
    d=decayed_factor(sub,nm,hz)
    if tt=="personal_opinion": d[nm]=-d[nm]     # keep original convention
    fb=fb.merge(d[["date","ticker",nm]],on=["date","ticker"],how="left")
    print(f"  {tt:<18} horizon={hz:>2}d  events={len(sub):>5}  nonnull={d[nm].notna().sum():>7}")

# ---- real controls ----
print("\n[controls] real dollar volume + real ln_mcap")
oh=pd.read_pickle(STOCK/"ohlcv.pkl")
dvol=(oh["Volume"]*oh["Close"]).rolling(20,min_periods=5).mean()
dvol=np.log1p(dvol).stack().rename("ln_dvol_20d").reset_index(); dvol.columns=["date","ticker","ln_dvol_20d"]
dvol["date"]=pd.to_datetime(dvol["date"])
turn=(oh["Volume"].rolling(20,min_periods=5).mean()).stack().rename("turnover_20d").reset_index()
turn.columns=["date","ticker","turnover_20d"]; turn["date"]=pd.to_datetime(turn["date"])
fund=pd.read_pickle(STOCK/"fundamentals_daily.pkl")
mcap=fund["ln_mcap"].stack().rename("ln_mcap").reset_index(); mcap.columns=["date","ticker","ln_mcap"]
mcap["date"]=pd.to_datetime(mcap["date"])

def finish(fac,route):
    df=fac.sort_values(["ticker","date"]).copy()
    df=df.merge(long,on=["date","ticker"],how="left")
    out=[]
    for tk,g in df.groupby("ticker",sort=False):
        g=g.sort_values("date").copy()
        for h in [1,5,10,21]:
            g[f"ret_{h}d"]=g["close"].shift(-h)/g["close"]-1.0
        g["mom_20d"]=g["close"].pct_change(20)
        g["mom_126d"]=g["close"].pct_change(126)
        g["vol_20d"]=g["close"].pct_change().rolling(20).std()*np.sqrt(252)
        d=g["close"].diff()
        gain=d.where(d>0,0).rolling(14).mean(); loss=(-d.where(d<0,0)).rolling(14).mean()
        g["rsi_14"]=100-100/(1+gain/loss)
        out.append(g)
    df=pd.concat(out,ignore_index=True)
    df=df.merge(turn,on=["date","ticker"],how="left")
    df=df.merge(dvol,on=["date","ticker"],how="left")
    df=df.merge(mcap,on=["date","ticker"],how="left")
    df["route"]=route
    return df

for fac,route,tag in [(fa,"A","a"),(fb,"B","b")]:
    d=finish(fac,route)
    p=DB/f"aligned_v2_{tag}.parquet"
    d.to_parquet(p,index=False)
    fcols=[c for c in d.columns if c.startswith("factor_")]
    print(f"\n[saved] {p}  {d.shape}")
    for c in fcols:
        s=d[c]
        print(f"   {c}: nonnull={s.notna().sum():>7} ({s.notna().mean():.2%})  mean={s.mean():+.4f}")
    print(f"   turnover_20d real? corr(close)={d[['close','turnover_20d']].corr().iloc[0,1]:+.3f}  (was ~+1.0 when it was mean price)")
