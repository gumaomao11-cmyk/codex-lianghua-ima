# -*- coding: utf-8 -*-
"""Rebuild HEAT panel from the full 11,387-line LLM cache (not the 53-row parquet)."""
import json, numpy as np, pandas as pd, duckdb, warnings, sys
warnings.filterwarnings("ignore")
sys.path.insert(0,".")
START, END = "2025-07-01", "2026-08-26"
STOCK_DIR = r"F:\even-codex\us-stock-data"

recs=[]
for line in open("backtest_output/zsxq_19_26_granular_cache.jsonl",encoding="utf-8"):
    line=line.strip()
    if not line: continue
    try: o=json.loads(line)
    except Exception: continue
    ct=o.get("create_time")
    evs=o.get("events") or []
    for e in evs:
        if not isinstance(e,dict): continue
        if not e.get("is_us_stock"): continue
        tk=e.get("ticker")
        if not tk or not isinstance(tk,str): continue
        recs.append(dict(create_time=ct, ticker=tk.strip().upper(),
                         text_type=e.get("text_type"),
                         tier=e.get("materiality_tier"),
                         sent=e.get("sentiment_score"),
                         conf=e.get("confidence")))
ev=pd.DataFrame(recs)
ev["ts"]=pd.to_datetime(ev.create_time, errors="coerce")
ev=ev.dropna(subset=["ts"])
ev["date"]=ev.ts.dt.normalize()
print("events:",len(ev),"tickers:",ev.ticker.nunique(),"date range",ev.date.min().date(),ev.date.max().date())
print("by text_type:", ev.text_type.value_counts().to_dict())

con=duckdb.connect()
px=con.execute(f"""select date,ticker,close,ret_1d,ret_5d,ret_21d,mom_20d,vol_20d
                   from 'data/duckdb/aligned_dataset_a_ortho.parquet'
                   where date>='2024-06-01' and date<='{END}'""").df()
px["date"]=pd.to_datetime(px["date"])
tdays=np.array(sorted(px.date.unique()))
univ=set(px.ticker.unique())
print("events with ticker in 515-univ:", ev.ticker.isin(univ).mean().round(3))
ev=ev[ev.ticker.isin(univ)].copy()

# event at T (any hour) -> tradable strictly AFTER T's close -> next trading day
i=np.searchsorted(tdays, ev.date.values, side="right")
ok=i<len(tdays); ev=ev[ok].copy(); ev["td"]=tdays[i[ok]]

agg=ev.groupby(["td","ticker"]).agg(n_ev=("ticker","size"),
                                    sent_mean=("sent","mean"),
                                    conf_mean=("conf","mean")).reset_index()
agg=agg.rename(columns={"td":"date"})
print("heat rows:",len(agg), agg.date.min().date(), agg.date.max().date())

df=px.merge(agg,on=["date","ticker"],how="left")
df["n_ev"]=df.n_ev.fillna(0.0)

fund=pd.read_pickle(rf"{STOCK_DIR}\fundamentals_daily.pkl")
mc=fund["ln_mcap"].loc["2024-06-01":END].stack().rename("ln_mcap").reset_index()
mc.columns=["date","ticker","ln_mcap"]; mc["date"]=pd.to_datetime(mc["date"])
df=df.merge(mc,on=["date","ticker"],how="left")

oh=pd.read_pickle(rf"{STOCK_DIR}\ohlcv.pkl")
dv=np.log1p((oh["Volume"].loc["2024-06-01":END]*oh["Close"].loc["2024-06-01":END])
            .rolling(20,min_periods=5).mean()).stack().rename("ln_dvol").reset_index()
dv.columns=["date","ticker","ln_dvol"]; dv["date"]=pd.to_datetime(dv["date"])
df=df.merge(dv,on=["date","ticker"],how="left")

w=df.pivot_table(index="date",columns="ticker",values="close")
m126=w.pct_change(126).stack().rename("mom_126d").reset_index()
m126.columns=["date","ticker","mom_126d"]
df=df.merge(m126,on=["date","ticker"],how="left")

try:
    from risk.industry_map import get_industry_map
    df=df.merge(get_industry_map()[["ticker","sector"]],on="ticker",how="left")
except Exception as e:
    print("no industry map:",e); df["sector"]=np.nan
df["sector"]=df["sector"].fillna("OTHER")

df=df.sort_values(["ticker","date"])
df["heat_dummy"]=(df.n_ev>0).astype(float)
df["heat_count"]=df.n_ev
df["heat_ln"]=np.log1p(df.n_ev)
df["ev20"]=df.groupby("ticker")["n_ev"].transform(lambda s: s.shift(1).rolling(20,min_periods=5).mean())
df["heat_surprise"]=df.n_ev-df["ev20"]

out=df[(df.date>=START)&(df.date<=END)].copy()
print("panel:",out.shape,"dates:",out.date.nunique(),"tickers:",out.ticker.nunique())
print("avg discussed per day:",out.groupby("date").heat_dummy.sum().mean().round(1))
print("ctrl missing rate: ln_mcap=%.1f%% ln_dvol=%.1f%% mom126=%.1f%%"%(
    out.ln_mcap.isna().mean()*100,out.ln_dvol.isna().mean()*100,out.mom_126d.isna().mean()*100))
out.to_parquet("data/duckdb/heat_panel.parquet",index=False)
print("saved data/duckdb/heat_panel.parquet")
