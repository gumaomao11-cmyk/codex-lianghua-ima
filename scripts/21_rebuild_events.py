# -*- coding: utf-8 -*-
"""
FIX 1: Rebuild the granular events parquet from the FULL LLM cache.
Fixes vs old build_parquet_from_cache.py:
  - old hardcoded a 42-ticker whitelist -> we use the real 515-name price universe
  - old kept only date, dropping intraday time -> we keep timestamp for correct T+1 mapping
  - applies ticker_norm (CN name -> ticker, blacklist A/HK, enum repair)
"""
import json, sys, warnings
from pathlib import Path
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, r"F:\even-codex\lianghua+IMA")
sys.stdout.reconfigure(encoding="utf-8")

PROJ = Path(r"F:\even-codex\lianghua+IMA")
OUT  = PROJ / "backtest_output"
DB   = PROJ / "data" / "duckdb"

# real universe from prices
px = pd.read_parquet(DB / "prices.parquet")
UNIV = set(c for c in px.columns if c != "date")
print(f"[universe] {len(UNIV)} tickers from prices.parquet")

try:
    from ticker_norm import norm_ticker, clean_events
    HAVE_NORM = True
except Exception as e:
    print("[warn] ticker_norm unavailable:", e); HAVE_NORM = False

VALID_TYPE = {"research_report","news_summary","single_event","personal_opinion","noise"}
VALID_TIER = {"tier_1_hard_data","tier_2_soft_logic","tier_3_macro_industry"}
DEFAULT_HORIZON = {"research_report":20,"single_event":3,"news_summary":1,"personal_opinion":1,"noise":1}

def fuzzy(v, valid, default):
    if not isinstance(v,str): return default
    v=v.strip().lower()
    if v in valid: return v
    for c in valid:
        if v.replace("_","") == c.replace("_",""): return c
    # token overlap
    best,bs=default,0
    for c in valid:
        s=len(set(v.split("_"))&set(c.split("_")))
        if s>bs: best,bs=c,s
    return best

CACHES = ["zsxq_19_26_granular_cache.jsonl", "zsxq_v3_clean_sample.jsonl", "zsxq_19_26_events_cache.jsonl"]
rows=[]; stats=dict(lines=0,ev=0,not_us=0,no_tk=0,off_univ=0,kept=0,bad_json=0)
seen=set()
for cn in CACHES:
    p=OUT/cn
    if not p.exists(): print(f"[skip] {cn} missing"); continue
    for line in p.read_text(encoding="utf-8",errors="replace").splitlines():
        line=line.strip()
        if not line: continue
        stats["lines"]+=1
        try: o=json.loads(line)
        except Exception: stats["bad_json"]+=1; continue
        ct=o.get("create_time")
        if not ct: continue
        ts=pd.to_datetime(str(ct), errors="coerce", utc=True)
        if ts is not None and not pd.isna(ts): ts=ts.tz_convert(None) if ts.tzinfo else ts
        if pd.isna(ts): continue
        sig=o.get("sig") or ""
        evs=o.get("events") or []
        for e in evs:
            if not isinstance(e,dict): continue
            stats["ev"]+=1
            if not e.get("is_us_stock"): stats["not_us"]+=1; continue
            tk=e.get("ticker")
            if not isinstance(tk,str) or not tk.strip(): stats["no_tk"]+=1; continue
            tk=tk.strip().upper()
            if HAVE_NORM:
                try:
                    nt=norm_ticker(tk)
                    if nt: tk=nt
                except Exception: pass
            if tk not in UNIV: stats["off_univ"]+=1; continue
            tt=fuzzy(e.get("text_type"), VALID_TYPE, "noise")
            if tt=="noise": continue
            tier=fuzzy(e.get("materiality_tier"), VALID_TIER, "tier_3_macro_industry")
            try: s=float(e.get("sentiment_score",0.0))
            except Exception: s=0.0
            try: c=float(e.get("confidence",0.5))
            except Exception: c=0.5
            s=max(-1.0,min(1.0,s)); c=max(0.0,min(1.0,c))
            try: hz=int(e.get("expected_horizon_days") or DEFAULT_HORIZON[tt])
            except Exception: hz=DEFAULT_HORIZON[tt]
            hz=max(1,min(60,hz))
            key=(ts.strftime("%Y-%m-%d %H:%M"),tk,tt,round(s,3),(e.get("evidence") or "")[:40])
            if key in seen: continue
            seen.add(key)
            stats["kept"]+=1
            rows.append(dict(event_time=ts, date=ts.normalize(), ticker=tk,
                             text_type=tt, materiality_tier=tier,
                             sentiment_score=s, confidence=c, raw_signal=s*c,
                             horizon_days=hz, evidence=(e.get("evidence") or "")[:300],
                             sig=sig[:60]))
print("[stats]",stats)
df=pd.DataFrame(rows).sort_values("event_time")
print(f"[events] {len(df)} rows, {df.ticker.nunique()} tickers, {df.date.min().date()} ~ {df.date.max().date()}")
print("[by type]",df.text_type.value_counts().to_dict())
print("[by tier]",df.materiality_tier.value_counts().to_dict())
print("[horizon]",df.groupby('text_type').horizon_days.median().to_dict())
outp=DB/"zsxq_events_rebuilt.parquet"
df.to_parquet(outp,index=False)
print("[saved]",outp)
