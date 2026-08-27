# -*- coding: utf-8 -*-
"""动量 + 多源文本因子 walk-forward 回测（浑水调研Plus + IMA 美国科技日报）。
严格 walk-forward：每个调仓日前用此前所有样本重新训练 XGBoost。
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
import xgboost as xgb
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path(r"F:\even-codex\us-stock-data")
OUT  = Path(r"F:\even-codex\lianghua+IMA\backtest_output")

px = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
recent = px.loc[px.index >= pd.Timestamp("2025-01-01")]
px = px.loc[:, recent.notna().sum() >= 150]
daily_ret = px.pct_change().fillna(0.0)
cols = list(px.columns)
ml = px.resample("ME").last()
mom = (ml.shift(1)/ml.shift(7)-1.0).replace([np.inf,-np.inf], np.nan)
rebal = list(ml.truncate("2025-09-30", px.index[-1]).index)

def load_text(src, prefix):
    df = pd.read_csv(OUT / src, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    keep = ["date", "ticker", "tsm", "tna", "disagreement", "risk_flag", "text_cov"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c not in ["date", "ticker"]})
    return df

text_hunshui = load_text("text_sentiment_lexicon_浑水调研Plus.csv", "hs")
text_ima = load_text("text_sentiment_ima.csv", "ima")
print(f"浑水: {len(text_hunshui)} rows, {text_hunshui['ticker'].nunique()} tickers, {text_hunshui['date'].nunique()} dates")
print(f"IMA : {len(text_ima)} rows, {text_ima['ticker'].nunique()} tickers, {text_ima['date'].nunique()} dates")

text = text_hunshui.merge(text_ima, on=["date", "ticker"], how="outer")
for c in text.columns:
    if c not in ["date", "ticker"]:
        text[c] = text[c].fillna(0)
print(f"合并后: {len(text)} rows")

def rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    return 100 - (100 / (1 + gain / loss))

def build_features(d):
    feats = pd.DataFrame(index=cols)
    feats["mom_7m"] = mom.loc[d] if d in mom.index else np.nan
    d_idx = px.index.get_indexer([d], method="nearest")[0]
    feats["mom_10d"] = (px.iloc[d_idx] / px.iloc[d_idx-10] - 1.0).replace([np.inf,-np.inf], np.nan) if d_idx >= 10 else np.nan
    feats["mom_20d"] = (px.iloc[d_idx] / px.iloc[d_idx-20] - 1.0).replace([np.inf,-np.inf], np.nan) if d_idx >= 20 else np.nan
    feats["rsi_14"] = rsi(px.iloc[:d_idx+1], 14).iloc[-1] if d_idx >= 14 else np.nan
    feats["vol_20d"] = daily_ret.iloc[d_idx-19:d_idx+1].std() * np.sqrt(252) if d_idx >= 20 else np.nan

    t = text[(text["date"] < d) & (text["date"] >= d - pd.Timedelta(days=20))]
    for prefix in ["hs", "ima"]:
        for col in ["tsm", "tna", "disagreement"]:
            cname = f"{prefix}_{col}"
            if cname in t.columns:
                feats[cname] = t.groupby("ticker")[cname].mean().reindex(cols).fillna(0)
        for col in ["risk_flag", "text_cov"]:
            cname = f"{prefix}_{col}"
            if cname in t.columns:
                feats[cname] = t.groupby("ticker")[cname].max().reindex(cols).fillna(0)
    return feats

def get_fwd_ret(d):
    d_idx = px.index.get_indexer([d], method="nearest")[0]
    if d_idx + 21 >= len(px): return pd.Series(np.nan, index=cols)
    return (px.iloc[d_idx+21] / px.iloc[d_idx] - 1.0).replace([np.inf,-np.inf], np.nan)

# ---- precompute training samples ----
print("precomputing training samples...")
samples = []
for d in rebal[:-1]:
    feats = build_features(d)
    y = get_fwd_ret(d)
    valid = feats.notna().all(axis=1) & y.notna()
    if valid.sum() >= 5:
        samples.append((d, feats[valid], y[valid]))
print(f"samples per month: {len(samples)}")

text_feat_cols = [c for c in samples[0][1].columns if any(p in c for p in ["hs_", "ima_"])]

def train_model(up_to_date):
    X_all, y_all = [], []
    for d, X, y in samples:
        if d < up_to_date:
            X_all.append(X); y_all.append(y)
    if len(X_all) < 2: return None
    Xtr = pd.concat(X_all, ignore_index=True)
    ytr = pd.concat(y_all, ignore_index=True)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=4)
    model.fit(Xtr, ytr)
    return model

# train isolation forest once on all text features (unsupervised, less sensitive)
all_text = pd.concat([X[text_feat_cols] for _, X, _ in samples], ignore_index=True)
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(all_text)

# ---- monthly backtest ----
def monthly(mode, cost_bps=10):
    rets = []; positions = []
    for i, d in enumerate(rebal[:-1]):
        feat = build_features(d)
        if feat.empty: continue
        if mode == "momentum":
            sc = feat["mom_7m"].rank(pct=True)
        elif mode in ("xgb", "xgb_filter"):
            model = train_model(d)
            valid = feat.notna().all(axis=1)
            if model is None or valid.sum() < 5:
                sc = feat["mom_7m"].rank(pct=True)
            else:
                pred = pd.Series(np.nan, index=feat.index)
                pred[valid] = model.predict(feat[valid])
                if mode == "xgb_filter":
                    risk = iso.predict(feat[text_feat_cols])
                    pred.loc[valid & (risk == -1)] = -np.inf
                sc = pred.rank(pct=True)
        else:
            sc = pd.Series(dtype=float)
        if sc.empty: continue
        sel = sc.dropna().sort_values(ascending=False).index[:10]
        if len(sel) == 0: continue
        w = pd.Series(0.0, index=cols); w[sel] = 1/len(sel)
        end = rebal[i+1] if i+1 < len(rebal) else px.index[-1]
        days = px.index[(px.index > d) & (px.index <= end)]
        if len(days) == 0: continue
        r = (w.reindex(cols).values * daily_ret.loc[days,:].values).sum(axis=1)
        r = r - ((w - pd.Series(0.0,index=cols)).abs().sum()/2.0) * cost_bps/10000.0
        rets.append(float(r.sum()))
        positions.append((d, list(sel)))
    return pd.Series(rets, index=rebal[:len(rets)]), positions

def summ(s):
    s = s.dropna()
    if len(s) == 0: return {"n":0,"mean":np.nan,"vol":np.nan,"sharpe":np.nan,"cum":np.nan,"maxdd":np.nan}
    if len(s) == 1: return {"n":1,"mean":float(s.mean()),"vol":np.nan,"sharpe":np.nan,"cum":float(s.sum()),"maxdd":np.nan}
    mr = float(s.mean()); v = float(s.std(ddof=0))
    cum = (1+s).cumprod()
    maxdd = float((cum / cum.cummax() - 1).min())
    return {"n":len(s),"mean":mr*100,"vol":v*100,"sharpe":mr/v if v>0 else np.nan,"cum":float(cum.iloc[-1]-1)*100,"maxdd":maxdd*100}

print("\n" + "="*60)
print("动量 + 多源文本因子 walk-forward 回测结果（严格逐月训练）")
print("="*60)
results = {}
for mode, label in [("momentum","纯动量"), ("xgb","XGBoost_双源文本"), ("xgb_filter","XGBoost_双源文本+风控")]:
    r, pos = monthly(mode)
    s = summ(r)
    results[mode] = (r, s, pos)
    print(f"\n[{label}]")
    print(f"  月均收益: {s['mean']:.2f}%  夏普: {s['sharpe']:.2f}  累计: {s['cum']:.1f}%  最大回撤: {s['maxdd']:.1f}%  n={s['n']}")

print(f"\n{'='*60}")
print("最近调仓选股对比")
feat = build_features(rebal[-2])
if not feat.empty:
    mom_sel = feat["mom_7m"].rank(pct=True).sort_values(ascending=False).index[:10]
    valid = feat.notna().all(axis=1)
    pred = pd.Series(np.nan, index=feat.index)
    model = train_model(rebal[-2])
    if model is not None and valid.sum() >= 5:
        pred[valid] = model.predict(feat[valid])
    xgb_sel = pred.rank(pct=True).sort_values(ascending=False).index[:10]
    print(f"纯动量:  {', '.join(mom_sel)}")
    print(f"XGBoost双源: {', '.join(xgb_sel)}")

print(f"\n{'='*60}")
print("结论速览:")
for mode, label in [("momentum","纯动量"), ("xgb","XGBoost_双源文本"), ("xgb_filter","XGBoost_双源文本+风控")]:
    s = results[mode][1]
    print(f"  {label}: 夏普 {s['sharpe']:.2f} | 最大回撤 {s['maxdd']:.1f}% | 累计 {s['cum']:.1f}%")

summary = pd.DataFrame({label: results[mode][1] for mode, label in [("momentum","纯动量"),("xgb","XGBoost_双源文本"),("xgb_filter","XGBoost_双源文本+风控")]}).T
summary.to_csv(OUT / "wf_combined_summary.csv", encoding="utf-8-sig")
