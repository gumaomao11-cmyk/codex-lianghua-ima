# -*- coding: utf-8 -*-
"""动量+文本因子回测：TSM/TNA/Disagreement + XGBoost/线性复合 + Isolation Forest 风控。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
import xgboost as xgb
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path(r"F:\even-codex\us-stock-data")
OUT  = Path(r"F:\even-codex\lianghua+IMA\backtest_output")
SRC  = "浑水调研Plus"

# ---- load prices ----
px = pd.read_csv(DATA/"prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
recent = px.loc[px.index >= pd.Timestamp("2025-01-01")]
px = px.loc[:, recent.notna().sum() >= 150]
daily_ret = px.pct_change().fillna(0.0)
cols = list(px.columns)
ml = px.resample("ME").last()
mom = (ml.shift(1)/ml.shift(7)-1.0).replace([np.inf,-np.inf], np.nan)
rebal = list(ml.truncate("2025-09-30", px.index[-1]).index)

# ---- load text factors ----
text = pd.read_csv(OUT/f"text_sentiment_lexicon_{SRC}.csv", encoding="utf-8-sig")
text["date"] = pd.to_datetime(text["date"])
print(f"text rows: {len(text)}, tickers: {text['ticker'].nunique()}, date range: {text['date'].min().date()} ~ {text['date'].max().date()}")

def rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def build_features(d):
    """为调仓日 d 构建每个 ticker 的特征。"""
    feats = pd.DataFrame(index=cols)
    # price features
    if d in mom.index:
        feats["mom_7m"] = mom.loc[d]
    else:
        feats["mom_7m"] = np.nan
    # 10-day momentum
    d_idx = px.index.get_indexer([d], method="nearest")[0]
    if d_idx >= 10:
        p10 = px.iloc[d_idx-10]
        p0 = px.iloc[d_idx]
        feats["mom_10d"] = (p0 / p10 - 1.0).replace([np.inf,-np.inf], np.nan)
    else:
        feats["mom_10d"] = np.nan
    # 20-day momentum
    if d_idx >= 20:
        p20 = px.iloc[d_idx-20]
        feats["mom_20d"] = (p0 / p20 - 1.0).replace([np.inf,-np.inf], np.nan)
    else:
        feats["mom_20d"] = np.nan
    # RSI
    if d_idx >= 14:
        rsi_s = rsi(px.iloc[:d_idx+1], 14).iloc[-1]
        feats["rsi_14"] = rsi_s
    else:
        feats["rsi_14"] = np.nan
    # 20-day volatility
    if d_idx >= 20:
        vol = daily_ret.iloc[d_idx-19:d_idx+1].std() * np.sqrt(252)
        feats["vol_20d"] = vol
    else:
        feats["vol_20d"] = np.nan

    # text features
    window_3 = text[(text["date"] < d) & (text["date"] >= d - pd.Timedelta(days=3))]
    window_20 = text[(text["date"] < d) & (text["date"] >= d - pd.Timedelta(days=20))]

    # TSM: short-term sentiment minus medium-term sentiment
    sent_3 = window_3.groupby("ticker")["sentiment"].mean()
    sent_20 = window_20.groupby("ticker")["sentiment"].mean()
    feats["tsm"] = (sent_3.reindex(cols) - sent_20.reindex(cols)).fillna(0)

    # TNA: sentiment * novelty (latest value)
    latest = text[text["date"] < d].sort_values("date").groupby("ticker").last()
    tna = (latest["sentiment"] * latest["novelty"]).reindex(cols).fillna(0)
    feats["tna"] = tna

    # Disagreement: variance of sentiment in 20d window
    disc = window_20.groupby("ticker")["sentiment"].var().reindex(cols).fillna(0)
    feats["disagreement"] = disc

    # Risk flag: any risk event in 20d window
    risk = window_20.groupby("ticker")["risk_flag"].max().reindex(cols).fillna(0)
    feats["risk_flag"] = risk

    # text coverage: count of mentions in 20d
    cov = window_20.groupby("ticker").size().reindex(cols).fillna(0)
    feats["text_cov"] = cov

    return feats.dropna(how="all")

def build_labels():
    """构建用于训练 XGBoost 的标签：下月收益。"""
    X_rows = []
    y_vals = []
    for i, d in enumerate(rebal):
        if i+1 >= len(rebal): continue
        end = rebal[i+1]
        days = px.index[(px.index > d) & (px.index <= end)]
        if len(days) == 0: continue
        period_ret = daily_ret.loc[days].sum()
        feat = build_features(d)
        if feat.empty: continue
        f = feat.reset_index().rename(columns={"index":"ticker"})
        f = f[f["ticker"].isin(period_ret.index)]
        if len(f) < 5: continue
        f = f.set_index("ticker")
        X_rows.append(f)
        y_vals.append(period_ret.reindex(f.index))
    if not X_rows: return None, None
    X = pd.concat(X_rows)
    y = pd.concat(y_vals)
    return X, y

print("building training data for XGBoost...")
X_train, y_train = build_labels()
if X_train is not None:
    print(f"training samples: {len(X_train)}, features: {list(X_train.columns)}")
    # train XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {"objective":"reg:squarederror", "max_depth":3, "eta":0.1, "subsample":0.7, "colsample_bytree":0.7}
    xgb_model = xgb.train(params, dtrain, num_boost_round=50)
    print("XGBoost feature importance:")
    print(xgb_model.get_score(importance_type="gain"))
else:
    xgb_model = None
    print("not enough data for XGBoost")

# ---- Isolation Forest risk model (fit on text features) ----
print("training Isolation Forest on text features...")
text_features_for_if = text[["sentiment","novelty","risk_flag"]].dropna()
if len(text_features_for_if) > 10:
    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    iso.fit(text_features_for_if)
    text["anomaly_score"] = iso.decision_function(text_features_for_if)
else:
    text["anomaly_score"] = 0.0

# ---- backtest functions ----
def portfolio_score(feat, mode="momentum"):
    """mode: momentum / linear / xgb / risk_off"""
    f = feat.copy()
    if f.empty: return pd.Series(dtype=float)
    if mode == "momentum":
        return f["mom_7m"].rank(pct=True)
    elif mode == "linear":
        # weighted composite
        w = {"mom_7m":0.5, "tsm":0.15, "tna":0.15, "disagreement":-0.1, "rsi_14":0.05, "vol_20d":-0.05}
        sc = pd.Series(0.0, index=f.index)
        for k, weight in w.items():
            if k in f.columns:
                sc += weight * f[k].rank(pct=True).fillna(0.5)
        return sc
    elif mode == "xgb" and xgb_model is not None:
        dtest = xgb.DMatrix(f)
        pred = xgb_model.predict(dtest)
        return pd.Series(pred, index=f.index)
    elif mode == "risk_off":
        # linear composite but zero out stocks with high anomaly/risk
        w = {"mom_7m":0.5, "tsm":0.15, "tna":0.15, "disagreement":-0.1, "rsi_14":0.05, "vol_20d":-0.05}
        sc = pd.Series(0.0, index=f.index)
        for k, weight in w.items():
            if k in f.columns:
                sc += weight * f[k].rank(pct=True).fillna(0.5)
        # penalize risk
        if "risk_flag" in f.columns:
            sc = sc.where(f["risk_flag"] == 0, sc - 1.0)
        return sc
    return pd.Series(dtype=float)

def train_xgb_up_to(d):
    """walk-forward: 用 d 之前的所有调仓期数据训练 XGBoost"""
    X_rows, y_vals = [], []
    for i, dd in enumerate(rebal):
        if dd >= d: break
        if i+1 >= len(rebal): continue
        end = rebal[i+1]
        days = px.index[(px.index > dd) & (px.index <= end)]
        if len(days) == 0: continue
        period_ret = daily_ret.loc[days].sum()
        feat = build_features(dd)
        if feat.empty: continue
        f = feat.reset_index().rename(columns={"index":"ticker"})
        f = f[f["ticker"].isin(period_ret.index)]
        if len(f) < 5: continue
        f = f.set_index("ticker")
        X_rows.append(f)
        y_vals.append(period_ret.reindex(f.index))
    if not X_rows: return None
    X = pd.concat(X_rows)
    y = pd.concat(y_vals)
    if len(X) < 30: return None
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective":"reg:squarederror", "max_depth":3, "eta":0.1, "subsample":0.7, "colsample_bytree":0.7}
    return xgb.train(params, dtrain, num_boost_round=30)

def monthly(mode="momentum", cost_bps=5):
    rets = []
    positions = []
    for i, d in enumerate(rebal):
        if d not in mom.index: continue
        feat = build_features(d)
        if feat.empty: continue
        if mode == "momentum":
            sc = feat["mom_7m"].rank(pct=True)
        elif mode == "linear":
            w = {"mom_7m":0.5, "tsm":0.15, "tna":0.15, "disagreement":-0.1, "rsi_14":0.05, "vol_20d":-0.05}
            sc = pd.Series(0.0, index=feat.index)
            for k, weight in w.items():
                if k in feat.columns:
                    sc += weight * feat[k].rank(pct=True).fillna(0.5)
        elif mode == "xgb":
            # walk-forward train
            model = train_xgb_up_to(d)
            if model is None or feat["text_cov"].max() == 0:
                sc = feat["mom_7m"].rank(pct=True)  # fallback
            else:
                dtest = xgb.DMatrix(feat)
                pred = model.predict(dtest)
                sc = pd.Series(pred, index=feat.index)
        elif mode == "risk_off":
            w = {"mom_7m":0.5, "tsm":0.15, "tna":0.15, "disagreement":-0.1, "rsi_14":0.05, "vol_20d":-0.05}
            sc = pd.Series(0.0, index=feat.index)
            for k, weight in w.items():
                if k in feat.columns:
                    sc += weight * feat[k].rank(pct=True).fillna(0.5)
            if "risk_flag" in feat.columns:
                sc = sc.where(feat["risk_flag"] == 0, sc - 1.0)
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
print("动量+文本因子回测结果")
print("="*60)

results = {}
for mode, label in [("momentum","纯动量"), ("linear","线性复合"), ("xgb","XGBoost复合"), ("risk_off","线性+风控")]:
    r, pos = monthly(mode)
    s = summ(r)
    results[mode] = (r, s, pos)
    print(f"\n[{label}]")
    print(f"  月均收益: {s['mean']:.2f}%  夏普: {s['sharpe']:.2f}  累计: {s['cum']:.1f}%  最大回撤: {s['maxdd']:.1f}%  n={s['n']}")

# recent selection comparison
print(f"\n{'='*60}")
print("最近调仓选股对比")
d = pd.Timestamp("2026-07-31")
if d in mom.index:
    feat = build_features(d)
    if not feat.empty:
        mom_sel = feat["mom_7m"].rank(pct=True).sort_values(ascending=False).index[:10]
        lin_sel = (0.5*feat["mom_7m"].rank(pct=True) + 0.15*feat["tsm"].rank(pct=True).fillna(0.5) + 0.15*feat["tna"].rank(pct=True).fillna(0.5) - 0.1*feat["disagreement"].rank(pct=True).fillna(0.5) + 0.05*feat["rsi_14"].rank(pct=True).fillna(0.5) - 0.05*feat["vol_20d"].rank(pct=True).fillna(0.5)).sort_values(ascending=False).index[:10]
        risk_sel = lin_sel
        print(f"纯动量:  {', '.join(mom_sel)}")
        print(f"线性复合: {', '.join(lin_sel)}")
        print(f"+风控:   {', '.join(risk_sel)}")

print(f"\n{'='*60}")
print("结论速览:")
for mode, label in [("momentum","纯动量"), ("linear","线性复合"), ("xgb","XGBoost复合"), ("risk_off","线性+风控")]:
    s = results[mode][1]
    print(f"  {label}: 夏普 {s['sharpe']:.2f} | 最大回撤 {s['maxdd']:.1f}% | 累计 {s['cum']:.1f}%")

