# -*- coding: utf-8 -*-
"""板块分散版 top10（相关性聚类代理行业）
数据里没有官方 GICS 行业，用近2年日收益率相关性做聚类当作“板块族”：
同一簇=高度同涨同跌（例如半导体/AI 会聚成一簇），选股时每簇最多 max_k 只。
- 动量分：与原策略一致（6m-skip1）
- 输出: current_holdings_6m_skip1_top10_div.csv
- 只出方案，不自动下单。
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import AgglomerativeClustering

DATA = Path(os.environ.get("STOCK_DATA_DIR") or r"F:\even-codex\us-stock-data")
from _paths import OUT

TOP = 10
MAX_K = 3          # 每板块(簇)最多 3 只
K_CLUSTERS = 12    # 大致对应 GICS 12 大族群
CAPITAL = 20000.0
LOOK_DAYS = 504    # 近2年用于相关性聚类

px = pd.read_csv(DATA / "prices.csv", index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")
px = px.loc[:, px.count() >= 2400]

def ml(x): return x.resample("ME").last()
def mom(px, p=6, k=1):
    m = ml(px); return m.shift(k) / m.shift(p + k) - 1

scores = mom(px)
last_label = scores.index[scores.index <= px.index[-1].to_period("M").to_timestamp()]
sig = last_label[-1]
sc = scores.loc[sig].dropna()
cands = sc.sort_values(ascending=False)

# ---- 相关性聚类（用近2年日收益）----
rt = px.pct_change().tail(LOOK_DAYS).dropna(how="all")
rt = rt.dropna(axis=1, how="any") if rt.shape[1] > 200 else rt
corr = rt.corr().clip(-1, 1).fillna(0.0)
dist = 1.0 - corr
syms = list(corr.columns)
model = AgglomerativeClustering(n_clusters=K_CLUSTERS, metric="precomputed", linkage="average")
labels = model.fit_predict(dist.values)
lab = dict(zip(syms, labels))

# 簇内成员（用于人工识别“这是什么族”）
mem = {}
for t, l in lab.items():
    mem.setdefault(l, []).append(t)
cluster_name = {}
for l, ms in mem.items():
    top_m = sorted(ms, key=lambda t: (px[t].iloc[-1] if pd.notna(px[t].iloc[-1]) else 0), reverse=True)[:4]
    cluster_name[l] = "+".join(top_m)

# ---- 纯动量 top10（对照）----
pure = cands.index[:TOP].tolist()

# ---- 分散选股：每簇最多 MAX_K ----
selected = []
cnt = {}
for t in cands.index:
    if t not in lab:
        continue
    l = lab[t]
    if cnt.get(l, 0) >= MAX_K:
        continue
    selected.append(t); cnt[l] = cnt.get(l, 0) + 1
    if len(selected) >= TOP:
        break

comp = []
rank = 0
for t in selected:
    rank += 1
    if t not in px.columns or t not in lab:
        continue
    price = float(px.iloc[-1][t])
    if not np.isfinite(price) or price <= 0:
        continue
    l = lab[t]
    alloc = CAPITAL / TOP
    comp.append(dict(rank=rank, ticker=t, sector=f"板块{l}", sector_peers=cluster_name[l],
                     momentum=float(cands[t]), signal_date=str(pd.Timestamp(sig).date()),
                     weight=1.0/TOP, price=price, alloc_usd=alloc, shares=alloc/price))
dfd = pd.DataFrame(comp)
out_csv = OUT / "current_holdings_6m_skip1_top10_div.csv"
dfd.to_csv(out_csv, index=False, encoding="utf-8-sig")

print("="*84)
print(f"板块分散版 top{TOP}  (相关性聚类 K={K_CLUSTERS}, 每板块最多 {MAX_K} 只)  信号日={pd.Timestamp(sig).date()}")
print("="*84)
print(dfd[["rank","ticker","sector","sector_peers","momentum","price","alloc_usd","shares"]].round(4).to_string(index=False))
print("-"*84)
from collections import Counter as _C
pc = _C(lab[t] for t in pure if t in lab); dc = _C(lab[t] for t in selected)
print("纯动量 top10 板块分布:")
for l,n in pc.most_common():
    print(f"  板块{l:<3} {cluster_name[l]:<30} {n}只")
print("分散版 top10 板块分布:")
for l,n in sorted(dc.items(), key=lambda x:-x[1]):
    print(f"  板块{l:<3} {cluster_name[l]:<30} {n}只")
print("\nsaved:", out_csv)
