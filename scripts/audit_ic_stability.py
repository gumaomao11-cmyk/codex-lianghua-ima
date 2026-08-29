import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path

df = pd.read_parquet(Path(r"F:\even-codex\lianghua+IMA\data\duckdb\aligned_dataset_b_ortho.parquet"))
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= "2025-01-01"]

fac = ["factor_research_20d_ortho","factor_event_3d_ortho",
       "factor_news_1d_ortho","factor_opinion_1d_ortho"]
fac = [c for c in fac if c in df.columns]
fwd = "ret_21d"

def daily_ic(sub, f):
    ics=[]
    for d, g in sub.groupby("date"):
        g = g[[f, fwd]].dropna()
        g = g[g[f]!=0]                 # 只在真正有信号的截面上算
        if len(g) < 5: continue
        ic,_ = spearmanr(g[f], g[fwd])
        if not np.isnan(ic): ics.append(ic)
    return np.array(ics)

halves = {
 "2025H1 (2025-01~06)": ("2025-01-01","2025-06-30"),
 "2025H2 (2025-07~12)": ("2025-07-01","2025-12-31"),
 "2026H1 (2026-01~06)": ("2026-01-01","2026-06-30"),
 "2026Q3 (2026-07~08)": ("2026-07-01","2026-08-31"),
}

print(f"{fwd} 上的 Rank IC —— 仅统计有信号的截面 (>=5只)")
print()
hdr=f'{"因子":30}' + "".join(f'{k:>22}' for k in halves)
print(hdr); print("-"*len(hdr))
for f in fac:
    line=f'{f.replace("factor_","").replace("_ortho",""):30}'
    for k,(a,b) in halves.items():
        sub = df[(df["date"]>=a)&(df["date"]<=b)]
        ics = daily_ic(sub,f)
        if len(ics)==0:
            line += f'{"无数据":>22}'
        else:
            ir = ics.mean()/ics.std() if ics.std()>0 else 0
            line += f'{f"IC{ics.mean():+.3f} IR{ir:+.2f} n{len(ics)}":>22}'
    print(line)
