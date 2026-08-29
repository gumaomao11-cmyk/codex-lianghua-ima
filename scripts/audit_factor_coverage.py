import pandas as pd, numpy as np
from pathlib import Path
p = Path(r"F:\even-codex\lianghua+IMA\data\duckdb\aligned_dataset_b_ortho.parquet")
df = pd.read_parquet(p)
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= "2025-01-01"]
df["month"] = df["date"].dt.to_period("M").astype(str)

fac = [c for c in df.columns if c.startswith("factor_") and c.endswith("_ortho")]
raw = ["factor_research_20d","factor_event_3d","factor_news_1d","factor_opinion_1d"]
raw = [c for c in raw if c in df.columns]

print("每月 非零因子覆盖率（原始因子，非 ortho）")
print()
rows=[]
for mo, sub in df.groupby("month"):
    r={"month":mo}
    for c in raw:
        nz = (sub[c].fillna(0)!=0).mean()*100
        r[c.replace("factor_","")] = round(nz,1)
    rows.append(r)
cov = pd.DataFrame(rows)
print(cov.to_string(index=False))
