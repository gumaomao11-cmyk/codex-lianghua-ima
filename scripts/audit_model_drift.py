import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

CACHE = Path(r"F:\even-codex\us-stock-data\知识星球\zsxq_19_26_granular_cache.jsonl")
recs = []
for ln in CACHE.read_text(encoding="utf-8").splitlines():
    if not ln.strip(): continue
    try: o = json.loads(ln)
    except Exception: continue
    ct = (o.get("create_time") or "")[:7]
    if len(ct) != 7: continue
    ev = o.get("events") or []
    sents = [e.get("sentiment_score") for e in ev
             if isinstance(e.get("sentiment_score"),(int,float))]
    recs.append({"month": ct, "model": o.get("model") or "unknown",
                 "n_ev": len(ev), "mean_sent": (sum(sents)/len(sents)) if sents else None})
df = pd.DataFrame(recs)

print("每月主力模型 / 事件产出强度 / 平均情绪")
print()
g = df.groupby("month")
out = []
for mo, sub in g:
    top = sub["model"].value_counts()
    share = top.iloc[0]/len(sub)*100
    out.append({"month": mo, "n": len(sub), "主力模型": top.index[0][:30],
                "占比%": round(share,1),
                "事件/条": round(sub["n_ev"].mean(),2),
                "均情绪": round(sub["mean_sent"].mean(),3) if sub["mean_sent"].notna().any() else None})
print(pd.DataFrame(out).to_string(index=False))
