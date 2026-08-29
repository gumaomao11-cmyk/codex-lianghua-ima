import pandas as pd, io, numpy as np
LOG=io.open(r"F:\even-codex\lianghua+IMA\logs\day_probe.log","w",encoding="utf-8",buffering=1)
def log(m): LOG.write(str(m)+"\n"); LOG.flush()

px = pd.read_csv(r"F:\even-codex\us-stock-data\prices.csv", parse_dates=["date"]).set_index("date").sort_index()
ret = px.pct_change()

for d in ["2026-08-18","2026-08-17","2026-08-14"]:
    if pd.Timestamp(d) in ret.index:
        row = ret.loc[pd.Timestamp(d)].dropna()
        log(f"--- {d} ---  n={len(row)}  median={row.median()*100:.2f}%  mean={row.mean()*100:.2f}%")
        log("  top10: " + ", ".join(f"{k}{v*100:+.1f}%" for k,v in row.nlargest(10).items()))
        log(f"  >20% count: {(row>0.20).sum()}  >50%: {(row>0.50).sum()}")
        big = row[row>0.20]
        if len(big): log("  suspicious: " + ", ".join(f"{k}{v*100:+.0f}%" for k,v in big.items()))
        log("")

log("=== 检查 08-18 附近价格是否有跳变 ===")
sus = ret.loc["2026-08-10":"2026-08-20"]
for c in sus.columns:
    s = sus[c].dropna()
    if len(s) and s.abs().max() > 0.25:
        log(f"  {c}: " + ", ".join(f"{i.date()}:{v*100:+.0f}%" for i,v in s.items() if abs(v)>0.15))
        log(f"      prices: {px[c].loc['2026-08-12':'2026-08-20'].round(2).to_dict()}")
