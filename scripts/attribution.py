import pandas as pd, numpy as np, io
from pathlib import Path

LOG = io.open(r"F:\even-codex\lianghua+IMA\logs\attribution.log","w",encoding="utf-8",buffering=1)
def log(m): LOG.write(str(m)+"\n"); LOG.flush()

res = pd.read_csv(r"F:\even-codex\lianghua+IMA\backtest_output\walkforward_v4_dynamic_results.csv")
res["date"] = pd.to_datetime(res["date"])

bench = pd.read_csv(r"F:\even-codex\panda\backtest\prices_2016.csv", parse_dates=["date"])
cols = [c for c in bench.columns if c.upper() in ("SPY","QQQ")]
bench = bench[["date"]+cols].set_index("date").sort_index()
bret = bench.pct_change()

log("=== 策略 vs 基准 超额归因 ===\n")
for route in ["A","B"]:
    sub = res[res["route"]==route].set_index("date").sort_index()
    r = sub["xgb_dynamic"]
    j = pd.concat([r.rename("strat"), bret], axis=1, join="inner").dropna()
    if j.empty:
        log(f"[{route}] 无重叠区间"); continue
    n = len(j)
    ann = lambda x: x.mean()*252
    sharpe = lambda x: x.mean()/x.std()*np.sqrt(252) if x.std()>0 else np.nan
    log(f"--- Route {route} ---  {j.index.min().date()} ~ {j.index.max().date()}  n={n}")
    log(f"  策略  年化{ann(j['strat'])*100:7.1f}%  夏普{sharpe(j['strat']):5.2f}  累计{((1+j['strat']).prod()-1)*100:7.1f}%")
    for b in cols:
        log(f"  {b:5} 年化{ann(j[b])*100:7.1f}%  夏普{sharpe(j[b]):5.2f}  累计{((1+j[b]).prod()-1)*100:7.1f}%")
    for b in cols:
        cov = np.cov(j["strat"], j[b]); beta = cov[0,1]/cov[1,1]
        alpha_d = j["strat"].mean() - beta*j[b].mean()
        resid = j["strat"] - beta*j[b]
        corr = j["strat"].corr(j[b])
        log(f"  vs {b}: beta={beta:5.2f}  年化alpha={alpha_d*252*100:6.1f}%  "
            f"corr={corr:5.2f}  信息比={alpha_d/resid.std()*np.sqrt(252):5.2f}")
        log(f"          -> beta贡献{beta*ann(j[b])*100:6.1f}%  alpha贡献{alpha_d*252*100:6.1f}%  "
            f"占比 beta{beta*ann(j[b])/ann(j['strat'])*100:5.1f}%")
    log("")
log("DONE")
