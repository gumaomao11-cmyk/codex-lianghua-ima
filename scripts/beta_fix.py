import pandas as pd, numpy as np, io
LOG=io.open(r"F:\even-codex\lianghua+IMA\logs\beta_fix.log","w",encoding="utf-8",buffering=1)
def log(m): LOG.write(str(m)+"\n"); LOG.flush()

r = pd.read_csv(r"F:\even-codex\lianghua+IMA\backtest_output\walkforward_v4_dynamic_results.csv")
r["date"]=pd.to_datetime(r["date"])
b = pd.read_csv(r"F:\even-codex\panda\backtest\prices_2016.csv", parse_dates=["date"]).set_index("date").sort_index()
cols=[c for c in b.columns if c.upper() in ("SPY","QQQ")]

spot = b[cols].pct_change()                                   # t-1 -> t
fwd  = b[cols].shift(-1)/b[cols] - 1.0                        # t   -> t+1  (与策略同定义)

for route in ["A","B"]:
    s = r[r["route"]==route].set_index("date")["xgb_dynamic"].sort_index()
    log(f"===== Route {route} =====")
    for name, bench in [("同期(t-1->t) 错位", spot), ("前瞻(t->t+1) 对齐", fwd)]:
        j = pd.concat([s.rename("st"), bench], axis=1, join="inner").dropna()
        log(f"  [{name}]  n={len(j)}")
        for c in cols:
            cov=np.cov(j["st"], j[c]); beta=cov[0,1]/cov[1,1]
            corr=j["st"].corr(j[c]); al=(j["st"].mean()-beta*j[c].mean())*252*100
            log(f"    vs {c}: beta={beta:6.3f}  corr={corr:6.3f}  年化alpha={al:7.1f}%")
    log("")
log("DONE")
