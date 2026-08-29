import pandas as pd, numpy as np, io
LOG=io.open(r"F:\even-codex\lianghua+IMA\logs\sanity.log","w",encoding="utf-8",buffering=1)
def log(m): LOG.write(str(m)+"\n"); LOG.flush()

t = pd.read_csv(r"F:\even-codex\lianghua+IMA\backtest_output\walkforward_v4_dynamic_trades.csv")
log("trades columns: " + str(list(t.columns)))
log(f"rows={len(t)}")
log(t.head(12).to_string())
log("")
r = pd.read_csv(r"F:\even-codex\lianghua+IMA\backtest_output\walkforward_v4_dynamic_results.csv")
r["date"]=pd.to_datetime(r["date"])
a = r[r["route"]=="A"].set_index("date")["xgb_dynamic"]
log(f"return series: n={len(a)} mean={a.mean():.5f} std={a.std():.5f}")
log(f"  min={a.min():.4f} max={a.max():.4f}")
log(f"  autocorr(1)={a.autocorr(1):.3f}")
log(f"  positive days={(a>0).mean()*100:.1f}%")
log(f"  |ret|>5% days={(a.abs()>0.05).sum()}")
log("\ntop 8 single-day gains:")
log(a.nlargest(8).to_string())
