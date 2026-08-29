import pandas as pd, numpy as np, io
LOG=io.open(r"F:\even-codex\lianghua+IMA\logs\impact.log","w",encoding="utf-8",buffering=1)
def log(m): LOG.write(str(m)+"\n"); LOG.flush()

px = pd.read_csv(r"F:\even-codex\us-stock-data\prices.csv", parse_dates=["date"]).set_index("date").sort_index()
ret = px.pct_change()

log("=== 全样本 |单日涨跌| > 50% 的记录（疑似数据错误）===")
cnt=0
for c in ret.columns:
    s = ret[c].dropna()
    ext = s[s.abs() > 0.5]
    if len(ext):
        cnt += len(ext)
        for i,v in ext.items():
            log(f"  {c:6} {i.date()} {v*100:+8.0f}%   px {px[c].shift(1).loc[i]:.2f} -> {px[c].loc[i]:.2f}")
log(f"总计 {cnt} 条异常\n")

log("=== 剔除 2026-08-18 单日后 Route A/B 表现 ===")
r = pd.read_csv(r"F:\even-codex\lianghua+IMA\backtest_output\walkforward_v4_dynamic_results.csv")
r["date"]=pd.to_datetime(r["date"])
for route in ["A","B"]:
    a = r[r["route"]==route].set_index("date")["xgb_dynamic"].sort_index()
    b = a.drop(pd.Timestamp("2026-08-18"), errors="ignore")
    f = lambda x: (x.mean()*252*100, x.mean()/x.std()*np.sqrt(252), ((1+x).prod()-1)*100)
    o1, o2 = f(a), f(b)
    log(f"Route {route}")
    log(f"  含异常日 : 年化{o1[0]:6.1f}%  夏普{o1[1]:5.2f}  累计{o1[2]:6.1f}%")
    log(f"  剔除后   : 年化{o2[0]:6.1f}%  夏普{o2[1]:5.2f}  累计{o2[2]:6.1f}%")
    log(f"  夏普下降 : {o1[1]-o2[1]:5.2f}   累计下降 {o1[2]-o2[2]:5.1f}pp")
