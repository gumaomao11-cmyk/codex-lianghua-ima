import pandas as pd, numpy as np, io
LOG=io.open(r"F:\even-codex\lianghua+IMA\logs\final.log","w",encoding="utf-8",buffering=1)
def log(m): LOG.write(str(m)+"\n"); LOG.flush()

r=pd.read_csv(r"F:\even-codex\lianghua+IMA\backtest_output\walkforward_v4_dynamic_results.csv")
r["date"]=pd.to_datetime(r["date"])
b=pd.read_csv(r"F:\even-codex\panda\backtest\prices_2016.csv",parse_dates=["date"]).set_index("date").sort_index()
cols=[c for c in b.columns if c.upper() in ("SPY","QQQ")]
fwd=b[cols].shift(-1)/b[cols]-1.0     # 与策略同为前瞻收益

log("="*72)
log("最终结论：动量+星球文本 动态调仓策略 (2025-07 ~ 2026-08)")
log("="*72)

for route in ["A","B"]:
    s=r[r["route"]==route].set_index("date")["xgb_dynamic"].sort_index()
    j=pd.concat([s.rename("st"),fwd],axis=1,join="inner").dropna()
    ann=lambda x:x.mean()*252*100
    shp=lambda x:x.mean()/x.std()*np.sqrt(252)
    cum=lambda x:((1+x).prod()-1)*100
    mdd=lambda x:((1+x).cumprod()/(1+x).cumprod().cummax()-1).min()*100

    log(f"\n--- Route {route} ({'纯净单因子' if route=='A' else '四因子组合'}) ---")
    log(f"  样本 {len(s)} 交易日   {s.index.min().date()} ~ {s.index.max().date()}")
    log(f"  年化收益 {ann(s):6.1f}%    夏普 {shp(s):5.2f}    累计 {cum(s):6.1f}%    最大回撤 {mdd(s):6.1f}%")
    for c in cols:
        cov=np.cov(j['st'],j[c]); beta=cov[0,1]/cov[1,1]
        al=(j['st'].mean()-beta*j[c].mean())*252*100
        resid=j['st']-beta*j[c]
        log(f"  vs {c}: beta {beta:5.2f}  相关 {j['st'].corr(j[c]):5.2f}  "
            f"年化alpha {al:6.1f}%  信息比 {al/100/(resid.std()*np.sqrt(252)):5.2f}")
        log(f"          收益拆解 -> beta贡献 {beta*ann(j[c]):5.1f}%  +  alpha {al:5.1f}%")

    # 集中度：剔除最好的N天
    log(f"  收益集中度:")
    for n in [1,3,5]:
        idx=s.nlargest(n).index
        cut=s.drop(idx)
        log(f"    剔除最佳{n}天 -> 年化 {ann(cut):6.1f}%  夏普 {shp(cut):5.2f}  累计 {cum(cut):6.1f}%")
    top=s.nlargest(3)
    log(f"    最佳3日: " + ", ".join(f"{d.date()}({v*100:+.1f}%)" for d,v in top.items()))

log("\n" + "="*72)
log("基准（同期前瞻口径）")
for c in cols:
    x=fwd[c].loc["2025-07-01":"2026-08-26"].dropna()
    log(f"  {c}: 年化 {x.mean()*252*100:5.1f}%  夏普 {x.mean()/x.std()*np.sqrt(252):5.2f}  累计 {((1+x).prod()-1)*100:5.1f}%")
