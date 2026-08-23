import sys
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    try: sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
# -*- coding: utf-8 -*-
"""每日 paper 账户报告：只跟踪策略 10 只持仓，用 $20,000 当基准，附详细指标。"""
import os, sys, json
from pathlib import Path
from datetime import date
import requests, pandas as pd, numpy as np

from _paths import WS, OUT as _OUT, LOG as _LOG
OUT = _OUT
LOG = _OUT / "paper_log.csv"
STATE = _OUT / "paper_state.json"
TARGET = _OUT / "current_holdings_6m_skip1_accel_top10.csv"
SPY_FILE  = Path(os.environ.get("ETFS_REF_FILE") or r"F:\even-codex\panda\backtest\prices_2016.csv")
TARGET_EQ = float(os.environ.get("PAPER_TARGET_EQUITY", "20000"))

# ----- 凭据（与 alpaca_buy 保持一致的加载顺序） -----
def _from_env_file(p):
    out = {}
    if not Path(p).exists(): return out
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s: continue
        k, v = [x.strip() for x in s.split("=", 1)]
        out[k] = v
    return out
KEY = (os.environ.get("ALPACA_KEY_ID") or os.environ.get("ALPACA_API_KEY") or "").strip()
SEC = (os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_SECRET") or "").strip()
EP  = (os.environ.get("ALPACA_ENDPOINT") or "https://paper-api.alpaca.markets").strip()
if not (KEY and SEC):
    ef = _from_env_file(r"F:\even-codex\panda\backtest\alpaca.env")
    KEY = KEY or ef.get("ALPACA_API_KEY","")
    SEC = SEC or ef.get("ALPACA_SECRET_KEY","")
if not (KEY and SEC):
    print("ERROR: 未找到 Alpaca 凭据。"); sys.exit(1)
hdr = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC}

# ----- 拉账户与持仓（对 Alpaca 异常做防护：返回 503/错误时不 KeyError） -----
_acc_r = requests.get(f"{EP}/v2/account", headers=hdr, timeout=15)
try:
    acc = _acc_r.json()
except Exception:
    acc = {"raw": _acc_r.text[:300]}
if not (isinstance(acc, dict) and "equity" in acc and "cash" in acc):
    print(f"ERROR: Alpaca 账户接口返回异常 (HTTP {_acc_r.status_code}): {str(acc)[:300]}")
    sys.exit(1)
_pos_r = requests.get(f"{EP}/v2/positions", headers=hdr, timeout=15)
try:
    pos = _pos_r.json()
except Exception:
    pos = {"raw": _pos_r.text[:300]}
if not isinstance(pos, list):
    print(f"ERROR: Alpaca 持仓接口返回异常 (HTTP {_pos_r.status_code}): {str(pos)[:300]}")
    sys.exit(1)
total_equity = float(acc['equity']); cash = float(acc['cash'])

# ----- 目标清单（只跟踪这些） -----
target_tickers = []
if TARGET.exists():
    target_tickers = sorted(pd.read_csv(TARGET)['ticker'].astype(str).tolist())
target_set = set(target_tickers)
strategy_pos = [p for p in pos if p['symbol'] in target_set]
strategy_nav = sum(float(p['market_value']) for p in strategy_pos)
invested = strategy_nav  # 等价于"投入的策略部分"
non_strat_cash = cash  # 简化：账户里所有现金都视为非策略部分

# ----- 读/写 state -----
ref = pd.read_csv(SPY_FILE, parse_dates=['date']).set_index('date').sort_index()
spy = float(ref['SPY'].iloc[-1]); qqq = float(ref['QQQ'].iloc[-1])
if STATE.exists():
    state = json.loads(STATE.read_text(encoding="utf-8"))
    # 自动修正之前误把账户总额当作策略基准的问题
    if state.get("start_equity", 0) > TARGET_EQ * 1.2:
        print(f"[修复] 之前 start_equity={state['start_equity']} 异常，已修正为 {TARGET_EQ}")
        state["start_equity"] = TARGET_EQ
        STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
else:
    state = {
        "start_date": str(date.today()),
        "start_equity": TARGET_EQ,
        "start_spy": spy,
        "start_qqq": qqq,
        "target_tickers": target_tickers,
    }
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

# ----- 指标 -----
ret_total  = strategy_nav / state["start_equity"] - 1
ret_spy    = spy / state["start_spy"] - 1
ret_qqq    = qqq / state["start_qqq"] - 1
alpha_spy  = ret_total - ret_spy
alpha_qqq  = ret_total - ret_qqq

# ----- 追加日志 -----
log_row = {
    "date": str(date.today()),
    "strategy_nav": round(strategy_nav, 2),
    "total_equity": round(total_equity, 2),
    "cash": round(cash, 2),
    "n_strategy_pos": len(strategy_pos),
    "spy": round(spy, 4), "qqq": round(qqq, 4),
    "strat_pct": round(ret_total*100, 4),
    "spy_pct": round(ret_spy*100, 4),
    "qqq_pct": round(ret_qqq*100, 4),
    "alpha_spy": round(alpha_spy*100, 4),
    "alpha_qqq": round(alpha_qqq*100, 4),
}
if LOG.exists():
    log = pd.read_csv(LOG, parse_dates=["date"])
else:
    log = pd.DataFrame(columns=list(log_row.keys()))
log = log[log["date"] != str(pd.Timestamp(date.today()))]
log = pd.concat([log, pd.DataFrame([log_row])], ignore_index=True)
log["date"] = pd.to_datetime(log["date"])
log = log.sort_values("date")
log.to_csv(LOG, index=False, encoding="utf-8-sig")

# 滚动指标
daily = log["strategy_nav"].pct_change().dropna()
def _last_n_ret(n):
    if len(daily) < n: return None
    return float((1 + daily.tail(n)).prod() - 1)
def _last_n_vol(n):
    if len(daily) < n: return None
    return float(daily.tail(n).std(ddof=1) * np.sqrt(252))
def _last_n_sharpe(n):
    if len(daily) < n: return None
    r = _last_n_ret(n); v = _last_n_vol(n)
    return float((r*252/n) / v) if v and v>0 else None

# 持仓偏离
n = max(len(target_tickers), 1)
target_w = 1.0 / n
holdings = []
for p in strategy_pos:
    mv = float(p['market_value'])
    w = mv/strategy_nav if strategy_nav>0 else 0
    holdings.append((p['symbol'], mv, w*100, (w-target_w)*100, float(p.get('unrealized_plpc', 0))*100))
holdings.sort(key=lambda x: x[2], reverse=True)
max_drift = max((abs(h[3]) for h in holdings), default=0)

# 最大回撤
hwm = log["strategy_nav"].cummax()
dd = (log["strategy_nav"]/hwm - 1)
max_dd = float(dd.min()); cur_dd = float(dd.iloc[-1])

# ----- 打印 -----
days_held = (pd.Timestamp(date.today()) - pd.Timestamp(state["start_date"])).days + 1
print("="*64)
print(f"日报 {date.today()}    (策略已运行 {days_held} 天)")
print("="*64)
print("【策略概况】")
print(f"  策略预算 / 起始权益 : ${state['start_equity']:>12,.2f}")
print(f"  当前策略部分 NAV    : ${strategy_nav:>12,.2f}    (10 只等权持仓市值合计)")
print(f"  当日盈亏            : ${(strategy_nav - float(log.iloc[-2]['strategy_nav'])):>+12,.2f}" if len(log)>=2 else "  当日盈亏            : (无前一日数据)")
pnl = strategy_nav - state['start_equity']
print(f"  累计盈亏            : ${pnl:>+12,.2f}    ({ret_total*100:+.2f}%)")
print(f"  距最高点回撤        : {cur_dd*100:>6.2f}%     历史最大回撤 {max_dd*100:>6.2f}%")
print()
print("【账户总览(非策略部分)】")
print(f"  账户总权益          : ${total_equity:>12,.2f}")
print(f"  账户现金            : ${cash:>12,.2f}")
print(f"  已投入策略          : ${invested:>12,.2f}    占比 {invested/total_equity*100 if total_equity else 0:>5.1f}%")
print()
print("【收益对比(自 {0} 至今)】".format(state["start_date"]))
print(f"  策略               : {ret_total*100:+7.2f}%")
print(f"  SPY                : {ret_spy*100:+7.2f}%   alpha vs SPY  {alpha_spy*100:+6.2f}%")
print(f"  QQQ                : {ret_qqq*100:+7.2f}%   alpha vs QQQ  {alpha_qqq*100:+6.2f}%")
print()
print("【影子策略对比】(自 paper 起始日, 仅供对比, 非实际持仓)")
try:
    import shadow_report
    print(shadow_report.build(start=state["start_date"]))
except Exception as e:
    print("  (影子对比暂不可用: %s)" % e)
print()
print("【滚动指标】")
def _fmt(x, kind="pct"):
    if x is None: return "  (数据不足)"
    if kind=="pct": return f"{x*100:+6.2f}%"
    if kind=="sharpe": return f"{x:5.2f}"
    if kind=="vol": return f"{x*100:5.1f}%"
    return str(x)
print(f"  1日  收益/夏普      : {_fmt(_last_n_ret(1))} / {_fmt(_last_n_sharpe(1),'sharpe')}")
print(f"  5日  收益/夏普/波动 : {_fmt(_last_n_ret(5))} / {_fmt(_last_n_sharpe(5),'sharpe')} / {_fmt(_last_n_vol(5),'vol')}")
print(f"  21日 收益/夏普/波动 : {_fmt(_last_n_ret(21))} / {_fmt(_last_n_sharpe(21),'sharpe')} / {_fmt(_last_n_vol(21),'vol')}")
print(f"  全部 收益/夏普/波动 : {_fmt(ret_total)} / {_fmt((ret_total)*0+np.nan if len(daily)<2 else daily.mean()/daily.std(ddof=1)*np.sqrt(252),'sharpe')} / {_fmt(_last_n_vol(len(daily)),'vol')}")
print()
print(f"【持仓明细(等权目标 {target_w*100:.1f}%)】")
print(f"  {'代码':<6} {'市值':>10} {'权重':>6} {'偏离':>7} {'浮盈%':>8}")
for sym, mv, w, drift, upl in holdings:
    flag = " ⚠" if abs(drift) > 1.0 else ""
    print(f"  {sym:<6} ${mv:>9,.2f} {w:>5.2f}% {drift:>+6.2f}pp {upl:>+7.2f}%{flag}")
print(f"  持仓最大偏离等权: {max_drift:.2f}pp  (>{5.0}pp 建议调仓)")
print()
print("【回测预期(全期)】  年化 45.7%  |  夏普 1.33  |  最大回撤 -40%")
print("【实盘期行动建议】")
if max_drift > 5:
    print(f"  ⚠ 持仓偏离等权 {max_drift:.1f}pp, 建议尽快调仓")
elif max_dd < -0.40:
    print(f"  ⚠ 累计回撤 {max_dd*100:.1f}%, 超过回测预期-40%, 考虑切到 top20 或加波动率目标")
else:
    print(f"  ✓ 节奏正常, 继续按月频调仓")
print()
print(f"日志: {LOG}")

# ----- 卫星仓(机会仓,单独标记,不进核心回测) -----
sat_set = set(); sat_budget = 0.0
if (OUT / "satellite_targets.json").exists():
    try:
        _s = json.loads((OUT / "satellite_targets.json").read_text(encoding="utf-8"))
        sat_set = set(_s.get("tickers", [])); sat_budget = float(_s.get("budget", 0.0))
    except Exception:
        sat_set = set()
satellite_pos = [pp for pp in pos if pp["symbol"] in sat_set and pp["symbol"] not in target_set]
sat_nav = sum(float(pp["market_value"]) for pp in satellite_pos)
sat_pnl = sat_nav - sat_budget
print(); print(f"【卫星仓 (实验机会仓, 单独标记, 不计入策略回测)】")
if sat_set:
    print(f"  卫星预算 : ${sat_budget:,.2f}  (≤主仓15%)")
    print(f"  卫星市值 : ${sat_nav:,.2f}   浮动 {sat_pnl:+,.2f} ({sat_pnl/sat_budget*100 if sat_budget else 0:+.2f}%)")
    for q in satellite_pos:
            sym_q = q["symbol"]; mv_q = float(q["market_value"]); up_q = float(q.get("unrealized_plpc", 0)) * 100
            print(f"    {sym_q:<6} ${mv_q:>9,.2f}  浮盈 {up_q:+.2f}%")
else:
    print("  无卫星仓(尚未配置)")

