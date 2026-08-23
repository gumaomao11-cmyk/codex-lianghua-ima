# ⬛ 本仓库 = IMA 因子实验副本

> 此仓库是从原项目 `lianghua2` 独立出来的**实验策略**，核心差异是叠加 IMA 知识库因子。
> 原策略仓库保持不变：https://github.com/gumaomao11-cmyk/codex--
> 实验说明见：[README_IMA.md](README_IMA.md)

---
# 美股量化 · 非日内动量策略（激进前 10 只版）

> **资金量 2 万美元起步 · Alpaca Paper 实盘跟踪中 · 全自动 GitHub Actions 运行**
> 策略核心：6 个月动量（跳过最近 1 月）+ 严格回测筛选 + 每周/每月再平衡。

---

## 📌 目录速查

```
lianghua2/
├── 策略本体
│   ├── current_momentum_list.py     # 当前 top 候选清单
│   ├── backtest_momentum_v2.py      # 6m_skip1 动量回测主脚本
│   ├── optimize_v3.py / v4 / v5     # 参数扫描（动量窗口 / 持仓数 / 波动率）
│   ├── walkforward_v6.py            # 滚动样本外验证
│   ├── cost_sensitivity.py          # 成本敏感度（10bps 假设）
│   ├── correct_metrics.py           # 指标计算（夏普 / 回撤 / alpha）
│   ├── regime_filters.py            # 市场环境过滤
│   └── shadow_compare.py            # 影子策略对比（不真下）
│
├── 账户跟踪
│   ├── alpaca_buy.py                # 买入 / 调仓（Paper & Live）
│   ├── paper_tracker.py             # 每日 paper 日报
│   └── current_holdings.py          # 月末生成新 top10
│
├── 数据维护
│   ├── update_prices_wide.py        # 增量更新 515 只股票宽表
│   ├── update_etf_ref.py            # 增量更新 8 个 ETF 参照表
│   └── update_prices.py             # 旧版（raw/ 模式）
│
├── 自动运行
│   ├── auto_run.py                  # 总编排：日 / 周 / 月 / 季
│   ├── mailer.py                    # QQ 邮箱 SMTP
│   ├── weekly_report_pdf.py         # 周日 PDF 周报
│   └── .github/workflows/daily.yml  # GitHub Actions 定时
│
├── 文档
│   ├── README.md                    # 本文件
│   ├── backtest_output/manual_checklist.md
│   ├── backtest_output/paper_3month_plan.md
│   ├── backtest_output/aggressive_strategy_spec.md
│   └── backtest_output/optimization_report.md
│
├── 数据快照（commit 进 GitHub，cloud 端用）
│   └── data/
│       ├── prices.csv               # 515 只股票宽表 ~8MB
│       ├── master_tickers.csv
│       ├── summary.csv
│       └── etf-ref.csv              # SPY/QQQ/IWM 等 8 个 ETF
│
└── 输出
    ├── backtest_output/
    │   ├── paper_log.csv            # paper 跟踪日志
    │   ├── paper_state.json
    │   ├── weekly_report_YYYYMMDD.pdf
    │   └── current_holdings_6m_skip1_top10.csv
    └── logs/                        # 自动运行的子日志
```

---

## 🚀 第一次部署（10 分钟）

### 1. 装 Python 依赖
```powershell
cd F:\even-codex\panda
.\.venv\Scripts\Activate.ps1
pip install requests pandas numpy reportlab matplotlib
```

### 2. 配置两个本地 .env 文件（不要 commit）
**`F:\even-codex\panda\backtest\alpaca.env`**
```
ALPACA_API_KEY=你的PaperKey
ALPACA_SECRET_KEY=你的PaperSecret
ALPACA_ENDPOINT=https://paper-api.alpaca.markets
```

**`F:\even-codex\lianghua2\mail.env`**
```
QQ_MAIL_AUTH_CODE=你的QQ授权码
QQ_MAIL_FROM=869357594@qq.com
QQ_MAIL_TO=869357594@qq.com
```

### 3. 跑一次手动回测，确认基线
```powershell
cd F:\even-codex\lianghua2
python backtest_momentum_v2.py
```
预期：夏普 0.8-1.5、最大回撤 -35%~-45%、年化 15-30%。

### 4. 推送 GitHub 并配 4 个 Secret
```powershell
git add -A
git status          # 确认 mail.env / alpaca.env 不在列表里
git commit -m "feat: 量化策略 + GitHub Actions"
git push origin main
```

打开 `Settings → Secrets and variables → Actions → New repository secret`，**4 个值**：
| Name | 说明 |
|---|---|
| `ALPACA_API_KEY`     | Alpaca Paper Key（用全新生成的，别用贴过聊天的） |
| `ALPACA_SECRET_KEY`  | Alpaca Paper Secret |
| `QQ_MAIL_AUTH_CODE`  | QQ 邮箱授权码（用全新生成的） |
| `WECHAT_SEND_KEY`    | Server酱 SendKey（见下文） |

### 5. 配 Server酱（微信推送）
1. 微信扫码关注「**Server酱**」公众号（https://sct.ftqq.com/）
2. 登录拿到 SendKey（`SCT...` 开头）
3. 把 SendKey 填到 GitHub Secret `WECHAT_SEND_KEY`

免费额度 5 条/天，跑挂时告警够用。

### 6. 手动触发一次
`Actions → daily.yml → Run workflow`，看 5 件事：
- ✅ `update-data` job 完成
- ✅ `run-strategy` job 完成
- 📧 QQ 邮箱收到 `[策略 日报] ...` 邮件
- 📂 仓库 `data/prices.csv` 日期更新
- 📱 微信没收到推送（说明没失败，正常）

---

## ⏰ 自动节奏

| 时间 (北京) | 做了什么 |
|---|---|
| 每天 05:30 | 抓 515 只股票 + 8 个 ETF 新数据，commit/push |
| 每天 06:00 | 跑 paper_tracker，把日报发到 QQ 邮箱 |
| 每周日 06:00 | 额外跑 shadow_compare + 生成 PDF 周报（附件） |
| 每月最后交易日 | 跑 current_holdings + 调仓计划（默认 dry-run） |
| 每季最后交易日 | 跑 walkforward + 成本敏感度 |

> 电脑不用开机、不用管 VPS，全靠 GitHub 免费额度（500 分钟/月，2 个 job ≈ 150 分钟）。

---

## 🛠️ 手动做的事（仅月度/季度）

| 频率 | 动作 | 命令 |
|---|---|---|
| 月末调仓 | 看 dry-run 计划 → 决定是否真下 | `python alpaca_buy.py --rebalance --dry-run`<br>`python alpaca_buy.py --rebalance --execute` |
| 季度评估 | 看 walkforward + 成本敏感度 | 邮件里自动收到 |
| 异常时 | 回撤超 -40% / alpha 连续 1 月 -5% | 暂停调仓，参考 `manual_checklist.md` |
| 半年 | 重做回测体检 | 跑 `python backtest_momentum_v2.py` 对比 |

详见 `backtest_output/manual_checklist.md` 和 `backtest_output/paper_3month_plan.md`。

---

## 🐞 排错速查

| 现象 | 原因 | 解决 |
|---|---|---|
| 微信没收到推送 | Server酱额度用完 / SendKey 错 | https://sct.ftqq.com/ 看额度 |
| 邮件没收到 | QQ 授权码过期 / 邮箱拦截 | 看 `logs/auto_*.log` |
| prices.csv 没更新 | NASDAQ 接口限流 | 看 Actions log，等下次 |
| Alpaca 报错 | Key 失效 / Paper 账户 reset | 控制台 Regenerate |
| GitHub Actions 跑挂 | 看 Actions 红 ❌ 日志 | 大多是 NASDAQ 限流，重跑就行 |

---

## 🔒 安全红线

- ❌ **不要**把 `mail.env` / `alpaca.env` 任何内容贴到聊天 / 推 GitHub
- ❌ **不要**用贴过聊天的 Key/授权码 → 立即去 Alpaca / QQ 邮箱换新
- ✅ Key 失效或疑似泄露 → 立即在 Alpaca 控制台 Revoke + 重建
- ✅ 改完策略 / 加新实验 → `git add -A && git commit` 留个回滚点

---

## 📊 3 个月验证目标

| 指标 | 目标 | 红灯线 |
|---|---|---|
| 实际夏普 | 0.8 - 1.5 | < 0.5 |
| 实际最大回撤 | -40% 以内 | > -50% |
| 相对 SPY alpha | 0% 以上 | < -5% |
| 微信告警频率 | < 2 次/月 | > 5 次/月（说明经常跑挂） |

3 个月期满 → `backtest_output/3m_review.md` 写评估报告 → 决定保留 / 切换 / 切真钱。
---

## 🧭 止盈止损执行（碎股账户）

Alpaca Paper 的碎股订单不支持 GTC/Stop/OCO，所以对这个 2 万美金碎股账户采用**软件级止盈止损**：每天收盘后由 GitHub Actions 检查一次，触发就在 paper 里市价卖出。

常用命令（本地手动跑）：

| 操作 | 命令 |
|---|---|
| 看会不会触发（dry-run） | `python manage_orders.py --tpsl` |
| 立即执行一次 | `python manage_orders.py --tpsl --execute` |
| 看持仓和挂单 | `python manage_orders.py --status` |
| 组合级清仓线检查 | `python manage_orders.py --portfolio [--execute]` |

参数可通过 `--tp 0 --sl 0.30 --warn 0.20 --liq 0.25` 改（**默认：关闭止盈、只留 -30% 止损**；组合 -20% 预警、-25% 清仓）。

> 依据回测（见 `backtest_output/tpsl_takeaway.md`）：+20% 止盈会砍掉动量主升浪，默认已关闭；止损只做极端保护；回撤管控主要靠组合级 -20% 预警 / -25% 清仓。

> 云端每日 `auto_run.py` 已自动接入 `--tpsl --execute --tp 0 --sl 0.30`；每个交易日收盘后跑，触发记录写到 `backtest_output/tpsl_log.csv` 并回推 GitHub。

## 🌐 板块分散版 top10（下轮调仓建议）
数据里没有官方行业标签，用近 2 年日收益率相关性聚类当成“板块族”（半导体/AI 会聚成一簇），选股时每簇最多 3 只，防止 10 只全梭哈同一板块。

```bash
python diversified_holdings.py                      # 生成 current_holdings_6m_skip1_top10_div.csv
python plan_rebalance.py --csv backtest_output/current_holdings_6m_skip1_top10_div.csv --budget 20000 --out backtest_output/rebalance_plan_YYYYMMDD_div.csv
```
- 只出方案，不下单；月末邮件里会自动附上「下轮调仓建议：板块分散版 top10」。
- 当前（2026-08）分散版：3 半导体族 + 3 能源 + 2 消费 + 2 医疗。

## 🗓 周频动量策略（影子，和月频同时跑）

- 信号：6 个月动量、跳过最近 1 个月（日线近似），每周最后一个交易日打分，次周建 top10 等权。
- 生成当前周频持仓：`python weekly_strategy.py` → `backtest_output/current_holdings_6m_skip1_top10_weekly.csv` + 周频回测指标 `weekly_backtest_metrics.csv`。
- 每周日 `auto_run.py` 会同时跑：月/周频影子对比（`shadow_compare.py`）+ 刷新周频持仓清单。
- 每周日影子对比**固定跟踪**：月频 base、**月频+vol25**、周频 base、**周频+vol25**，以及 top15/top20/3m/9m/vol25 等候选。
- 最新（2026-08-21 信号）全期夏普约 1.35、样本外约 1.23；**只做影子对照，不接管主配置**。
- 优化扫描（见 `backtest_output/weekly_optimize_takeaway.md`）：**周频+vol25 波动率目标**夏普更高、回撤更小（全期 ~1.38~1.44，回撤 -30%/-25%），推荐作为周频的升级配置；持仓清单仍按 top10 等权展示，执行时按目标波动缩放仓位。
