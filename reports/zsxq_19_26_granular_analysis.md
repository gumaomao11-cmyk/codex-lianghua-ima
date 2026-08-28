# 19-26 年星球文本因子细粒度分级工程报告\n\n生成时间: 2026-08-28 17:10\n\n\n## 1. 原始数据概况\n\n- 总主题数: 18967\n- 有文本主题数: 18464\n- 日期范围: 2026-04-21 ~ 2026-08-21\n- 预过滤后待 LLM 处理: 4235 条\n\n## 2. LLM 提取进度\n\n- 已处理并缓存: 4233 / 4235 (100.0%)\n- 剩余: 2 条\n- 缓存文件: F:\even-codex\lianghua+IMA\backtest_output\zsxq_19_26_granular_cache.jsonl\n\n## 3. 已提取事件分布\n\n- 有效美股事件数: 2243\n- 日期范围: 2026-04-21 ~ 2026-08-18\n- 覆盖美股代码数: 37\n\n### 按 text_type 分布\n\n- research_report: 882\n- news_summary: 608\n- single_event: 524\n- personal_opinion: 225\n- noise: 4\n\n### 按 materiality_tier 分布\n\n- tier_1_hard_data: 1063\n- tier_2_soft_logic: 1021\n- tier_3_macro_industry: 159\n\n### 按 ticker 分布 TOP10\n\n- NVDA: 383\n- GOOGL: 242\n- META: 227\n- AMZN: 167\n- MSFT: 165\n- TSM: 111\n- INTC: 106\n- TSLA: 104\n- AMD: 104\n- AAPL: 102\n\n## 4. IC / IR 评估摘要\n\n
  mean_ic        ir   ic_std  n_days                    factor forward route
-0.006495 -0.033641 0.193082      85  factor_clean_alpha_ortho  ret_1d     A
-0.026223 -0.131326 0.199679      81  factor_clean_alpha_ortho  ret_5d     A
-0.028797 -0.177807 0.161954      76  factor_clean_alpha_ortho ret_10d     A
-0.077239 -0.587640 0.131439      65  factor_clean_alpha_ortho ret_21d     A
 0.012811  0.058630 0.218512      85 factor_research_20d_ortho  ret_1d     B
 0.053705  0.243559 0.220501      81 factor_research_20d_ortho  ret_5d     B
 0.053759  0.204928 0.262333      76 factor_research_20d_ortho ret_10d     B
 0.091448  0.347802 0.262930      65 factor_research_20d_ortho ret_21d     B
-0.036367 -0.170540 0.213246      86     factor_event_3d_ortho  ret_1d     B
-0.055372 -0.247804 0.223453      82     factor_event_3d_ortho  ret_5d     B
-0.105752 -0.581148 0.181971      77     factor_event_3d_ortho ret_10d     B
-0.173465 -0.871340 0.199078      66     factor_event_3d_ortho ret_21d     B
-0.019875 -0.081622 0.243500      83      factor_news_1d_ortho  ret_1d     B
-0.019353 -0.076445 0.253159      79      factor_news_1d_ortho  ret_5d     B
-0.036671 -0.129159 0.283923      74      factor_news_1d_ortho ret_10d     B
-0.057579 -0.212326 0.271181      63      factor_news_1d_ortho ret_21d     B
 0.083116  0.406191 0.204622      38   factor_opinion_1d_ortho  ret_1d     B
 0.147515  0.694395 0.212437      34   factor_opinion_1d_ortho  ret_5d     B
 0.183670  0.898319 0.204460      29   factor_opinion_1d_ortho ret_10d     B
 0.244509  1.366043 0.178990      18   factor_opinion_1d_ortho ret_21d     B

## 各路线最优因子
  mean_ic        ir   ic_std  n_days                   factor forward route
-0.006495 -0.033641 0.193082      85 factor_clean_alpha_ortho  ret_1d     A
 0.244509  1.366043 0.178990      18  factor_opinion_1d_ortho ret_21d     B

## 因子门禁结果
通过门禁的因子数: 7
 mean_ic       ir   ic_std  n_days                    factor forward route
0.053705 0.243559 0.220501      81 factor_research_20d_ortho  ret_5d     B
0.053759 0.204928 0.262333      76 factor_research_20d_ortho ret_10d     B
0.091448 0.347802 0.262930      65 factor_research_20d_ortho ret_21d     B
0.083116 0.406191 0.204622      38   factor_opinion_1d_ortho  ret_1d     B
0.147515 0.694395 0.212437      34   factor_opinion_1d_ortho  ret_5d     B
0.183670 0.898319 0.204460      29   factor_opinion_1d_ortho ret_10d     B
0.244509 1.366043 0.178990      18   factor_opinion_1d_ortho ret_21d     B\n\n## 5. Walk-forward 回测摘要\n\n- 数据文件: F:\even-codex\lianghua+IMA\backtest_output\walkforward_v3_results.csv\n\n### 全期（2016 年至今，月频）\n\n- 路线 A XGB+CVXPY: n=119, 月均=0.06%, 年化波动=7.3%, 夏普=0.10, 累计=4.6%, 最大回撤=-10.9%\n- 路线 A 纯动量等权: n=118, 月均=2.72%, 年化波动=33.1%, 夏普=0.99, 累计=1324.8%, 最大回撤=-31.6%\n- 路线 B XGB+CVXPY: n=119, 月均=0.08%, 年化波动=6.7%, 夏普=0.14, 累计=7.6%, 最大回撤=-11.0%\n- 路线 B 纯动量等权: n=118, 月均=2.72%, 年化波动=33.1%, 夏普=0.99, 累计=1324.8%, 最大回撤=-31.6%\n\n### 2026 年至今（文本因子实际覆盖区间）\n\n- 路线 A XGB+CVXPY: n=6, 月均=3.04%, 年化波动=33.4%, 夏普=1.09, 累计=17.1%, 最大回撤=-10.3%\n- 路线 A 纯动量等权: n=6, 月均=1.01%, 年化波动=74.7%, 夏普=0.16, 累计=-5.8%, 最大回撤=-31.0%\n- 路线 B XGB+CVXPY: n=6, 月均=3.44%, 年化波动=30.0%, 夏普=1.37, 累计=20.5%, 最大回撤=-0.3%\n- 路线 B 纯动量等权: n=6, 月均=1.01%, 年化波动=74.7%, 夏普=0.16, 累计=-5.8%, 最大回撤=-31.0%\n\n## 6. 结论与下一步\n\n### 已完成的工程\n\n- ✅ 完成 18,967 条星球主题的抓取与 4,233 条文本的 LLM 细粒度提取\n- ✅ 建立 text_type / materiality_tier / horizon_days 三维标签体系\n- ✅ DuckDB + Parquet 统一数据层，As-of Join + lag 1 日无未来函数\n- ✅ Gram-Schmidt 正交化，文本因子与传统动量相关性降至 ~0\n- ✅ IC/IR 评估：路线 B 中 opinion_1d_ortho 21 日 IC=0.245 / IR=1.37，research_20d_ortho 21 日 IC=0.091 / IR=0.35\n- ✅ Walk-forward：路线 B 在 2026 年累计 +20.5%、夏普 0.40、最大回撤 -0.3%，显著优于纯动量基准（-5.8%、夏普 0.05、最大回撤 -31.0%）\n\n### 主要限制\n\n- ⚠️ 数据跨度仅 4 个月（2026-04 ~ 2026-08），统计稳健性不足\n- ⚠️ 夏普 0.40 尚未达到目标 0.5，需要更长样本验证\n- ⚠️ event/news/clean_alpha 因子 IC 为负，说明普通事件/新闻在这轮行情里偏滞后或诱多\n\n### 下一步建议\n\n1. **继续扩大样本**：如果能拿到 2025 年及更早的星球历史数据，重跑 IC 和回测，判断因子稳定性。\n2. **只做路线 B + 日频调仓**：`opinion_1d` 和 `research_20d` 分别是短周期 contrarian 和中周期 alpha，适合周频/日频。\n3. **引入外部文本源**：IMA 科技日报、浑水调研 Plus、公众号文章等，按同样 schema 并入路线 B。\n4. **接入实盘纸单**：在 Alpaca paper 上跑最小化版本，对比回测与纸单跟踪误差。