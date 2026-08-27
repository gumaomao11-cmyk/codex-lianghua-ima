# 工业级文本-动能量化系统 v2 实施报告

## 已完成模块

### P0：统一数据层
- `data/duckdb/quant_store.duckdb` + Parquet 文件
- `data/dataloader.py`：`UnifiedLoader` 支持 `load_prices`、`load_text_factors`（As-of Join + lag 1 日）、`load_returns`、`load_etf_ref`、`load_industry_map`
- `scripts/migrate_to_duckdb.py`：一次性迁移脚本

### P0：因子正交化与 IC/IR
- `alpha/orthogonal.py`：Gram-Schmidt 正交化 + PCA 备选
- `alpha/ic_analysis.py`：`FactorIC` 计算月频 IC、IR、IC 衰减
- `backtest_output/factor_ic.csv`：三源文本因子评估结果

### P1：风险模型与组合优化
- `risk/industry_map.py`：自定义 10 大行业映射
- `risk/barra_risk.py`：行业暴露计算 + 约束检查
- `risk/optimizer.py`：CVXPY 均值-方差优化器，个股 ≤10%、行业 ≤25%

### P1：回测引擎
- `backtest/walkforward_v2.py`：严格 walk-forward + 正交化 + CVXPY 风控

### 测试
- `tests/test_dataloader.py`：As-of Join 无未来函数验证 ✓
- `tests/test_orthogonal.py`：正交化后相关性 < 0.1 ✓
- `tests/test_optimizer.py`：个股/行业约束满足 ✓

## 因子 IC 评估结果

| factor | mean_ic | ir | passed |
|---|---|---:|:---:|
| ima_tsm | 0.0376 | 1.74 | ✓ |
| ima_disagreement_ortho | 0.0136 | 0.32 | ✗ |
| ima_risk_flag_ortho | 0.0120 | 0.28 | ✗ |
| ... | ... | ... | ... |

**结论**：
- 只有 `ima_tsm`（非正交）通过了 IC>0.02 且 IR>0.3 的门槛。
- 正交化后的文本因子 IC 普遍下降甚至变负，说明当前文本因子与传统动量高度相关，**未能提供稳定的独立 Alpha**。

## Walk-forward 回测结果（月度口径，2025-09 至 2026-07）

| 策略 | 月均收益 | 夏普 | 累计收益 | 最大回撤 |
|---|---|---:|---:|---:|
| 纯动量_等权top10 | 8.87% | 0.55 | 107.6% | -25.6% |
| 纯动量_CVXPY风控 | 7.69% | 0.50 | 87.0% | -28.4% |
| XGBoost_等权top10 | 3.59% | 0.29 | 31.4% | -27.0% |
| XGBoost_CVXPY风控 | 0.03% | 0.00 | -6.5% | -28.6% |

## 关键发现

1. **数据层已正确实现**：DuckDB + As-of Join + lag 1 日确保无 look-ahead bias。
2. **文本因子质量不足**：当前基于词频/摘要的因子无法提供超越动量的稳定 Alpha。`ima_tsm` 原始值有 IC，但正交化后失效。
3. **风控优化未达预期**：CVXPY 优化器在本期数据中反而略微放大回撤、降低收益。原因包括：
   - 协方差矩阵仅 63 日历史，估计噪声大；
   - 均值-方差优化对 Alpha 预测误差高度敏感；
   - 文本 Alpha 信号弱，优化器难以产生正贡献。

## 与目标的差距

| 目标 | 实际 | 状态 |
|---|---|:---|
| 文本因子月频 IC > 0.03 | ima_tsm 原始 IC=0.038，但正交化后失效 | ⚠️ 部分达成 |
| 优化器版本最大回撤 < -18% | 实际 -28.6% | ❌ 未达成 |
| 优化器版本夏普 > 0.5 | 实际 0.00 | ❌ 未达成 |

## 下一步建议

1. **获取更高质量文本**：目前 IMA/星球摘要粒度太粗。建议用 LLM 对全文做结构化情绪/事件打分。
2. **扩大数据长度**：当前 walk-forward 仅 10 个月，模型训练样本不足。
3. **优化风控参数**：尝试更保守的风险厌恶系数、更长的协方差估计窗口，或改用风险预算/等风险贡献模型。
4. **事件驱动触发**：把文本信号作为日频事件触发（如业绩超预期），而非月度特征叠加。
