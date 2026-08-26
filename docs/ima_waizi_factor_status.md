# 外资研报信息因子 - 接入进展

（2026-08-25 记录）

## 突破
- 接入 ima 官方开放接口（`ima_openapi.py`，凭 .env.ima 内 Client ID / API Key），无需再抓浏览器会话头。
- 已定位「浑水调研」共享知识库【原文可查看4】（base_type=共享知识库，可查原文）。
- 外资研报文件夹已可遍历：导出 4850 条（data/ima_waizi_folder.csv），标题含大摩/高盛/瑞银/野村研报。
- 标题含美股代码：NVDA/AMD/MRVL/AMAT/WDC/INTC/STX/GLW/LITE/ARM/DELL/AVGO/QCOM 等均有覆盖。

## 限制
- 原文链路被挡：get_media_info 返回 220030（须在 ima 客户端内查看）。
- 标题仅 231 条带可靠日期，且大多为 2025 年；时间序列对齐不足。
- 今日 API 配额已用完（220021，明天重置）。

## 下一步（配额恢复后）
1. 运行 `ima_kb_sweep.py` 补齐：电话会总结/上市公司模型/库根目录 AI总结 markdown + 对候选美股做高亮检索。
2. 用高亮片段 + ima 总结正文构造「外资研报词频/情绪」因子。
3. 叠加现有「动量+IMA词频(win60/λ1.2)」配方跑 IS/OOS 对比（复用 llm_ima_oos_compare.py 管线）。

## 安全
- ima 凭证存 `.env.ima`（gitignore），不提交公开仓库。
- 研报正文/素材为付费内容，产物只留本地（backtest_output/kb_* 已在 gitignore）。
