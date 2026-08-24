# IMA 数据更新 SOP（本机操作，可抄）

> 适用：IMA 影子仓（$20k paper，动量+IMA 词频 win60/λ1.2）
> 要点：**原文研报只在本机 us-stock-data，不进公开 GitHub**；进仓库的只有"结果表+名单+行情"。

## 什么时候需要更新
- ima 客户端「美国科技日报 / AI总结 / 上市公司模型」新增了很多内容（建议最多每周一次，不必每天）。
- 或你想让名单每月跟着最新研报内容重选一次。

## 一次性准备（只有首次）
1. 进入项目目录：
   ```powershell
   cd F:\even-codex\lianghua+IMA
   & F:\even-codex\panda\.venv\Scripts\Activate.ps1
   ```
2. 确保本机 `F:\even-codex\us-stock-data\ima_all_meta.json` 是最新的合并结果。

## 日常更新步骤（核心就是一条命令）
```powershell
python update_ima_data.py --push
```
它自动做完：
1. 用最新 `ima_all_meta.json` 重建词频因子 → `backtest_output\kb_abstract_factors.csv`
2. 复制结果表到仓库 `data\ima\`
3. 重选目标名单 → `data\ima\ima_final_top10.csv`（动量+IMA，win60/λ1.2）
4. commit + push 到 GitHub

> 不想 push（只想本地看）就运行：`python update_ima_data.py`

## 更细的全量流程（想一步一步手工做时）
```powershell
# 0) 先刷新 ima 客户端知识库，让别人把新内容同步进来（登录态在你这台电脑）

# 1) 重新合并 ima 摘要清单 → 本机 us-stock-data\ima_all_meta.json
python kb_build_all_meta.py

# 2) 重建词频因子（读 ima_all_meta.json，只在本地算）
python kb_abstract_factor.py

# 3) 复制结果表到仓库
python sync_local_ima.py

# 4) 重选名单（读 data\ima\kb_abstract_factors.csv）
python select_ima_final.py

# 5) 同步 + push
python sync_local_ima.py --push
```

## 如果名单变了，要动 Alpaca 仓位吗？
- **先不要自己调。** 因为 GLW/NBIS/STX/WDC 和你主策略（lianghua2）在**同一个 Alpaca paper 账户重叠**。
- 正确做法：`python update_ima_data.py --push` 后，把新名单发我看一眼，**我先确认要不要调仓、怎么调**（避免把主策略的份额误卖）。
- 只有当你以后开了**独立的 IMA paper 账户**（另一套 Alpaca key），才能让 Alpaca 调仓也全自动。

## 风险与红线
- 原文研报/摘要（`ima_all_meta.json`、`kb_*` 原始文件、`kb_export/*`）**绝不 push**。
- `alpaca.env` / `mail.env` / `.env.llm` 是本地凭据，严禁进 GitHub。
- 名单变更是低频操作；日常净值/日报自动更新已由 GitHub Actions 负责，不用你手动跑。
