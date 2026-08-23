# IMA 因子实验策略（独立副本）

> 本仓库是从原 `lianghua2` 项目独立出来的实验副本，仅在**本地**使用 IMA 知识库因子，原策略仓库保持不动。
> 数据源仍统一读取 `F:\even-codex\us-stock-data`（环境变量 `STOCK_DATA_DIR`）。

## 和原策略的区别
- 原：6 个月动量 + 加速 top10 月频。
- 本实验：在动量基础上叠加 IMA 订阅知识库的「提及热度 / 情感」因子，暂不接管主策略。

## IMA 数据怎么来（官方权限内，安全路径）
1. 订阅库 PDF 不能直接下载，IMA 搜索接口也只返回标题，不返回正文。
2. 从 ima 客户端 Copilot 把报告正文转成 Markdown，保存到：
   `F:\even-codex\us-stock-data\kb_export\`
3. 跑：
   `python kb_import_local.py`
   输出 `backtest_output/kb_local_signals.csv`（本地保留，不入 git）。

## 本地 IMA 因子脚本
| 脚本 | 作用 |
|---|---|
| kb_scan.py | 扫描订阅库标题/目录 |
| kb_factor.py / kb_search_factor.py / kb_mention_factor.py / kb_miner.py | 标题级提及/情感因子的多版尝试 |
| kb_import_local.py | 读取 ima 客户端导出的 md/txt，生成正文级信号 |

## 安全红线
- 订阅库内容（`backtest_output/kb_*.csv/json`、`kb_export/*`）**不进公开 GitHub**。
- `mail.env` / `alpaca.env` 不进 GitHub。

## ✅ 已解决：从 ima 客户端本地缓存还原 PDF 正文
- 官方 API `get_media_info` 对订阅库返回 `220030`（无权限），搜索接口也不返回正文片段。
- 但 ima 客户端在本地浏览器缓存里会留下已打开 PDF 的数据。`kb_cache_extract.py` 会自动扫描：
  `C:\Users\ASUS\AppData\Local\ima.copilot\User Data\Default\Cache\Cache_Data`
- 它能还原可读的 `.pdf` 和 `.txt` 全文到：
  - `F:\even-codex\us-stock-data\kb_cache_pdfs`
  - `F:\even-codex\us-stock-data\kb_cache_text`
- 再输出过滤后的本地因子：`backtest_output\kb_cache_factors.csv`
- 运行：`python kb_cache_extract.py`
- 已安装依赖：`pypdf`
- ⚠️ 这些还原的 PDF/文本是订阅库内容，**只保存在本地数据目录，绝不推入 GitHub**。
