# -*- coding: utf-8 -*-
"""
统一路径解析：兼容 Windows 本地 + Linux 云端（GitHub Actions）。
用法（在每个脚本顶部）:
    from _paths import WS, OUT
或自己用 project_root() 拿根。
"""
import os
from pathlib import Path

def project_root() -> Path:
    """项目根目录优先级：
    1) 环境变量 PROJECT_ROOT（如果设了）
    2) 本文件所在目录（兼容本地+云端）
    """
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent

WS  = project_root()
OUT = WS / "backtest_output"
LOG = WS / "logs"
for p in (OUT, LOG):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass