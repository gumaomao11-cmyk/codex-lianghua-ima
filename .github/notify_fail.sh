#!/usr/bin/env bash
# 失败告警脚本：跑挂时同时推微信（Server酱）+ QQ 邮箱
# 用法：bash notify_fail.sh "<主题>" "<链接>"
set -u
SUB="${1:-GitHub Actions 跑挂}"
LINK="${2:-}"
TS=$(date "+%Y-%m-%d %H:%M:%S")
SUB_SHORT="${SUB:0:30}"

echo "[$TS] >>> 失败告警: $SUB"
echo "[$TS] >>> 链接: $LINK"

# === 1) 微信（Server酱）===
if [ -n "${WECHAT_SEND_KEY:-}" ]; then
  DESP="${TS}%0A${LINK}%0A%0A请尽快去 Actions 看日志"
  HTTP=$(curl -s -o /tmp/wx_resp.txt -w "%{http_code}" \
    "https://sctapi.ftqq.com/${WECHAT_SEND_KEY}.send" \
    --data-urlencode "title=策略跑挂: ${SUB_SHORT}" \
    --data-urlencode "desp=${DESP}" \
    --max-time 15 || echo "000")
  if [ "$HTTP" = "200" ]; then
    echo "[$TS] 微信推送: OK"
  else
    echo "[$TS] 微信推送失败 http=$HTTP, resp=$(cat /tmp/wx_resp.txt 2>/dev/null | head -c 200)"
  fi
else
  echo "[$TS] [跳过] WECHAT_SEND_KEY 未配置"
fi

# === 2) QQ 邮箱 ===
if [ -n "${QQ_MAIL_AUTH_CODE:-}" ] && [ -n "${QQ_MAIL_TO:-}" ]; then
  EXPORT_TS="$TS" EXPORT_SUB="$SUB" EXPORT_SUB_SHORT="$SUB_SHORT" EXPORT_LINK="$LINK" \
  QQ_FROM="${QQ_MAIL_FROM:-869357594@qq.com}" \
  python3 - <<'PYEOF'
import os, smtplib, ssl
from email.message import EmailMessage
auth  = os.environ["QQ_MAIL_AUTH_CODE"].strip()
frm   = os.environ.get("QQ_FROM", "869357594@qq.com").strip()
to    = os.environ["QQ_MAIL_TO"].strip()
ts    = os.environ["EXPORT_TS"]
sub   = os.environ["EXPORT_SUB"]
short = os.environ["EXPORT_SUB_SHORT"]
link  = os.environ["EXPORT_LINK"]
subj  = f"\u26a0\ufe0f 策略跑挂: {short}"
body  = (
    "策略自动跑失败，请去 Actions 查看日志：\n"
    f"链接: {link}\n\n"
    f"时间: {ts}\n"
    f"主题: {sub}\n\n"
    "—— GitHub Actions 自动告警\n"
)
msg = EmailMessage()
msg["Subject"] = subj
msg["From"] = frm
msg["To"] = to
msg.set_content(body)
try:
    with smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=15, context=ssl.create_default_context()) as s:
        s.login(frm, auth)
        s.send_message(msg)
    print(f"[{ts}] QQ 邮箱推送: OK")
except Exception as e:
    print(f"[{ts}] QQ 邮箱推送失败: {e}")
PYEOF
else
  echo "[$TS] [跳过] QQ_MAIL_AUTH_CODE 未配置"
fi