# -*- coding: utf-8 -*-
"""QQ 邮箱 SMTP 发邮件。从本地 mail.env 读授权码，不在聊天里贴。
mail.env 格式（一行一个 KEY=VAL，# 开头为注释）：
  QQ_MAIL_AUTH_CODE=你的授权码
  QQ_MAIL_FROM=869357594@qq.com  (可选，默认就是上面)
  QQ_MAIL_TO=869357594@qq.com    (可选，默认与 FROM 相同)
"""
import os, smtplib, ssl, html
from pathlib import Path
from email.message import EmailMessage

ENV_FILE = Path(__file__).parent / "mail.env"


def _from_env_file(p):
    out = {}
    if not Path(p).exists():
        return out
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = [x.strip() for x in s.split("=", 1)]
        out[k] = v
    return out


_cfg = _from_env_file(ENV_FILE)
os.environ.update(_cfg)

AUTH = (os.environ.get("QQ_MAIL_AUTH_CODE") or "").strip()
FROM = (os.environ.get("QQ_MAIL_FROM") or "869357594@qq.com").strip()
TO = (os.environ.get("QQ_MAIL_TO") or FROM).strip()
SMTP = (os.environ.get("QQ_MAIL_SMTP") or "smtp.qq.com").strip()
PORT = int(os.environ.get("QQ_MAIL_PORT") or "465")


def _sanitize(text):
    """去掉会破坏邮件的非法控制字符，但保留中文、标点、换行。"""
    if not isinstance(text, str):
        text = str(text)
    return "".join(ch for ch in text if ch in "\t\n\r" or (ord(ch) >= 0x20 and ord(ch) != 0x7F))


def send(subject, body, attachments=None):
    if not AUTH:
        print("[mailer] QQ_MAIL_AUTH_CODE 未设置")
        return False

    body = _sanitize(body)
    subject = _sanitize(subject)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = FROM
    msg["To"] = TO

    # 纯文本：使用 quoted-printable，QQ 邮箱/Outlook 兼容度比 base64 高
    msg.set_content(body, subtype="plain", charset="utf-8", cte="quoted-printable")

    # HTML：必须转义后再放入 <pre>，否则正文里的 < > & 会破坏标签
    safe_body = html.escape(body)
    html_body = (
        "<html><body>"
        "<pre style=\"font-family:monospace,\"Microsoft YaHei\",sans-serif;"
        "font-size:14px;line-height:1.5;white-space:pre-wrap;word-break:break-word;\">"
        f"{safe_body}"
        "</pre></body></html>"
    )
    msg.add_alternative(html_body, subtype="html", charset="utf-8", cte="quoted-printable")

    for p in (attachments or []):
        p = Path(p)
        if not p.exists():
            continue
        data = p.read_bytes()
        suffix = p.suffix.lower()
        main, sub = "application", "octet-stream"
        if suffix == ".csv":
            main, sub = "text", "csv"
        elif suffix == ".md":
            main, sub = "text", "markdown"
        elif suffix == ".png":
            main, sub = "image", "png"
        elif suffix == ".txt":
            main, sub = "text", "plain"
        msg.add_attachment(data, maintype=main, subtype=sub, filename=p.name)

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP, PORT, context=ctx, timeout=30) as s:
            s.login(FROM, AUTH)
            s.send_message(msg)
        print(f"[mailer] 发送成功: {subject}")
        return True
    except Exception as e:
        print(f"[mailer] 发送失败: {e}")
        return False
