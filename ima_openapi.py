# -*- coding: utf-8 -*-
"""ima OpenAPI helper: 读取 IMA_OPENAPI_CLIENTID / IMA_OPENAPI_APIKEY。
用法:
  $env:IMA_OPENAPI_CLIENTID=...
  $env:IMA_OPENAPI_APIKEY=...
  python ima_openapi.py search_knowledge_base '{"query":"","cursor":"","limit":20}'
"""
import json, os, sys, requests
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "https://ima.qq.com/openapi/wiki/v1"
def creds():
    cid = os.environ.get("IMA_OPENAPI_CLIENTID") or ""
    key = os.environ.get("IMA_OPENAPI_APIKEY") or ""
    envp = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.ima")
    if os.path.exists(envp):
        for line in open(envp, encoding="utf-8"):
            line = line.strip()
            if line.startswith("IMA_OPENAPI_CLIENTID="):
                cid = line.split("=",1)[1].strip()
            elif line.startswith("IMA_OPENAPI_APIKEY="):
                key = line.split("=",1)[1].strip()
    if not cid or not key:
        print("缺少 IMA_OPENAPI_CLIENTID / IMA_OPENAPI_APIKEY")
        sys.exit(1)
    return cid, key

def api(ep, body):
    cid, key = creds()
    h = {"Content-Type": "application/json",
         "ima-openapi-clientid": cid,
         "ima-openapi-apikey": key}
    r = requests.post(f"{BASE_URL}/{ep}", headers=h, json=body, timeout=60)
    try:
        return r.json()
    except Exception:
        return {"http": r.status_code, "raw": r.text[:800]}

if __name__ == "__main__":
    ep = sys.argv[1]
    body = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    print(json.dumps(api(ep, body), ensure_ascii=False, indent=2)[:6000])
