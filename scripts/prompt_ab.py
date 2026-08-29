import os
import json, sys, io, requests
sys.path.insert(0, r"F:\even-codex\lianghua+IMA")
import extract_granular_v2 as E
LOG=io.open(r"F:\even-codex\lianghua+IMA\logs\prompt_ab.log","w",encoding="utf-8",buffering=1)
def log(m): LOG.write(m+"\n"); LOG.flush()

KEY=os.environ.get("SILICONFLOW_API_KEY","")
URL="https://api.siliconflow.cn/v1/chat/completions"

# 中性版 prompt：去掉激进的"忽略/输出[]"指令，保留 schema 与分类定义
NEUTRAL = f"""你是美股量化研究员。从下面的投资星球内容中，抽取所有**被提及的美股上市公司**的结构化信号。

规则：
- 只输出合法 JSON 数组，无任何解释文字。
- 只要文中提及某美股公司（中文名或代码均算），就为它输出一条记录。
- A股/港股公司不要输出。若确实没有任何美股公司，才输出 []
- text_type: research_report(研报/目标价/EPS) | news_summary(新闻汇总/日报) | single_event(单一催化剂) | personal_opinion(个人观点) | noise(闲聊)
- materiality_tier: tier_1_hard_data(含明确数字) | tier_2_soft_logic(逻辑推演) | tier_3_macro_industry(行业宏观)
- sentiment_score: -1(极空) ~ +1(极多)
- expected_horizon_days: research_report=20, single_event=3, news_summary=1, personal_opinion=1, noise=0
- confidence: 0~1
- evidence: 原文中的关键依据片段

JSON Schema: {E.SCHEMA}
内容：{{text}}"""

data=json.loads((E.OUT/"zsxq_group_48418411254128_web.json").read_text(encoding="utf-8"))
cands=[it for it in data if E.prefilter(it.get("text",""))]
step=len(cands)//10
sample=[cands[i*step] for i in range(10)]

def run(model, tmpl, tag):
    tot=0; empty=0
    log(f"\n=== {tag} | {model} ===")
    for i,it in enumerate(sample,1):
        try:
            r=requests.post(URL, headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"},
                json={"model":model,"temperature":0.0,
                      "messages":[{"role":"system","content":"You return only valid JSON arrays."},
                                  {"role":"user","content":tmpl.replace("{text}", it.get("text","")[:6000])}]},
                timeout=90)
            c=r.json()["choices"][0]["message"].get("content","") or ""
            import re
            c=re.sub(r"^```(?:json)?\s*","",c.strip()); c=re.sub(r"\s*```$","",c)
            try: arr=json.loads(c)
            except:
                m=re.search(r"\[.*\]",c,re.S); arr=json.loads(m.group(0)) if m else []
            n=len(arr) if isinstance(arr,list) else 0
            tot+=n
            if n==0: empty+=1
            log(f" [{i:2}] {it['create_time'][:10]} {n:2}ev {[e.get('ticker') for e in arr][:5] if n else ''}")
        except Exception as ex:
            log(f" [{i:2}] EXC {type(ex).__name__}")
    log(f" -> events={tot} ({tot/10:.2f}/item) empty={empty}/10")

run("deepseek-ai/DeepSeek-V3", E.PROMPT,  "A-当前激进prompt")
run("deepseek-ai/DeepSeek-V3", NEUTRAL,   "B-中性prompt")
log("\nDONE")
