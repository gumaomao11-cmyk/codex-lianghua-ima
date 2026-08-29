import os
import json, sys, time, io
sys.path.insert(0, r"F:\even-codex\lianghua+IMA")
import extract_granular_v2 as E

LOG = io.open(r"F:\even-codex\lianghua+IMA\logs\bench_v3.log","w",encoding="utf-8",buffering=1)
def log(m): LOG.write(m+"\n"); LOG.flush()

E.PROVIDERS = [{"base":"https://api.siliconflow.cn/v1",
                "key": os.environ.get("SILICONFLOW_API_KEY",""),
                "models":["deepseek-ai/DeepSeek-V3","deepseek-ai/DeepSeek-V4-Flash","deepseek-ai/DeepSeek-V3.2"]}]

data = json.loads((E.OUT/"zsxq_group_48418411254128_web.json").read_text(encoding="utf-8"))
cands = [it for it in data if E.prefilter(it.get("text",""))]
step = len(cands)//10
sample = [cands[i*step] for i in range(10)]
log(f"candidates={len(cands)}  sample=10")

VT={"research_report","news_summary","single_event","personal_opinion","noise"}
VM={"tier_1_hard_data","tier_2_soft_logic","tier_3_macro_industry"}

for M in ["deepseek-ai/DeepSeek-V3","deepseek-ai/DeepSeek-V4-Flash","deepseek-ai/DeepSeek-V3.2"]:
    E.FORCE_MODEL=M; E.EXHAUSTED.clear()
    ok=fail=nev=bad=empty=0; sents=[]; t0=time.time()
    log(f"\n=== {M} ===")
    for i,it in enumerate(sample,1):
        ts=time.time()
        try: ev,tag = E.call_llm(it.get("text",""), timeout=90)
        except Exception as ex: fail+=1; log(f" [{i:2}] EXC {type(ex).__name__}"); continue
        dt=time.time()-ts
        if tag is None: fail+=1; log(f" [{i:2}] FAIL {dt:.1f}s"); continue
        ok+=1; nev+=len(ev)
        if not ev: empty+=1
        for e in ev:
            if e.get("text_type") not in VT or e.get("materiality_tier") not in VM: bad+=1
            s=e.get("sentiment_score")
            if isinstance(s,(int,float)): sents.append(s)
        log(f" [{i:2}] {it['create_time'][:10]} {len(ev):2}ev {dt:5.1f}s {[e.get('ticker') for e in ev][:4]}")
    el=time.time()-t0
    log(f" -> ok{ok}/10 fail{fail} empty{empty} events{nev}({nev/max(ok,1):.2f}/item) "
        f"enum_err{bad} sent{(sum(sents)/len(sents) if sents else 0):+.3f} "
        f"{el/10:.2f}s/item  full11513@8way={11513*el/10/3600/8:.2f}h")
log("\nDONE")
