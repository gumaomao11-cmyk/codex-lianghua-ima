import json, sys, time, io
sys.path.insert(0, r"F:\even-codex\lianghua+IMA")
import extract_granular_v2 as E

LOG = io.open(r"F:\even-codex\lianghua+IMA\logs\kimi_bench.log", "w", encoding="utf-8", buffering=1)
def log(m):
    LOG.write(m + "\n"); LOG.flush()

M = "Pro/moonshotai/Kimi-K2.6"
E.FORCE_MODEL = M
E.EXHAUSTED.clear()

data = json.loads((E.OUT/"zsxq_group_48418411254128_web.json").read_text(encoding="utf-8"))
cands = [it for it in data if E.prefilter(it.get("text",""))]
step = len(cands)//15
sample = [cands[i*step] for i in range(15)]
log(f"model={M}  sample=15  (of {len(cands)} candidates)")

VT = {"research_report","news_summary","single_event","personal_opinion","noise"}
VM = {"tier_1_hard_data","tier_2_soft_logic","tier_3_macro_industry"}
ok=fail=nev=bad=empty=0; sents=[]; t0=time.time()

for i,it in enumerate(sample,1):
    ts=time.time()
    try:
        ev, tag = E.call_llm(it.get("text",""), timeout=90)
    except Exception as ex:
        fail+=1; log(f"[{i:2}] EXC {type(ex).__name__}"); continue
    dt=time.time()-ts
    if tag is None:
        fail+=1; log(f"[{i:2}] FAIL  {dt:.1f}s"); continue
    ok+=1; nev+=len(ev)
    if not ev: empty+=1
    for e in ev:
        if e.get("text_type") not in VT or e.get("materiality_tier") not in VM: bad+=1
        s=e.get("sentiment_score")
        if isinstance(s,(int,float)): sents.append(s)
    log(f"[{i:2}] {it['create_time'][:10]} {len(ev):2}ev {dt:5.1f}s {[e.get('ticker') for e in ev][:4]}")

el=time.time()-t0
log("")
log(f"RESULT {M}")
log(f"  success {ok}/15  fail {fail}  empty {empty}")
log(f"  events {nev} ({nev/max(ok,1):.2f}/item)  enum_err {bad}")
log(f"  mean_sentiment {(sum(sents)/len(sents) if sents else 0):+.3f}")
log(f"  elapsed {el:.1f}s ({el/15:.2f}s/item)")
log(f"  full 11513 items: {11513*el/15/3600:.1f}h serial / {11513*el/15/3600/8:.1f}h at 8-way")
log("DONE")
