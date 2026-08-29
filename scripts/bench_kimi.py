import json, sys, time
sys.path.insert(0, r"F:\even-codex\lianghua+IMA")
import extract_granular_v2 as E

M = "Pro/moonshotai/Kimi-K2.6"
E.FORCE_MODEL = M
E.EXHAUSTED.clear()

data = json.loads((E.OUT/"zsxq_group_48418411254128_web.json").read_text(encoding="utf-8"))
cands = [it for it in data if E.prefilter(it.get("text",""))]
step = len(cands)//20
sample = [cands[i*step] for i in range(20)]

VT = {"research_report","news_summary","single_event","personal_opinion","noise"}
VM = {"tier_1_hard_data","tier_2_soft_logic","tier_3_macro_industry"}

ok=fail=nev=bad=0; sents=[]; t0=time.time()
for i,it in enumerate(sample,1):
    ev, tag = E.call_llm(it.get("text",""))
    if tag is None:
        fail+=1; print(f"  [{i:2}] FAIL"); continue
    ok+=1; nev+=len(ev)
    for e in ev:
        if e.get("text_type") not in VT or e.get("materiality_tier") not in VM: bad+=1
        s=e.get("sentiment_score")
        if isinstance(s,(int,float)): sents.append(s)
    tk = [e.get("ticker") for e in ev][:4]
    print(f"  [{i:2}] {it['create_time'][:10]}  {len(ev)} ev  {tk}")

el=time.time()-t0
print(f"\n{M}")
print(f"  成功 {ok}/20   失败 {fail}   空返回 {sum(1 for _ in [] )}")
print(f"  事件总数 {nev}  ({nev/max(ok,1):.2f}/条)")
print(f"  枚举错误 {bad}")
print(f"  均情绪 {(sum(sents)/len(sents) if sents else 0):+.3f}")
print(f"  耗时 {el:.1f}s  ({el/20:.2f}s/条)")
print(f"  全量 11513 条预估: {11513*el/20/3600:.1f} 小时 (单线程) / {11513*el/20/3600/8:.1f} 小时 (8并发)")
