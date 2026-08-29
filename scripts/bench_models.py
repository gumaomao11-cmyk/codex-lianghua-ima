import json, sys, io, time
sys.path.insert(0, r"F:\even-codex\lianghua+IMA")
import extract_granular_v2 as E

data = json.loads((E.OUT/"zsxq_group_48418411254128_web.json").read_text(encoding="utf-8"))
cands = [it for it in data if E.prefilter(it.get("text",""))]
# 跨时间均匀取 30 条，覆盖不同月份
step = len(cands)//30
sample = [cands[i*step] for i in range(30)]
print(f"样本 30 条，覆盖 {sample[-1]['create_time'][:7]} ~ {sample[0]['create_time'][:7]}\n")

VT = {"research_report","news_summary","single_event","personal_opinion","noise"}
VM = {"tier_1_hard_data","tier_2_soft_logic","tier_3_macro_industry"}

for model in ["deepseek-v4-flash","glm-5.2","sensenova-6.8-flash-lite","minimax-m2.7"]:
    E.FORCE_MODEL = model
    E.EXHAUSTED.clear()
    ok=fail=nev=bad=0; sents=[]; t0=time.time()
    for it in sample:
        ev, tag = E.call_llm(it.get("text",""))
        if tag is None: fail+=1; continue
        ok+=1; nev+=len(ev)
        for e in ev:
            if e.get("text_type") not in VT or e.get("materiality_tier") not in VM: bad+=1
            s=e.get("sentiment_score")
            if isinstance(s,(int,float)): sents.append(s)
    el=time.time()-t0
    ms = (sum(sents)/len(sents)) if sents else 0
    print(f"{model:26} 成功{ok:2}/30 失败{fail:2} 事件{nev:3} 枚举错{bad:2} "
          f"均情绪{ms:+.3f} 耗时{el:5.1f}s ({el/30:.1f}s/条)")
