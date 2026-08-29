import json, io, sys
from collections import Counter
sys.path.insert(0, r"F:\even-codex\lianghua+IMA")
from ticker_norm import universe, VALID_TT, VALID_MT
LOG=io.open(r"F:\even-codex\lianghua+IMA\logs\q400.log","w",encoding="utf-8",buffering=1)
def log(m): LOG.write(str(m)+"\n"); LOG.flush()

L=[l for l in io.open(r"F:\even-codex\lianghua+IMA\backtest_output\zsxq_v3_clean_sample.jsonl",
                      encoding="utf-8").read().splitlines() if l.strip()]
bad=0; evs=[]; models=Counter(); empty=0; months=Counter()
for l in L:
    try: o=json.loads(l)
    except: bad+=1; continue
    models[o.get("model")]+=1
    e=o.get("events") or []
    if not e: empty+=1
    evs.extend(e)
    months[(o.get("create_time") or "")[:7]]+=1

U=universe()
log("=== 400条样本质量报告 (DeepSeek-V3 + 中性prompt + 规范化) ===\n")
log(f"记录数        {len(L)}")
log(f"JSON损坏      {bad}          <- 修复前是 10/11392")
log(f"空返回        {empty} ({empty/len(L)*100:.1f}%)   <- 修复前 glm-5-turbo 88.7%")
log(f"事件总数      {len(evs)} ({len(evs)/len(L):.2f}/条)  <- 修复前 0.40/条")
log(f"模型一致性    {dict(models)}")
log("")
log(f"ticker 全在池内   {all(e['ticker'] in U for e in evs)}")
log(f"枚举全部合法      {all(e['text_type'] in VALID_TT and e['materiality_tier'] in VALID_MT for e in evs)}")
log(f"情绪分全在[-1,1]  {all(-1<=e['sentiment_score']<=1 for e in evs)}")
log(f"horizon 合法      {all(e['expected_horizon_days'] in (0,1,3,20) for e in evs)}")
log("")
log("text_type 分布: " + str(dict(Counter(e['text_type'] for e in evs))))
log("tier 分布     : " + str(dict(Counter(e['materiality_tier'] for e in evs))))
log(f"独立标的数    : {len(set(e['ticker'] for e in evs))}")
log("Top15 标的    : " + ", ".join(f"{k}({v})" for k,v in Counter(e['ticker'] for e in evs).most_common(15)))
s=[e['sentiment_score'] for e in evs]
log(f"情绪  mean={sum(s)/len(s):+.3f}  min={min(s):+.2f} max={max(s):+.2f}")
c=[e['confidence'] for e in evs]
log(f"置信  mean={sum(c)/len(c):.3f}")
log("")
log("时间覆盖: " + ", ".join(f"{k}:{v}" for k,v in sorted(months.items())[:8]) + " ...")
