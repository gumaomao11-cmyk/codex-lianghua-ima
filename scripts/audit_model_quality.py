import json, re
from pathlib import Path
from collections import Counter, defaultdict

CACHE = Path(r"F:\even-codex\us-stock-data\知识星球\zsxq_19_26_granular_cache.jsonl")
VALID_TT = {"research_report","news_summary","single_event","personal_opinion","noise"}
VALID_MT = {"tier_1_hard_data","tier_2_soft_logic","tier_3_macro_industry"}

rows = CACHE.read_text(encoding="utf-8").splitlines()
stat = defaultdict(lambda: {"lines":0,"events":0,"bad_tt":0,"bad_mt":0,"bad_sent":0,
                            "sent_sum":0.0,"sent_n":0,"conf_sum":0.0,"empty":0})
parse_err = 0
for ln in rows:
    if not ln.strip():
        continue
    try:
        o = json.loads(ln)
    except Exception:
        parse_err += 1
        continue
    m = o.get("model") or "unknown"
    s = stat[m]
    s["lines"] += 1
    ev = o.get("events") or []
    if not ev:
        s["empty"] += 1
    for e in ev:
        s["events"] += 1
        if e.get("text_type") not in VALID_TT: s["bad_tt"] += 1
        if e.get("materiality_tier") not in VALID_MT: s["bad_mt"] += 1
        sc = e.get("sentiment_score")
        if not isinstance(sc,(int,float)) or not (-1.0 <= sc <= 1.0):
            s["bad_sent"] += 1
        else:
            s["sent_sum"] += sc; s["sent_n"] += 1
        c = e.get("confidence")
        if isinstance(c,(int,float)): s["conf_sum"] += c

print(f"总行数 {len(rows)}  JSON解析失败 {parse_err}")
print()
hdr = f'{"模型":38} {"样本":>6} {"事件":>6} {"空率":>6} {"枚举错":>7} {"分数越界":>8} {"均情绪":>7} {"均置信":>7}'
print(hdr); print("-"*len(hdr))
for m, s in sorted(stat.items(), key=lambda kv: -kv[1]["lines"]):
    ev = s["events"] or 1
    print(f'{m:38} {s["lines"]:6d} {s["events"]:6d} '
          f'{s["empty"]/max(s["lines"],1)*100:5.1f}% '
          f'{(s["bad_tt"]+s["bad_mt"])/ev*100:6.2f}% '
          f'{s["bad_sent"]/ev*100:7.2f}% '
          f'{(s["sent_sum"]/s["sent_n"] if s["sent_n"] else 0):7.3f} '
          f'{(s["conf_sum"]/ev):7.3f}')
