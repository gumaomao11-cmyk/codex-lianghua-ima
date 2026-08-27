# -*- coding: utf-8 -*-
"""文本-动能因子快速基线：词典情绪 + TF-IDF 新颖度。
输出 text_sentiment_lexicon.csv（可被回测脚本复用）。
"""
import json, re, sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROJ = Path(r"F:\even-codex\lianghua+IMA")
OUT = _PROJ / "backtest_output"

CN2T = {
    "英伟达":"NVDA","超威半导体":"AMD","AMD":"AMD","美光":"MU","西部数据":"WDC","希捷":"STX",
    "迈威尔":"MRVL","应用材料":"AMAT","康宁":"GLW","伟创力":"FLEX","戴尔":"DELL",
    "Arm":"ARM","新易盛":"EOPT","博通":"AVGO","台积电":"TSM","高通":"QCOM",
    "英特尔":"INTC","超微":"SMCI","Meta":"META","苹果":"AAPL","微软":"MSFT",
    "亚马逊":"AMZN","特斯拉":"TSLA","谷歌":"GOOGL","礼来":"LLY","Moderna":"MRNA",
    "默沙东":"MRK","雅培":"ABT","奈飞":"NFLX","优步":"UBER","Roblox":"RBLX",
    "Palantir":"PLTR","Coinbase":"COIN","纳斯达克":"QQQ","阿里巴巴":"BABA","拼多多":"PDD",
    "京东":"JD","网易":"NTES","百度":"BIDU","腾讯":"TCEHY","美团":"MPNGF",
}
US_CODES = {"NVDA","AMD","INTC","MU","WDC","STX","MRVL","AMAT","GLW","FLEX","DELL","NBIS","ARM",
            "AVGO","QCOM","SMCI","META","AAPL","MSFT","GOOGL","AMZN","TSLA","LLY","MRNA","MRK",
            "ABT","NFLX","UBER","RBLX","PLTR","COIN","TSM","BABA","PDD","JD","NTES","BIDU"}

POS = ["上调","增持","买入","跑赢","超预期","强劲","增长","突破","创新高","利好","看涨","买入评级","Outperform","Buy","Overweight","看好","乐观","复苏","回暖","扩张","超预期","盈利增长","上调目标价"]
NEG = ["下调","减持","卖出","跑输","低于预期","疲软","下滑","跌破","利空","看跌","卖出评级","Underperform","Sell","Underweight","看空","悲观","衰退","恶化","亏损","暴雷","裁员","诉讼","调查","违约","破产","做空","召回","监管重罚"]
RISK = ["裁员","SEC","诉讼","调查","违约","破产","做空","召回","财务重述","业绩暴雷","监管重罚","违约","欺诈"]
NOISE_TITLES = ["中国收盘","韩国收盘","收盘点评","上证","深证","沪深","KOSPI","A股","港股","恒生","沪指","深指","两市","创业板指","科创板","上证指数"]

def clean_text(text):
    text = text or ""
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_tickers(text):
    found = set()
    for m in re.finditer(r"\b([A-Z]{1,6})\b", text or ""):
        if m.group(1) in US_CODES: found.add(m.group(1))
    for cn, tk in CN2T.items():
        if len(cn) >= 2 and cn in (text or ""): found.add(tk)
    return sorted(found)

def is_noise(title, text):
    t = (title or "") + " " + (text or "")
    return any(kw in t for kw in NOISE_TITLES)

def lexicon_sentiment(text):
    text = text or ""
    pos = sum(text.count(w) for w in POS)
    neg = sum(text.count(w) for w in NEG)
    total = pos + neg
    if total == 0: return 0.0
    return (pos - neg) / total

def risk_flag(text):
    text = text or ""
    return int(any(w in text for w in RISK))

def parse_date(ts):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", ts or "")
    return m.group(0) if m else ""

def main():
    src = "浑水调研Plus"
    path = OUT / f"zsxq_{src}.json"
    if not path.exists(): path = _PROJ / "data" / f"zsxq_{src}.json"
    items = json.loads(path.read_text(encoding="utf-8"))

    records = []
    for it in items:
        title = it.get("title", "")
        content = it.get("content", "")
        full = clean_text(title + " " + content)
        if not full: continue
        if is_noise(title, full): continue
        tickers = extract_tickers(full)
        if not tickers: continue
        d = parse_date(it.get("create_time") or "")
        if not d: continue
        records.append({"media_id": str(it.get("topic_id","")), "date": d, "tickers": tickers, "text": full})

    # dedup
    seen = set(); recs = []
    for r in records:
        if r["media_id"] in seen: continue
        seen.add(r["media_id"]); recs.append(r)
    records = recs
    print(f"records after filtering: {len(records)}")

    # score
    rows = []
    texts_for_tfidf = []
    for r in records:
        sent = lexicon_sentiment(r["text"])
        risk = risk_flag(r["text"])
        for tk in r["tickers"]:
            rows.append({"media_id": r["media_id"], "date": r["date"], "ticker": tk,
                         "sentiment": sent, "risk_flag": risk, "text": r["text"][:300]})
        texts_for_tfidf.append(r["text"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # TF-IDF novelty per ticker
    print("computing TF-IDF novelty...")
    df["novelty"] = 0.0
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(texts_for_tfidf)
    # map each row to its text index
    text_idx_map = {rec["media_id"]: i for i, rec in enumerate(records)}
    # compute per-ticker rolling novelty: 1 - cosine similarity with previous 7 days
    for tk, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        nov = []
        for i in range(len(g)):
            idx = text_idx_map[g.loc[i, "media_id"]]
            if i == 0:
                nov.append(1.0)
            else:
                # find most recent previous row within 7 days
                prev_idx = None
                for j in range(i-1, -1, -1):
                    if (g.loc[i, "date"] - g.loc[j, "date"]).days <= 7:
                        prev_idx = text_idx_map[g.loc[j, "media_id"]]
                        break
                if prev_idx is not None:
                    from sklearn.metrics.pairwise import cosine_similarity
                    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix[prev_idx])[0,0]
                    nov.append(max(0.0, 1.0 - sim))
                else:
                    nov.append(1.0)
        df.loc[g.index, "novelty"] = nov

    out_path = OUT / f"text_sentiment_lexicon_{src}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved {out_path}: {len(df)} rows")
    print(f"sentiment: mean={df['sentiment'].mean():.3f}, std={df['sentiment'].std():.3f}")
    print(f"novelty: mean={df['novelty'].mean():.3f}, std={df['novelty'].std():.3f}")
    print(f"risk rows: {df['risk_flag'].sum()}")
    print("top tickers:")
    print(df["ticker"].value_counts().head(10))

if __name__ == "__main__":
    main()
