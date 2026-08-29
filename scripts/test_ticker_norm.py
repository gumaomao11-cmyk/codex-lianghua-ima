import sys; sys.path.insert(0, r"F:\even-codex\lianghua+IMA")
from ticker_norm import norm_ticker, clean_events, universe
print("universe size:", len(universe()))
print()
cases = ["NVDA","nvda","NVDA.US","NASDAQ:NVDA"," MU ","英伟达","闪迪","默沙东","相干",
         "腾讯","长鑫科技","HXTX","ROBLOX","艺电","TAKETWO","SEA","AAOI","","中际旭创"]
for c in cases:
    print(f"  {c!r:16} -> {norm_ticker(c)}")
print()
bad = [
 {"ticker":"NVDA","text_type":"research_report","materiality_tier":"tership_1_hard_data",
  "sentiment_score":0.8,"expected_horizon_days":20,"confidence":0.9,"evidence":"x"},
 {"ticker":"MU","text_type":"single_event","materiality_tier":"ticker_3_macro_industry",
  "sentiment_score":1.7,"expected_horizon_days":99,"confidence":2,"evidence":"y"},
 {"ticker":"腾讯","text_type":"news_summary","materiality_tier":"tier_1_hard_data",
  "sentiment_score":0.5,"expected_horizon_days":1,"confidence":0.5,"evidence":"z"},
 {"ticker":"ROBLOX","text_type":"news_summary","materiality_tier":"tier_2_soft_logic",
  "sentiment_score":0.3,"expected_horizon_days":1,"confidence":0.4,"evidence":"w"},
 {"ticker":"AAPL","text_type":"noise","materiality_tier":"tier_3_macro_industry",
  "sentiment_score":0,"expected_horizon_days":0,"confidence":0.1,"evidence":"chat"},
 {"ticker":"MRK","text_type":"resarch_report","materiality_tier":"tior_3_macro_industry",
  "sentiment_score":"0.6","expected_horizon_days":"20","confidence":"0.7","evidence":"v"},
]
clean, st = clean_events(bad)
print("stats:", st)
for e in clean:
    print("  ", e["ticker"], e["text_type"], e["materiality_tier"],
          e["sentiment_score"], e["expected_horizon_days"], e["confidence"])
