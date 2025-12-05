import pandas as pd
from backtest import weekly_dates, sentiment_for_date
from demo import load_close_series
import config
import os

CSV_PATH = os.path.join("historical_price", rf"{config.STOCK_SYMBOL}.csv")   

def build_sentiment_cache():
    close = load_close_series(CSV_PATH)
    eval_days = weekly_dates(close)

    cache = []
    for d in eval_days:
        sent = sentiment_for_date(d)
        print(f"{d.date()} → sentiment={sent:.2f}")
        cache.append({"date": d, "sentiment": sent})

    df = pd.DataFrame(cache)
    df.to_csv(os.path.join("sentiment_cache", rf"{config.STOCK_SYMBOL}_sentiment_cache.csv"), index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    build_sentiment_cache()
