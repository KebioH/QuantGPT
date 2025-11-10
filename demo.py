import os
from gnews import GNews
from transformers import pipeline
import pandas as pd
from openai import OpenAI


CSV_PATH   = r"NVDA.csv"   
STOCK_NAME = "NVDA"       
QUERY      = "NVDA"        
N_NEWS     = 90            
W1, W2     = 0.6, 0.4      


client = OpenAI()



def load_close_series(csv_path=CSV_PATH, lookback_days=1500):
    df = pd.read_csv(csv_path)
    date_col = None
    for c in df.columns:
        if str(c).lower() in {"date", "time", "交易日期"}:
            date_col = c
            break
    if date_col is None:
        raise ValueError("CSV 檔找不到日期欄")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    close_col = None
    for c in ["Close", "Adj Close", "Price", "收盤", "收盤價"]:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        raise ValueError("CSV 檔找不到收盤價欄")

    close = pd.to_numeric(df[close_col], errors="coerce").dropna()
    close = close.asfreq("B", method="pad")
    close = close.tail(lookback_days)
    return close.rename("Close")


def compute_technical_score(csv_path=CSV_PATH, lookback_days=1500):
    close = load_close_series(csv_path, lookback_days)

    ma_signal = 0
    if len(close) >= 60:
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        ma_signal = 1 if ma20.iloc[-1] > ma60.iloc[-1] else -1

    rsi_signal, rsi = 0, None
    if len(close) >= 15:
        chg = close.diff()
        avg_gain = chg.clip(lower=0).tail(14).mean()
        avg_loss = (-chg.clip(upper=0)).tail(14).mean()
        rsi = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
        if rsi < 30:
            rsi_signal = 2
        elif rsi < 50:
            rsi_signal = 1
        elif rsi < 60:
            rsi_signal = 0
        elif rsi < 70:
            rsi_signal = -1
        else:
            rsi_signal = -2

    macd_signal = 0
    if len(close) >= 26:
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_signal = 1 if macd.iloc[-1] > signal.iloc[-1] else -1

    sum_signal = ma_signal + rsi_signal + macd_signal
    tech_score = ((sum_signal + 3) / 6 * 100) if (ma_signal or rsi_signal or macd_signal) else 50.0

    detail = {
        "MA_signal": ma_signal,
        "RSI_value": None if rsi is None else round(float(rsi), 2),
        "RSI_signal": rsi_signal,
        "MACD_signal": macd_signal,
        "TechScore": round(float(tech_score), 2),
    }
    return tech_score, detail


def fetch_titles(query=QUERY, n=N_NEWS, language="en", country="US", period="30d"):
    googlenews = GNews(language=language, country=country, period=period, max_results=n)
    items = googlenews.get_news(query) or []
    seen, titles = set(), []
    for it in items:
        t = (it.get("title") or "").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            titles.append(t)
        if len(titles) >= n:
            break
    return titles


def finbert_sentiment_index(titles):
    if not titles:
        return 50.0, {"Positive": 0, "Neutral": 0, "Negative": 0}, []

    nlp = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
    results = nlp(titles, truncation=True)

    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    scores = []
    samples = []
    for t, r in zip(titles, results):
        label, score = r["label"], float(r["score"])
        counts[label] = counts.get(label, 0) + 1
        s = score if label == "Positive" else (0.0 if label == "Negative" else 0.5)
        scores.append(s)
        if len(samples) < 5:
            samples.append((t, label, round(score, 3)))

    sentiment_index = (sum(scores) / len(scores)) * 100
    return round(sentiment_index, 2), counts, samples


def explain_with_llm(user_query, stock_name, tech_detail,
                     sentiment_index, counts, samples,
                     buy_score, action, w1=W1, w2=W2):
    """
    把指標與最終決策丟給 ChatGPT，請它用中文整理成解釋與建議。
    """
  
    tech_block = (
        f"MA_signal: {tech_detail['MA_signal']}\n"
        f"RSI_value: {tech_detail['RSI_value']}\n"
        f"RSI_signal: {tech_detail['RSI_signal']}\n"
        f"MACD_signal: {tech_detail['MACD_signal']}\n"
        f"TechScore: {tech_detail['TechScore']}\n"
    )

    news_block = f"SentimentIndex: {sentiment_index}\nCounts: {counts}\n"
    if samples:
        news_block += "Sample news (title, label, score):\n"
        for t, lab, sc in samples:
            news_block += f"- {t[:120]} → {lab} ({sc})\n"

    final_block = (
        f"加權公式: Score = {w1} * TechScore + {w2} * SentimentIndex\n"
        f"TechScore = {tech_detail['TechScore']}\n"
        f"SentimentIndex = {sentiment_index}\n"
        f"Score = {round(buy_score, 2)}\n"
        f"最終機器決策(不可更改): {action}\n"
        f"決策門檻: Buy >= 50, Sell <= 30, 其他為 Hold\n"
    )

    user_input = f"""
        使用者原始問題：
        {user_query}

        標的：{stock_name}

        [技術指標]
        {tech_block}

        [新聞/情緒分析]
        {news_block}

        [加權分數與決策]
        {final_block}

        請用繁體中文輸出，格式分成三個段落：
        1. 數據摘要（先講技術面，再講情緒面）
        2. 目前建議：直接用「{action}」這個決策，說明為何是這個結論，不要自己改成別的
        3. 結合使用者的問題下最後結論

        語氣偏向專業、冷靜，不要太誇張；不要做任何保證報酬。
    """

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="你是一位專門分析美股與新聞情緒的投資顧問，回答一律使用繁體中文。",
        input=user_input,
    )

    return response.output_text



def main():
    user_query = input(f"請輸入你對 {STOCK_NAME} 的問題\n>> ")

    result = run_analysis(user_query)

    print("\n================= 原始數值 =================")
    print("Tech detail:", result["tech_detail"])
    print("SentimentIndex:", result["sentiment_index"])
    print("Counts:", result["counts"])
    print(f"BuyScore = {W1:.1f}*Tech({result['tech_score']:.2f}) + "
          f"{W2:.1f}*Sent({result['sentiment_index']:.2f}) = {result['buy_score']:.2f}")
    print("Machine Decision:", result["action"])

    print("\n================= QuantGPT 說明 =================")
    print(result["explanation"])



def run_analysis(user_query: str):

    tech_score, tech_detail = compute_technical_score()


    titles = fetch_titles()
    sentiment_index, counts, samples = finbert_sentiment_index(titles)


    buy_score = W1 * tech_score + W2 * sentiment_index

    def decision(score, buy_th=50, sell_th=30):
        if score >= buy_th:
            return "BUY"
        if score <= sell_th:
            return "SELL"
        return "HOLD"

    action = decision(buy_score)


    explanation = explain_with_llm(
        user_query=user_query,
        stock_name=STOCK_NAME,
        tech_detail=tech_detail,
        sentiment_index=sentiment_index,
        counts=counts,
        samples=samples,
        buy_score=buy_score,
        action=action,
        w1=W1,
        w2=W2,
    )


    return {
        "tech_score": tech_score,
        "tech_detail": tech_detail,
        "sentiment_index": sentiment_index,
        "counts": counts,
        "buy_score": buy_score,
        "action": action,
        "explanation": explanation,
    }


if __name__ == "__main__":
    main()
