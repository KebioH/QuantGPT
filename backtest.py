from demo import (
    load_close_series,
    compute_technical_score,
    W1,
    W2,
    finbert_sentiment_index,   # 你原本在 demo.py 裡的 FinBERT 函式
)
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt   # 畫圖用
from gnews import GNews           # 新增：抓新聞
import datetime

# ========= 交易紀錄資料結構 =========
@dataclass
class Trade:
    date: pd.Timestamp
    action: str     # "BUY" or "SELL"
    price: float
    shares: float


# ========= 技術指標（依指定日期切片） =========
def compute_technical_score_at_date(close: pd.Series, asof_date: pd.Timestamp):
    """
    用「asof_date 以前（含當天）」的資料計算 TechScore，避免前視。
    和 demo.py 的 compute_technical_score 邏輯一致，只是多了一個切片 asof_date。
    """
    sub = close.loc[:asof_date].copy()
    if len(sub) < 60:
        # 資料太短就回中立
        return 50.0, {"MA_signal": 0, "RSI_value": None, "RSI_signal": 0, "MACD_signal": 0, "TechScore": 50.0}

    # MA 訊號
    ma20 = sub.rolling(20).mean()
    ma60 = sub.rolling(60).mean()
    ma_signal = 1 if ma20.iloc[-1] > ma60.iloc[-1] else -1

    # RSI 訊號（14）
    chg = sub.diff()
    avg_gain = chg.clip(lower=0).tail(14).mean()
    avg_loss = (-chg.clip(upper=0)).tail(14).mean()
    rsi = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))

    # ✅ 你現在的 RSI 規則：只有超賣才給 +1，其他 0（不給空訊）

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

    # MACD 訊號
    ema12 = sub.ewm(span=12, adjust=False).mean()
    ema26 = sub.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_signal = 1 if macd.iloc[-1] > signal.iloc[-1] else -1

    sum_signal = ma_signal + rsi_signal + macd_signal
    tech_score = ((sum_signal + 3) / 6 * 100)

    detail = {
        "MA_signal": ma_signal,
        "RSI_value": round(float(rsi), 2),
        "RSI_signal": rsi_signal,
        "MACD_signal": macd_signal,
        "TechScore": round(float(tech_score), 2),
    }
    return float(tech_score), detail


# ========= 產生每週一的評估日期 =========
def weekly_dates(close: pd.Series):
    """
    產生近一年的「週一」評估日期；若當日非交易日，滾到下一個交易日。
    """
    start = close.index.max() - pd.DateOffset(years=2)
    # 先建出每週一
    mondays = pd.date_range(start=start.normalize(), end=close.index.max(), freq="W-MON")
    bd = pd.tseries.offsets.BusinessDay()
    eval_days = []
    for d in mondays:
        # 滾到「當週的第一個有資料的交易日」
        roll = d
        for _ in range(7):  # 最多往後找 1 週
            if roll in close.index:
                eval_days.append(roll)
                break
            roll = roll + bd
    # 去重且排序
    eval_days = sorted(set(eval_days))
    return eval_days


# ========= 情緒：依日期抓當週新聞 + FinBERT =========

_sentiment_cache = {}

def sentiment_for_date(d: pd.Timestamp, query: str = "NVDA", window_days: int = 3) -> float:
    """
    給 backtest 用的情緒指標：
    - 以日期 d 為中心，抓 d ± window_days 天的新聞
    - 用 demo.py 的 finbert_sentiment_index 算成 0~100 指數
    - 有簡單 cache，避免重複抓同一週的新聞
    """
    key = d.strftime("%Y-%m-%d")
    if key in _sentiment_cache:
        return _sentiment_cache[key]

    # 決定搜尋日期區間（Google News 以 date 物件為主）
    start = (d - pd.Timedelta(days=window_days)).date()
    end   = (d + pd.Timedelta(days=window_days)).date()

    googlenews = GNews(language="en", country="US", max_results=90)
    googlenews.start_date = start
    googlenews.end_date = end

    items = googlenews.get_news(query) or []
    titles = []
    seen = set()
    for it in items:
        t = (it.get("title") or "").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            titles.append(t)

    # 沒新聞 → 中立
    if not titles:
        idx = 50.0
        _sentiment_cache[key] = idx
        return idx

    # 用你原來 demo.py 裡的 FinBERT 函式
    sentiment_index, counts, samples = finbert_sentiment_index(titles)

    # 你可以開 debug 看看
    # print(f"[DEBUG] {key} sentiment={sentiment_index}, counts={counts}")

    _sentiment_cache[key] = sentiment_index
    return sentiment_index


# ========= 每週回測主程式 =========
def backtest_weekly(
    csv_path,
    w1=W1,
    w2=W2,
    buy_th=50,
    sell_th=30,
    use_sentiment=False,
    sentiment_provider=None,
    initial_capital=10000.0,
):
    """
    每週一評估一次策略，無手續費/滑價。
    ✅ 一有買入訊號就加碼買 1 股
    ✅ 一有賣出訊號就把手上全部賣掉
    ✅ 情緒可由 sentiment_provider(date) 提供（0~100）
    """
    close = load_close_series(csv_path, lookback_days=2000)  # 抓長一點保險

    # 僅取近一年「評估點」，但計算 TechScore 仍會用到 asof_date 前所有歷史
    start_eval = close.index.max() - pd.DateOffset(years=2)
    sub_close = close.loc[start_eval:]

    eval_days = weekly_dates(close)
    eval_days = [d for d in eval_days if d >= sub_close.index.min()]

    # 初始資金 & 部位
    cash = float(initial_capital)
    position = 0.0               # 持有股數
    position_cost = 0.0          # 持股成本總和（用來算平均成本）
    trades = []
    equity_curve = []
    realized_pnls = []           # 每次完全賣出的 realized PnL

    for d in eval_days:
        price = float(close.loc[d])

        # 技術分數（當日以前資料）
        tech_score, _ = compute_technical_score_at_date(close, d)

        # 情緒分數：用 sentiment_provider(date)
        if use_sentiment and callable(sentiment_provider):
            sent = float(sentiment_provider(d))
        else:
            sent = 50.0  # 中立

        buy_score = w1 * tech_score + w2 * sent

        # ✅ 買進邏輯：只要有買入訊號，就再買 1 股（可以一直累積部位）
        if buy_score >= buy_th:
            position += 1.0
            cash -= price
            position_cost += price
            trades.append(Trade(date=d, action="BUY", price=price, shares=1.0))

        # ✅ 賣出邏輯：有賣出訊號就全數賣出（如果有持股）
        elif buy_score <= sell_th and position > 0:
            avg_cost = position_cost / position
            pnl = (price - avg_cost) * position
            realized_pnls.append(pnl)

            cash += position * price
            trades.append(Trade(date=d, action="SELL", price=price, shares=position))

            position = 0.0
            position_cost = 0.0

        # 記錄當期資產（現金 + 部位市值）
        equity = cash + position * price
        equity_curve.append(
            {
                "date": d,
                "equity": equity,
                "price": price,
                "tech": tech_score,
                "sent": sent,
                "score": buy_score,
            }
        )

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = (
        pd.DataFrame([t.__dict__ for t in trades])
        if trades
        else pd.DataFrame(columns=["date", "action", "price", "shares"])
    )

    # 若最後還有部位，按最後一天收盤價強制平倉（只反映在 equity，不另外新增交易紀錄）
    if position > 0:
        last_p = float(close.iloc[-1])
        equity_df.iloc[-1, equity_df.columns.get_loc("equity")] = cash + position * last_p

    # 總報酬率：以 initial_capital 為基準
    if len(equity_df) >= 1:
        end_equity = float(equity_df["equity"].iloc[-1])
        total_return = (end_equity - initial_capital) / initial_capital
    else:
        end_equity = initial_capital
        total_return = 0.0

    # === 交易統計（只算真的有完整 SELL 的 round-trip） ===
    if realized_pnls:
        wins = sum(1 for x in realized_pnls if x > 0)
        losses = sum(1 for x in realized_pnls if x <= 0)
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else None
        avg_pnl = float(pd.Series(realized_pnls).mean())
    else:
        win_rate = None
        avg_pnl = 0.0

    summary = {
        "n_weeks": len(equity_df),
        "n_trades": len(trades_df),
        "n_round_trips": len(realized_pnls),
        "win_rate": None if win_rate is None else round(win_rate * 100, 2),
        "total_return_pct": round(total_return * 100, 2),
        "first_equity": round(float(initial_capital), 2),
        "last_equity": round(float(end_equity), 2),
        "avg_trade_pnl": round(avg_pnl, 4),
    }
    return summary, equity_df, trades_df


# ======= 範例執行 =======
if __name__ == "__main__":
    initial_capital = 10000.0  # 統一放在這裡，策略跟 DCA 都用同一個起始資金

    summary, equity, trades = backtest_weekly(
        csv_path="NVDA.csv",
        w1=W1,
        w2=W2,
        use_sentiment=False,           # 要用情緒就改 True
        sentiment_provider=sentiment_for_date,
        initial_capital=initial_capital
    )

    print("=== 回測摘要（近一年、每週一執行）===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    roi = summary["total_return_pct"]
    print(f"\n策略總投資報酬率（ROI）: {roi:.2f}%")

    print("\n=== 每一筆交易 ===")
    if trades.empty:
        print("本期無交易。")
    else:
        trades_sorted = trades.sort_values("date").reset_index(drop=True)
        print(trades_sorted.to_string(index=False))

    # ✅ 資產曲線：全部列出來
    print("\n=== 資產曲線（每一次評估點）===")
    if equity.empty:
        print("沒有資產資料。")
    else:
        print(equity.to_string())

    # ===== 計算 DCA 資產曲線（每週一固定買 1 股）=====
    dca_df = None
    if not equity.empty:
        close = load_close_series("NVDA.csv", lookback_days=2000)

        dca_cash = float(initial_capital)
        dca_pos = 0.0
        dca_curve = []

        # 用跟策略一樣的評估日期（equity.index）
        for d in equity.index:
            price = float(close.loc[d])

            # 每個評估日都買 1 股，不管好壞
            dca_pos += 1.0
            dca_cash -= price

            dca_equity = dca_cash + dca_pos * price
            dca_curve.append({"date": d, "equity_dca": dca_equity})

        dca_df = pd.DataFrame(dca_curve).set_index("date")

        # 印出 DCA 結果
        dca_final = dca_df["equity_dca"].iloc[-1]
        dca_roi = (dca_final - initial_capital) / initial_capital * 100
        print(f"\n=== DCA 策略（每週一買 1 股） ===")
        print(f"最後資產: {dca_final:.2f}")
        print(f"投資報酬率 ROI: {dca_roi:.2f}%")

    # ===== 圖1：策略資產曲線 + DCA 資產曲線 =====
    if not equity.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(equity.index, equity["equity"], label="Strategy Equity")
        #if dca_df is not None:
        #    plt.plot(dca_df.index, dca_df["equity_dca"],
        #             label="DCA Equity (每週買1股)", linestyle="--")
        plt.xlabel("Date")
        plt.ylabel("Equity (資產)")
        plt.title("Equity Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("\n[WARN] 沒有資產資料可以畫圖。")

    # ===== 圖2：股價 + 買賣點標記（你原本那段可以接在這下面）=====
    if not trades.empty:
        close = load_close_series("NVDA.csv", lookback_days=2000)
        start_date = equity.index.min()
        end_date   = equity.index.max()
        price_series = close.loc[start_date:end_date]

        plt.figure(figsize=(12, 5))
        plt.plot(price_series.index, price_series.values,
                 label="NVDA Close Price", color="gray")

        trades_sorted = trades.sort_values("date").reset_index(drop=True)

        for row in trades_sorted.itertuples():
            d = row.date
            p = row.price

            if row.action == "BUY":
                label = "Buy(1)"
                plt.scatter(d, p, marker="^", color="green", s=60)
                plt.annotate(label, (d, p),
                             textcoords="offset points",
                             xytext=(0, 8),
                             ha="center",
                             fontsize=8,
                             color="green")
            elif row.action == "SELL":
                label = f"Sell({int(row.shares)})"
                plt.scatter(d, p, marker="v", color="red", s=60)
                plt.annotate(label, (d, p),
                             textcoords="offset points",
                             xytext=(0, -12),
                             ha="center",
                             fontsize=8,
                             color="red")

        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("NVDA Price with Buy/Sell Signals (Share Count)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("\n[INFO] 沒有交易紀錄，所以不畫買賣點圖。")



