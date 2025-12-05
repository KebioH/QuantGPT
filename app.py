import os
from flask import Flask, request, render_template_string, redirect, session, url_for
import matplotlib
matplotlib.use("Agg")  # Flask 環境用非互動 backend
import matplotlib.pyplot as plt

from demo import run_analysis, load_close_series, W1, W2
from backtest import backtest_weekly, sentiment_for_date  # 你的 backtest.py
import config

app = Flask(__name__)
app.secret_key = "MY_SECRET_KEY"

# ==========================================
# 共用設定：ChatGPT 風格 CSS 與 Header
# ==========================================
COMMON_HEAD = """
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>QuantGPT 智能分析</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            /* ChatGPT 經典深色配色 */
            --bg-body: #343541;       /* 主畫面背景 (深灰) */
            --bg-sidebar: #202123;    /* 側邊欄背景 (更深灰) */
            --card-bg: #444654;       /* 卡片/對話框背景 (稍亮灰) */
            --accent-color: #10a37f;  /* ChatGPT 標誌性綠色 */
            --accent-hover: #1a7f64;  /* 按鈕懸停深綠 */
            --text-main: #ececf1;     /* 主要文字 (灰白) */
            --text-muted: #c5c5d2;    /* 次要文字 */
            --border-color: #565869;  /* 邊框顏色 */
        }

        body {
            background-color: var(--bg-body);
            color: var(--text-main);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* 卡片樣式 - 模仿 GPT 對話塊 */
        .quant-card {
            background-color: var(--card-bg);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 2rem;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
        }

        /* 輸入框樣式 */
        .form-control {
            background-color: #40414f;
            border: 1px solid rgba(32,33,35,0.5);
            color: var(--text-main);
            border-radius: 6px;
            box-shadow: 0 0 0 0 rgba(255,255,255,0);
        }
        
        .form-control:focus {
            background-color: #40414f;
            color: white;
            border-color: var(--border-color);
            box-shadow: none;
        }
        
        .form-control::placeholder {
            color: #8e8ea0;
        }

        .input-group-text {
            background-color: #40414f;
            border: 1px solid rgba(32,33,35,0.5);
            border-right: none;
            color: #8e8ea0;
        }
        
        .input-group .form-control {
            border-left: none;
        }

        .btn-quant {
            background-color: var(--accent-color);
            color: white;
            font-weight: 500;
            border: none;
            border-radius: 4px;
            transition: background-color 0.2s;
        }
        .btn-quant:hover {
            background-color: var(--accent-hover);
            color: white;
        }
        
        .navbar-custom {
            background-color: var(--bg-sidebar);
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        #loading-overlay {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(52, 53, 65, 0.9);
            z-index: 9999;
            display: none;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        
        .font-mono {
            font-family: 'Söhne Mono', 'Monaco', 'Andale Mono', 'Ubuntu Mono', monospace;
        }
    </style>
</head>
"""

# ==========================================
# STEP 1: 輸入股票代號頁面
# ==========================================
INPUT_SYMBOL_PAGE = """
<!doctype html>
<html>
""" + COMMON_HEAD + """
<body class="justify-content-center align-items-center">
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-6 col-lg-5">
                <div class="text-center mb-5">
                    <div style="width: 60px; height: 60px; background: var(--accent-color); border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                        <i class="fa-solid fa-bolt fa-2x text-white"></i>
                    </div>
                    <h2 class="mt-3 fw-bold">QuantGPT</h2>
                    <p class="text-muted">智能量化分析助手</p>
                </div>

                <div class="quant-card">
                    <form method="post">
                        <div class="mb-4">
                            <label class="form-label text-muted small fw-bold">輸入股票代號</label>
                            <div class="input-group">
                                <span class="input-group-text"><i class="fa-solid fa-magnifying-glass"></i></span>
                                <input type="text" name="symbol" class="form-control form-control-lg" 
                                       placeholder="例如：AAPL, TSMC, NVDA, META" value="{{ symbol or '' }}" required autofocus>
                            </div>
                        </div>
                        <button type="submit" class="btn btn-quant w-100 py-2">
                            開始分析 <i class="fa-solid fa-arrow-right ms-2"></i>
                        </button>
                    </form>

                    {% if error %}
                    <div class="alert alert-danger mt-4 d-flex align-items-center" role="alert" style="background-color: rgba(220, 53, 69, 0.1); border: 1px solid rgba(220, 53, 69, 0.3); color: #ea868f;">
                        <i class="fa-solid fa-circle-exclamation me-2"></i>
                        <div>{{ error }}</div>
                    </div>
                    {% endif %}
                </div>
                
            </div>
        </div>
    </div>
</body>
</html>
"""

# ==========================================
# STEP 2: 分析結果頁面（含回測）
# ==========================================
MAIN_PAGE = """
<!doctype html>
<html>
""" + COMMON_HEAD + """
<body>
    <!-- 頂部導航欄 -->
    <nav class="navbar navbar-custom px-4 py-2">
        <div class="d-flex align-items-center">
            <span class="fs-5 fw-bold tracking-tight text-white">QuantGPT</span>
            <span class="mx-3 text-muted">|</span>
            <span class="text-muted small">當前分析模型 v4.0</span>
        </div>
        <div class="d-flex align-items-center">
            <div class="d-flex align-items-center bg-dark rounded px-3 py-1 me-3 border border-secondary">
                <span class="font-mono text-white fw-bold">標的：</span>
                <span class="font-mono text-white fw-bold">{{ stock }}</span>
            </div>
            <a href="/" class="btn btn-outline-secondary btn-sm text-light" style="border-color: #565869;">
                <i class="fa-solid fa-arrow-right-from-bracket me-1"></i> 更換股票
            </a>
        </div>
    </nav>

    <div class="container-fluid flex-grow-1 p-0">
        <div class="row h-100 g-0">
            <!-- 左側：輸入區 -->
            <div class="col-lg-3 d-flex flex-column p-4" style="background-color: var(--bg-sidebar); border-right: 1px solid rgba(255,255,255,0.1);">
                <h5 class="mb-4 text-white"><i class="fa-regular fa-message me-2"></i>分析師指令</h5>
                
                <div class="flex-grow-1">
                    <form method="post" onsubmit="document.getElementById('loading-overlay').style.display = 'flex';">
                        <div class="mb-3">
                            <label class="form-label text-muted small">你想了解什麼？</label>
                            <textarea name="question" class="form-control" rows="8" placeholder="例如：請分析近 30 天的趨勢，或是目前的 RSI 指標是否過熱？">{{ question or '' }}</textarea>
                        </div>
                        <div class="d-grid gap-2">
                            <button type="submit" class="btn btn-quant w-100 py-2" name="action" value="analyze">
                                <i class="fa-solid fa-paper-plane me-2"></i> 發送指令
                            </button>
                            <button type="submit" class="btn btn-secondary w-100 py-2" name="action" value="backtest">
                                <i class="fa-solid fa-rotate-right me-2"></i> 執行回測
                            </button>
                        </div>
                    </form>

                    <div class="mt-5">
                        <p class="small text-muted mb-2 fw-bold">試試看這樣問：</p>
                        <div class="d-grid gap-2">
                            <button class="btn btn-sm btn-outline-secondary text-start text-light border-secondary" style="font-size: 0.85rem;" onclick="pasteText('分析目前的支撐位與壓力位')">
                                <i class="fa-regular fa-comment-dots me-2"></i> "分析支撐位與壓力位"
                            </button>
                            <button class="btn btn-sm btn-outline-secondary text-start text-light border-secondary" style="font-size: 0.85rem;" onclick="pasteText('MACD 指標目前呈現什麼訊號？')">
                                <i class="fa-regular fa-comment-dots me-2"></i> "MACD 指標訊號"
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 右側：主畫面 -->
            <div class="col-lg-9 d-flex flex-column p-4">
                <div class="d-flex justify-content-center h-100">
                    <div class="w-100" style="max-width: 900px;">
                        
                        {% if not result and not backtest_summary %}
                            <!-- 初始空白狀態 -->
                            <div class="h-100 d-flex flex-column justify-content-center align-items-center text-muted opacity-50">
                                <div style="width: 60px; height: 60px; background: #444654; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                                    <i class="fa-solid fa-wand-magic-sparkles fa-xl"></i>
                                </div>
                                <h4>QuantGPT 準備就緒</h4>
                                <p>請在左側輸入問題或直接執行回測</p>
                            </div>
                        {% endif %}

                        {% if result %}
                            <!-- AI 分析回答 -->
                            <div class="d-flex gap-4 mb-4">
                                <div class="flex-shrink-0">
                                    <div style="width: 30px; height: 30px; background: #19c37d; border-radius: 2px; display: flex; align-items: center; justify-content: center;">
                                        <i class="fa-solid fa-robot text-white small"></i>
                                    </div>
                                </div>
                                <div class="flex-grow-1">
                                    <h6 class="fw-bold mb-3">QuantGPT 分析報告</h6>
                                    <div class="quant-card font-mono" style="line-height: 1.8; color: #d1d5db; font-size: 0.95rem;">
                                        <div style="white-space: pre-wrap;">{{ result.explanation }}</div>
                                    </div>
                                </div>
                            </div>
                        {% endif %}

                        {% if backtest_summary %}
                            <!-- 回測結果摘要 -->
                            <div class="d-flex gap-4 mb-4">
                                <div class="flex-shrink-0">
                                    <div style="width: 30px; height: 30px; background: #2dd4bf; border-radius: 2px; display: flex; align-items: center; justify-content: center;">
                                        <i class="fa-solid fa-chart-line text-white small"></i>
                                    </div>
                                </div>
                                <div class="flex-grow-1">
                                    <h6 class="fw-bold mb-3">策略回測結果</h6>
                                    <div class="quant-card font-mono" style="line-height: 1.8; color: #d1d5db; font-size: 0.9rem;">
                                        <div>總週數 (n_weeks): {{ backtest_summary.n_weeks }}</div>
                                        <div>交易次數 (n_trades): {{ backtest_summary.n_trades }}</div>
                                        <div>勝率 (win_rate): {{ backtest_summary.win_rate }}%</div>
                                        <div>總報酬率 (ROI): {{ backtest_summary.total_return_pct }}%</div>
                                        <div>初始資金 (first_equity): {{ backtest_summary.first_equity }}</div>
                                        <div>期末資金 (last_equity): {{ backtest_summary.last_equity }}</div>
                                        <div>平均每筆損益 (avg_trade_pnl): {{ backtest_summary.avg_trade_pnl }}</div>
                                    </div>
                                </div>
                            </div>

                            {% if backtest_charts %}
                                <div class="quant-card mt-3">
                                    <h6 class="fw-bold mb-3">回測圖表</h6>
                                    {% for cf in backtest_charts %}
                                        <div class="mb-3">
                                            <img src="{{ url_for('static', filename=cf) }}" class="img-fluid rounded border border-secondary" />
                                        </div>
                                    {% endfor %}
                                </div>
                            {% endif %}
                        {% endif %}

                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- 載入中動畫 -->
    <div id="loading-overlay">
        <div class="d-flex flex-column align-items-center">
            <div class="spinner-border text-success mb-3" role="status" style="color: var(--accent-color) !important;"></div>
            <div class="text-white fw-bold">正在思考中...</div>
            <div class="text-muted small mt-1">分析大量市場數據</div>
        </div>
    </div>

    <script>
        function pasteText(text) {
            document.querySelector('textarea[name="question"]').value = text;
        }
    </script>
</body>
</html>
"""

# ==========================================
# 後端邏輯
# ==========================================
@app.route("/", methods=["GET", "POST"])
def select_symbol():
    if request.method == "POST":
        symbol = request.form.get("symbol", "").upper().strip()
        price_csv = os.path.join("historical_price", f"{symbol}.csv")

        if not os.path.exists(price_csv):
            return render_template_string(
                INPUT_SYMBOL_PAGE,
                symbol=symbol,
                error=f"錯誤：找不到 '{symbol}' 的歷史資料！請確認 historical_price 資料夾。"
            )

        session["symbol"] = symbol
        config.STOCK_SYMBOL = symbol
        return redirect("/main")

    return render_template_string(INPUT_SYMBOL_PAGE)



@app.route("/main", methods=["GET", "POST"])
def main_page():
    symbol = session.get("symbol", None)
    if not symbol:
        return redirect("/")

    question = ""
    result = None
    backtest_summary = None
    backtest_charts = []

    if request.method == "POST":
        action = request.form.get("action", "analyze")
        question = request.form.get("question", "")

        # ---------------------------------
        # 只有按「分析」才執行 run_analysis
        # ---------------------------------
        if action == "analyze" and question.strip():
            result = run_analysis(question)

        # ---------------------------------
        # 回測模式：完全不跑分析
        # ---------------------------------
        if action == "backtest":
            price_csv = os.path.join("historical_price", f"{symbol}.csv")

            summary, equity_df, trades_df = backtest_weekly(
                csv_path=price_csv,
                w1=W1,
                w2=W2,
                use_sentiment=True,
                sentiment_provider=lambda d: sentiment_for_date(d, query=symbol),
                initial_capital=10000.0,
            )

            from types import SimpleNamespace
            backtest_summary = SimpleNamespace(**summary)

            out_dir = os.path.join("static", "backtests")
            os.makedirs(out_dir, exist_ok=True)

            # 圖1：Equity Curve
            if not equity_df.empty:
                eq_filename = f"{symbol}_equity.png"
                eq_path = os.path.join(out_dir, eq_filename)
                plt.figure(figsize=(10, 5))
                plt.plot(equity_df.index, equity_df["equity"])
                plt.title(f"{symbol} Strategy Equity Curve")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(eq_path, dpi=150)
                plt.close()
                backtest_charts.append(f"backtests/{eq_filename}")

            # 圖2：Signals
            if not trades_df.empty:
                close = load_close_series(price_csv, lookback_days=2000)
                start, end = equity_df.index.min(), equity_df.index.max()
                price_series = close.loc[start:end]

                sig_filename = f"{symbol}_signals.png"
                sig_path = os.path.join(out_dir, sig_filename)

                plt.figure(figsize=(12, 5))
                plt.plot(price_series.index, price_series.values, color="gray")
                for row in trades_df.itertuples():
                    plt.scatter(row.date, row.price,
                                color="green" if row.action == "BUY" else "red")
                plt.title(f"{symbol} Buy/Sell Signals")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(sig_path, dpi=150)
                plt.close()
                backtest_charts.append(f"backtests/{sig_filename}")

    return render_template_string(
        MAIN_PAGE,
        stock=symbol,
        question=question,
        result=result,
        backtest_summary=backtest_summary,
        backtest_charts=backtest_charts,
    )



if __name__ == "__main__":
    app.run(debug=True, port=5000)
