import os
from flask import Flask, request, render_template_string, url_for, send_from_directory
from demo import run_analysis, STOCK_NAME

app = Flask(__name__)

PAGE_TEMPLATE = """
<!doctype html>
<html lang="zh-Hant">
<head>
    <meta charset="utf-8">
    <title>QuantGPT</title>
    <style>
        body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
               max-width: 900px; margin: 40px auto; padding: 0 16px; background: #f7f7f7; }
        h1 { margin-bottom: 8px; }
        .card { background: white; padding: 16px 20px; border-radius: 10px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.08); margin-bottom: 16px; }
        label { font-weight: 600; }
        textarea { width: 100%; min-height: 80px; padding: 8px; font-size: 14px; }
        button { padding: 8px 16px; border-radius: 6px; border: none;
                 background: #2563eb; color: white; font-size: 14px; cursor: pointer; }
        button:hover { background: #1d4ed8; }
        .result-title { font-weight: 600; margin-top: 0; }
        .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                white-space: pre-wrap; font-size: 13px; background: #f3f4f6; padding: 8px 10px; border-radius: 6px; }
        .tag { display: inline-block; padding: 2px 8px; border-radius: 999px;
               font-size: 12px; margin-right: 4px; }
        .tag-buy { background: #dcfce7; color: #166534; }
        .tag-sell { background: #fee2e2; color: #b91c1c; }
        .tag-hold { background: #e5e7eb; color: #374151; }
        img { max-width: 100%; border-radius: 8px; margin-top: 8px; }
    </style>
</head>
<body>
    <h1>QuantGPT</h1>
    <p>結合技術分析、新聞情緒的智慧投資系統，以{{ stock_name }}為核心標的，讓AI為你讀懂市場脈動。</p>

    <div class="card">
        <form method="post">
            <label for="q">請輸入你對 {{ stock_name }} 的問題：</label><br>
            <textarea id="q" name="q" placeholder="詢問任何問題">{{ user_query or '' }}</textarea>
            <br><br>
            <button type="submit">送出分析</button>
        </form>
    </div>

    {% if result %}
    <div class="card">
        <p class="result-title">
            分數與決策：
            {% if result.action == 'BUY' %}
                <span class="tag tag-buy">BUY</span>
            {% elif result.action == 'SELL' %}
                <span class="tag tag-sell">SELL</span>
            {% else %}
                <span class="tag tag-hold">HOLD</span>
            {% endif %}
        </p>
        <div class="mono">
TechScore       : {{ "%.2f"|format(result.tech_score) }}
SentimentIndex  : {{ "%.2f"|format(result.sentiment_index) }}
BuyScore        : {{ "%.2f"|format(result.buy_score) }}
Decision Rule   : BUY ≥ 50, SELL ≤ 30, else HOLD
        </div>
    </div>

    <div class="card">
        <p class="result-title">技術指標細節</p>
        <div class="mono">
{{ tech_detail_text }}
        </div>
    </div>

    <div class="card">
        <p class="result-title">QuantGPT 說明</p>
        <div class="mono">
{{ result.explanation }}
        </div>
    </div>
    {% endif %}

    {% if image_files %}
    <div class="card">
        <p class="result-title">回測與示意圖</p>
        {% for img in image_files %}
            <div>
                <img src="{{ url_for('image_file', filename=img) }}" alt="Result Image">
            </div>
        {% endfor %}
    </div>
    {% endif %}

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    user_query = ""
    result = None
    tech_detail_text = ""
    image_files = []

    if request.method == "POST":
        user_query = request.form.get("q", "").strip()
        if user_query:
            result = run_analysis(user_query)
            td = result["tech_detail"]
            tech_detail_text = (
                f"MA_signal : {td['MA_signal']}\n"
                f"RSI_value : {td['RSI_value']}\n"
                f"RSI_signal: {td['RSI_signal']}\n"
                f"MACD_signal: {td['MACD_signal']}\n"
                f"TechScore : {td['TechScore']}"
            )

            candidate_imgs = ["demo.png", "backtest1.png", "backtest2.png"]
            image_files = [name for name in candidate_imgs if os.path.exists(name)]

    return render_template_string(
        PAGE_TEMPLATE,
        stock_name=STOCK_NAME,
        user_query=user_query,
        result=result,
        tech_detail_text=tech_detail_text,
        image_files=image_files,
    )


@app.route("/images/<path:filename>")
def image_file(filename):
    # '.' 表示 app.py 所在的資料夾
    return send_from_directory(".", filename)


if __name__ == "__main__":
    app.run(debug=True)
