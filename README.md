```text
QuantGPT/
|
├── historical_price/        # 存放多支股票歷史價格.csv
├── sentiment_cache/         # 過往日期情緒分析建表.csv
├── montecarlo_result/       # 蒙地卡羅模擬輸出
├── static/backtests/	       # 儲存來自app.py回測輸出的圖表檔案
│
├── app.py                   # Flask前後端整合，執行此檔即可開啟網頁介面和所有功能
│
├── demo.py                  # 計算技術面/情緒面並生成LLM分析報告，有需要再執行
│
├── montecarlo.py            # 蒙地卡羅分析，有需要再執行
│
├── backtest.py              # 不需執行
├── sentiment_cache.py       # 不需執行
├── config.py                # 不需執行
│
└── README.md



買入邏輯：當分數達到門檻時買入1股

賣出邏輯：觸發賣出訊號時全數清倉

代辦事項
    股價波動自動調整交易權重
    W1, W2參數自動最佳化
