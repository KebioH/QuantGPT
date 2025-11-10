import random
import pandas as pd
import matplotlib.pyplot as plt
from backtest import backtest_weekly

CSV_PATH = "NVDA.csv"
N_RUNS = 200
INITIAL_CAPITAL = 10000

def sample_weights():
    """隨機產生 W1, W2（總和固定為 1）"""
    w1 = random.random()
    w2 = 1.0 - w1
    return w1, w2

def main():
    records = []

    for i in range(N_RUNS):
        w1, w2 = sample_weights()
        summary, _, _ = backtest_weekly(
            csv_path=CSV_PATH,
            w1=w1,
            w2=w2,
            use_sentiment=False,
            initial_capital=INITIAL_CAPITAL,
        )

        records.append({
            "run": i,
            "w1": round(w1, 3),
            "w2": round(w2, 3),
            "roi_pct": summary["total_return_pct"],
            "win_rate": summary["win_rate"],
            "n_trades": summary["n_trades"],
        })

    df = pd.DataFrame(records).sort_values("roi_pct", ascending=False)
    print("=== Top 10 組權重（依 ROI 排序）===")
    print(df.head(10).to_string(index=False))
    print("\n=== ROI 統計 ===")
    print(df["roi_pct"].describe())


    plt.figure(figsize=(7, 4))
    plt.hist(df["roi_pct"], bins=20, color="#4CAF50", edgecolor="black", alpha=0.8)
    plt.title("Distribution of ROI (%)", fontsize=14)
    plt.xlabel("ROI (%)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("montecarlo_roi_distribution.png", dpi=150)
    plt.show()


    plt.figure(figsize=(7, 4))
    plt.scatter(df["w1"], df["roi_pct"], alpha=0.7, c=df["roi_pct"], cmap="viridis", edgecolors="black")
    plt.title("ROI vs W1 (W2 = 1 - W1)", fontsize=14)
    plt.xlabel("W1 (Technical Weight)")
    plt.ylabel("ROI (%)")
    plt.grid(alpha=0.3)
    plt.colorbar(label="ROI (%)")
    plt.tight_layout()
    plt.savefig("montecarlo_roi_vs_w1.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
