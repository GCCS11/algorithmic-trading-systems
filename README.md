# Algorithmic Trading System — BTC/USDT 5-Minute

Individual project for the Algorithmic Trading Systems course (ITESO).

## Overview

A systematic trading strategy built on 5-minute BTC/USDT data. The strategy uses a MACD crossover as the primary entry signal, confirmed by RSI and Bollinger Bands, with a long-term EMA regime filter to avoid trading against the trend. Position sizing is ATR-based and hyperparameters are optimized using Bayesian optimization (Optuna) via walk-forward analysis.

## Strategy Logic

- **Regime filter**: EMA200 vs EMA500 determines trend direction
- **Entry signal**: MACD line crosses signal line in the direction of the trend
- **Confirmation**: 1 of 2 must agree — RSI not in opposite extreme, or price on correct side of Bollinger mid
- **Exit**: ATR-based stop loss, take profit, or opposing signal
- **Position sizing**: Risk 0.5–2% of equity per trade, capped at 95% of equity (no leverage)
- **Transaction cost**: 0.125% per trade

## Project Structure
```
├── data/
│   ├── btc_project_train.csv
│   └── btc_project_test.csv
├── results/
├── src/
│   ├── data_loader.py
│   ├── indicators.py
│   ├── strategy.py
│   └── optimizer.py
├── main.py
└── requirements.txt
```

## How to Run
```bash
pip install -r requirements.txt
python main.py
```

The first run executes the full walk-forward optimization (~30 minutes). Results are saved to `results/` and loaded automatically on subsequent runs.

## Results

| Metric | Test Set |
|---|---|
| Sharpe Ratio | 1.74 |
| Sortino Ratio | 2.48 |
| Calmar Ratio | 18.56 |
| Max Drawdown | -19.63% |
| Win Rate | 19.64% |
| Annual Return | 364% |
| Total Trades | 56 |

Walk-forward optimization: 78 windows, 1 month train / 1 week test, 100 Optuna trials per window.

## Requirements

- Python 3.10+
- pandas, numpy, matplotlib, optuna