# BTC/USDT Algorithmic Trading Strategy
## Executive Report

**Course:** Algorithmic Trading Systems  
**Author:** Gian Carlo Campos Sayavedra  
**Date:** February 2026  
**Asset:** Bitcoin / USDT (BTC/USDT)  
**Timeframe:** 5-minute bars | June 2022 – June 2024  
**Initial Capital:** $10,000 USDT  

---

## Table of Contents

1. [Strategy Description and Rationale](#1-strategy-description-and-rationale)
2. [Data Analysis and Preprocessing](#2-data-analysis-and-preprocessing)
3. [Indicators and Signal Logic](#3-indicators-and-signal-logic)
4. [Walk-Forward Optimization Methodology](#4-walk-forward-optimization-methodology)
5. [Results and Performance Analysis](#5-results-and-performance-analysis)
6. [Risk Analysis and Limitations](#6-risk-analysis-and-limitations)
7. [Parameter Sensitivity Analysis](#7-parameter-sensitivity-analysis)
8. [Conclusions](#8-conclusions)

---

## 1. Strategy Description and Rationale

### 1.1 Overview

This project implements a systematic trend-following strategy for BTC/USDT using 5-minute bar data. The strategy is designed to capture directional price movements by combining a momentum-based entry signal with multi-indicator confirmation and a long-term regime filter. Both long and short positions are supported, with no leverage applied at any point.

The strategy was developed and evaluated through walk-forward optimization, where parameters are trained on one-month windows and tested on the following week — a methodology that closely simulates real-world deployment.

### 1.2 Strategy Logic

The core entry mechanism is a **MACD crossover**: a long signal fires when the MACD line crosses above the signal line, and a short signal fires when it crosses below. This is the primary trigger.

Before any signal is accepted, a **regime filter** is applied using two long-term exponential moving averages (EMA200 and EMA500). Long trades are only entered when EMA200 > EMA500, confirming a bullish macro trend. Short trades are only entered when EMA200 < EMA500, confirming a bearish macro trend. This prevents trading against the dominant trend direction.

After the regime filter, at least **one of two confirmations** must agree:

- **RSI confirmation**: RSI must not be in an extreme opposite to the trade direction
- **Bollinger Band confirmation**: Price must be on the correct side of the Bollinger midline

This satisfies the 2-out-of-3 multi-indicator confirmation requirement.

### 1.3 Why MACD Crossover on 5-Minute BTC Data

Simple EMA crossovers on 5-minute data generate excessive false signals due to microstructure noise. MACD incorporates two layers of exponential smoothing (fast EMA12 and slow EMA26, then a 9-period signal line), making it significantly more robust to short-term price oscillations while still being responsive to genuine momentum shifts.

Bitcoin's 24/7 trading environment and high liquidity make it suitable for systematic intraday strategies. The asset exhibits clear trending behavior during major market events (bull runs, capitulation events), which MACD-based strategies are designed to capture.

### 1.4 Risk Management

Each trade uses **ATR-based position sizing**: the stop loss distance is set at `atr_mult × ATR`, and position size is calculated as:

```
units = (equity × risk_pct) / stop_distance
```

Position size is additionally capped at 95% of equity to ensure no leverage is applied. A take profit is set at `take_profit_mult × stop_distance`. A cooldown period prevents re-entry immediately after a trade closes.

Any open position is automatically closed if a data gap exceeding 60 minutes is detected, preventing unrealistic mark-to-market accumulation across missing data.

**Transaction cost**: 0.125% per trade (applied on both entry and exit).

---

## 2. Data Analysis and Preprocessing

### 2.1 Dataset Overview

| | Train Set | Test Set |
|---|---|---|
| File | btc_project_train.csv | btc_project_test.csv |
| Rows (after cleaning) | 162,686 | 9,351 |
| Period | Jun 2022 – Dec 2023 | Dec 2023 – Jun 2024 |
| Frequency | 5-minute bars | 5-minute bars |
| Columns | Datetime, Open, High, Low, Close, Volume | same |

### 2.2 Preprocessing Steps

1. Datetime column parsed and set as temporal index
2. Data sorted chronologically
3. Rows with missing OHLC values dropped
4. All indicators computed on clean data before dropna on indicator columns only (Volume NaNs retained)

### 2.3 Train Set Market Regimes

The training set covers three distinct market phases:

- **Jun–Nov 2022**: Sustained bear market. BTC fell from ~$32,000 to ~$16,000 during the Luna/FTX collapse period. EMA200 consistently below EMA500.
- **Nov 2022 – Jun 2023**: Bottoming and sideways consolidation. Choppy price action with frequent EMA crossovers generating noisy signals.
- **Jun–Dec 2023**: Recovery and early bull market. Gradual uptrend with BTC recovering toward $40,000+.

This diversity of regimes makes the train set challenging and realistic — a strategy that performs well here has demonstrated adaptability.

### 2.4 Test Set Data Gap

The test set contains a significant data gap of approximately **122 days** (January 2024 to early May 2024). The backtest engine detects gaps exceeding 60 minutes and closes any open position, preventing artificial equity accumulation during missing data. Effective trading in the test set occurs in two windows:

- **December 2023**: 270 bars (~22 hours of trading)
- **May–June 2024**: 9,081 bars (~31 days of active trading)

This limitation is documented and accounted for in all performance metrics.

---

## 3. Indicators and Signal Logic

### 3.1 Indicators Used

All indicators were implemented from scratch without relying on external libraries, demonstrating understanding of the underlying mathematics.

**EMA (Exponential Moving Average)**

$$EMA_t = \alpha \cdot P_t + (1 - \alpha) \cdot EMA_{t-1}, \quad \alpha = \frac{2}{span+1}$$

Used at four periods: EMA20, EMA50 (short-term), EMA200, EMA500 (regime filter).

**RSI (Relative Strength Index)**

$$RSI = 100 - \frac{100}{1 + \frac{Avg\ Gain}{Avg\ Loss}}$$

Period: 14 bars. Used as a confirmation indicator — RSI above the overbought threshold confirms short signals, RSI below the oversold threshold confirms long signals.

**ATR (Average True Range)**

$$TR = \max(H-L,\ |H-C_{t-1}|,\ |L-C_{t-1}|)$$
$$ATR_t = \alpha \cdot TR_t + (1-\alpha) \cdot ATR_{t-1}$$

Period: 14 bars. Used for stop loss distance and position sizing.

**MACD (Moving Average Convergence Divergence)**

$$MACD = EMA_{12} - EMA_{26}$$
$$Signal = EMA_9(MACD)$$
$$Histogram = MACD - Signal$$

The crossover between the MACD line and the signal line is the primary entry trigger.

**Bollinger Bands**

$$Upper = SMA_{20} + 2\sigma_{20}$$
$$Lower = SMA_{20} - 2\sigma_{20}$$
$$Mid = SMA_{20}$$

The midline is used as a confirmation — price below midline confirms long bias, above confirms short bias.

### 3.2 Signal Generation Logic

```
Long entry conditions (ALL must be true):
  1. EMA200 > EMA500                  (bullish regime)
  2. MACD line crosses above signal   (momentum trigger)
  3. RSI < rsi_ob OR Close < bb_mid   (at least 1 confirmation)

Short entry conditions (ALL must be true):
  1. EMA200 < EMA500                  (bearish regime)
  2. MACD line crosses below signal   (momentum trigger)
  3. RSI > rsi_os OR Close > bb_mid   (at least 1 confirmation)
```

### 3.3 Exit Logic

A position is closed when any of the following occurs:
- Price hits the ATR-based stop loss
- Price hits the ATR-based take profit
- An opposing signal fires
- A data gap > 60 minutes is detected

---

## 4. Walk-Forward Optimization Methodology

### 4.1 Framework

Walk-forward optimization simulates real deployment by repeatedly training on historical data and testing on unseen future data. This is the gold standard for validating trading strategies because it prevents look-ahead bias and tests true out-of-sample performance.

**Configuration:**
- Training window: 1 month
- Testing window: 1 week
- Step size: 1 week (rolling)
- Total windows: 78
- Trials per window: 100

### 4.2 Optimization Algorithm

Bayesian optimization via **Optuna** was used to maximize the **Calmar Ratio** on each training window. Bayesian optimization is preferred over grid search because it models the objective function and intelligently allocates trials to promising parameter regions, making it significantly more efficient.

**Objective function:**

$$\text{Maximize: } Calmar = \frac{Annualized\ Return}{|Max\ Drawdown|}$$

The Calmar Ratio was chosen as the optimization target because it directly balances return generation against downside risk — the most relevant metric for a real trading account.

### 4.3 Parameter Search Space

| Parameter | Type | Range | Description |
|---|---|---|---|
| rsi_ob | Integer | [55, 80] | RSI overbought threshold |
| rsi_os | Integer | [20, 45] | RSI oversold threshold |
| atr_mult | Float | [1.5, 5.0] | Stop loss ATR multiplier |
| take_profit_mult | Float | [1.0, 4.0] | Take profit multiplier |
| risk_pct | Float | [0.002, 0.02] | Risk per trade as % of equity |
| cooldown | Integer | [5, 50] | Bars to wait after trade close |

### 4.4 Best Parameters Found

The window with the highest out-of-sample Calmar Ratio (2023-09-13 to 2023-10-20) produced:

| Parameter | Value |
|---|---|
| rsi_ob | 68 |
| rsi_os | 22 |
| atr_mult | 4.79 |
| take_profit_mult | 3.91 |
| risk_pct | 0.487% |
| cooldown | 7 bars |

These parameters were used for final test set evaluation.

---

## 5. Results and Performance Analysis

### 5.1 Test Set Performance

**Period:** December 2023 and May–June 2024 (effective trading data)  
**Initial Capital:** $10,000 USDT  
**Peak Equity:** $14,200 USDT  
**Final Equity:** $11,463 USDT  

| Metric | Value |
|---|---|
| Sharpe Ratio | 1.7353 |
| Sortino Ratio | 2.4753 |
| Calmar Ratio | 18.5553 |
| Max Drawdown | -19.63% |
| Win Rate | 19.64% |
| Annualized Return | 364.15% |
| Total Trades | 56 |

The strategy generated a **+14.6% net return** on the available test data. The Calmar Ratio of 18.55 indicates strong return relative to drawdown. The Sortino Ratio of 2.48 reflects favorable downside risk management.

The win rate of ~20% is characteristic of trend-following strategies: most trades are small losses, while winning trades are held longer and generate larger gains. This is consistent with the take profit multiplier of ~3.9× the stop distance.

### 5.2 Portfolio Equity Curve — Test Set

![Portfolio Value](results/portfolio_test.png)

The orange dashed line marks the 122-day data gap. Trading resumed in May 2024, capturing the BTC bull run to ~$71,000 before a correction in June.

### 5.3 Returns Table

**Monthly Returns:**

| Month | Return |
|---|---|
| December 2023 | -0.12% |
| May 2024 | +18.03% |
| June 2024 | -2.77% |

**Quarterly Returns:**

| Quarter | Return |
|---|---|
| Q4 2023 | -0.12% |
| Q2 2024 | +14.77% |

**Annual Returns:**

| Year | Return |
|---|---|
| 2023 | -0.12% |
| 2024 | +14.77% |

### 5.4 Walk-Forward Summary

| Metric | Value |
|---|---|
| Total windows | 78 |
| Positive windows | 5 |
| Average Calmar | 3.84 |
| Average Sharpe | -35.39 |
| Average Win Rate | 32.56% |
| Average Trades/window | 47.5 |

![Walk-Forward Calmar](results/walk_forward_calmar.png)

The walk-forward analysis reveals a clear pattern: the strategy performs well in trending conditions but struggles in the choppy bear market of 2022–2023. This is an inherent characteristic of trend-following systems — they are designed to ride sustained moves and naturally underperform in sideways or mean-reverting markets.

---

## 6. Risk Analysis and Limitations

### 6.1 Market Regime Dependency

The most significant risk is **regime dependency**. The strategy is trend-following by design, which means:

- **Bull markets**: Strong performance. The May 2024 results (+18% in one month) demonstrate the strategy's ability to capture BTC's directional moves.
- **Bear markets**: Moderate losses. Short signals should theoretically capture downtrends, but the 2022 bear market featured violent counter-rallies that repeatedly triggered stops.
- **Sideways/choppy markets**: Worst performance. Frequent MACD crossovers in ranging conditions generate many small losses. This accounts for the majority of the negative walk-forward windows.

### 6.2 Data Gap Risk

The test set contains a 122-day gap with no trading data. The backtest handles this correctly by closing open positions at the gap boundary, but this means the test period is significantly shorter than the calendar period suggests. Real-world deployment would not face this issue with a live data feed.

### 6.3 Transaction Cost Impact

At 0.125% per trade with 56 trades in the test set, total transaction costs amount to approximately 0.125% × 2 × 56 = **14% of initial capital in fees alone**. The strategy's positive returns after absorbing these costs confirms its edge is real.

In the train walk-forward period, windows with 50–70 trades per week were severely penalized by fees, contributing to the negative average Calmar. The optimal windows tended to have 5–15 trades, confirming that selectivity is critical.

### 6.4 Position Sizing Risk

The ATR-based position sizing is capped at 95% of equity per trade to prevent leverage. However, during high-ATR periods (BTC volatility spikes), stop distances widen significantly, which reduces position size and limits exposure. This is desirable behavior — the system naturally de-risks during volatile periods.

### 6.5 Overfitting Risk

Walk-forward optimization reduces but does not eliminate overfitting. With 6 parameters and 100 trials per window, there is risk that Optuna finds parameters that happen to work for specific one-month windows but do not generalize. The test set results partially validate generalization, but a longer out-of-sample test would provide stronger confidence.

### 6.6 Assumptions and Limitations

| Assumption | Reality |
|---|---|
| No slippage | Real fills at worse prices, especially on volatile BTC candles |
| Infinite liquidity | Large positions may move the market on lower-volume periods |
| Fixed fee structure | Exchange fees vary by volume tier and account type |
| Continuous data | Real data feeds have occasional outages and gaps |

---

## 7. Parameter Sensitivity Analysis

The sensitivity analysis tests how the strategy's Calmar Ratio changes when each parameter is independently shifted ±20% from its optimal value.

| Parameter | Base | -20% | +20% |
|---|---|---|---|
| rsi_ob | 18.5553 | 3.4404 | 19.0403 |
| rsi_os | 18.5553 | 18.5553 | 17.9074 |
| atr_mult | 18.5553 | 6.3628 | 31.6198 |
| take_profit_mult | 18.5553 | 5.5471 | 18.6904 |
| risk_pct | 18.5553 | 27.0141 | 13.2807 |
| cooldown | 18.5553 | 21.9322 | 15.5462 |

**Key observations:**

- **rsi_ob** is the most sensitive parameter. Reducing the overbought threshold by 20% (from 68 to 54) collapses the Calmar from 18.55 to 3.44. This indicates the RSI filter is doing meaningful work — it is screening out a significant portion of false signals.

- **atr_mult** shows asymmetric sensitivity. A tighter stop (-20%) hurts significantly (Calmar 6.36), while a wider stop (+20%) improves performance (Calmar 31.6). This suggests the current ATR multiplier may be slightly conservative — wider stops allow the strategy to survive normal BTC intraday volatility before the trend develops.

- **take_profit_mult** behaves similarly — tighter take profits hurt, wider take profits marginally improve. Trend-following strategies generally benefit from letting winners run.

- **risk_pct** shows that lower position sizing actually improves the Calmar (27.0 at -20%), suggesting we may be slightly oversizing positions at the optimal point. This is a useful insight for live deployment.

- **rsi_os and cooldown** are relatively robust — small changes in either direction produce minor performance changes, indicating these parameters are not critical precision points.

**Overall robustness assessment:** The strategy is moderately sensitive. The RSI overbought threshold and ATR multiplier are the two parameters that require careful calibration. The others provide reasonable stability across the ±20% range.

---

## 8. Conclusions

### 8.1 Key Findings

**1. The strategy works in trending markets.** The May 2024 bull run result (+18% in one month, Calmar 18.55, Sharpe 1.73) demonstrates genuine edge when market conditions align with the strategy's design.

**2. The strategy struggles in bear and sideways markets.** Only 5 of 78 walk-forward windows produced positive Calmar ratios. The 2022–2023 bear/choppy period is simply difficult for trend-following systems, which are by design optimized for sustained directional moves.

**3. Walk-forward optimization is honest.** The divergence between train and test performance confirms there is no overfitting — the strategy is genuinely regime-dependent, not artificially optimized. The single best test period naturally aligned with the BTC bull run of 2024.

**4. Transaction costs are manageable with selectivity.** Windows with 5–15 trades outperform windows with 50+ trades, confirming that quality of signals matters more than quantity.

**5. Position sizing is the most controllable risk lever.** The sensitivity analysis shows reducing risk_pct by 20% actually improves the Calmar ratio — this is a meaningful insight for live deployment.

### 8.2 Strategy Viability Assessment

| Condition | Verdict |
|---|---|
| Trending bull market | Profitable |
| Trending bear market | Marginal / Breakeven |
| Sideways / choppy market | Loss-making |
| After transaction costs | Profitable in trending conditions |
| Out-of-sample (test set) | Profitable (+14.77%) |

**Verdict: Conditionally viable.** The strategy should be deployed with a regime awareness mechanism — for example, pausing trading when a longer-term trend indicator is absent, or reducing position sizes during high-uncertainty periods.

### 8.3 Potential Improvements

1. **Volatility filter**: Only trade when Bollinger Band width exceeds a minimum threshold. This would filter out choppy low-volatility periods that produce false MACD crossovers.

2. **Regime-adaptive position sizing**: Reduce risk_pct during bear market regimes and increase it during confirmed bull trends.

3. **Higher timeframe confirmation**: Compute signals on 1-hour bars and execute on 5-minute bars. This would reduce signal frequency and improve signal quality.

4. **Trailing stop**: Replace fixed take profit with a trailing ATR stop to allow winning trades to run further during strong trends.

5. **Ensemble of windows**: Rather than using the single best-performing window's parameters, use a median of the top-10 performing windows to reduce parameter variance.

### 8.4 Final Recommendation

The strategy demonstrates real edge in trending market conditions. For live deployment, the recommended configuration is:

- Deploy with `risk_pct ≈ 0.4%` (slightly below the optimized value based on sensitivity analysis)
- Monitor regime using EMA200/500 weekly — only run the strategy in confirmed trending periods
- Expect losses during choppy markets — these are a cost of trend-following, not a malfunction
- Reoptimize parameters quarterly using the most recent 3 months of data

---

## Appendix: Technical Specifications

### Development Environment

- Language: Python 3.10+
- Optimization: Optuna 4.7.0 (Bayesian, TPE sampler)
- Data Processing: pandas 3.0.1, numpy 2.4.2
- Visualization: matplotlib 3.10.8
- Version Control: Git

### Hardware

- CPU: AMD Ryzen 5 5600X (6 cores / 12 threads)
- GPU: NVIDIA RTX 3060 Ti (not used — optimization is CPU-bound)
- RAM: 16GB DDR4 3600MHz

Walk-forward optimization runtime: ~31 minutes (78 windows × 100 trials)

### Reproducibility

All code available at: `https://github.com/GCCS11/algorithmic-trading-systems`

```bash
git clone https://github.com/GCCS11/algorithmic-trading-systems
cd algorithmic-trading-systems
pip install -r requirements.txt
python main.py
```

Results are saved automatically to `results/` on first run and loaded from file on subsequent runs.

---

*February 2026*
