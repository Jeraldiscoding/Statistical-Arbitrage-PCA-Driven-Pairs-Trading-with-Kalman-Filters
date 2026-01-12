# Statistical Arbitrage: PCA-Driven Pairs Trading with Kalman Filters

A quantitative trading system that identifies and exploits price inefficiencies between statistically similar asset pairs.  
The system combines **factor-based discovery via Principal Component Analysis (PCA)** with **dynamic hedge ratio estimation using a Kalman Filter**, enabling adaptive, market-neutral trading.

---

## Project Overview

This project implements a **market-neutral pairs trading strategy** based on the principle of mean reversion.  
Assets with highly similar factor exposures are expected to maintain a stable long-term relationship. When short-term deviations occur, the system enters trades betting on a reversion to equilibrium.

The framework is designed to be **modular, reproducible, and out-of-sample validated**, mirroring industry research workflows.

---

## Core Components

- **Factor-Based Pair Discovery**  
  Uses PCA to identify statistically similar assets based on shared exposure to latent market and sector factors.

- **Dynamic Hedging**  
  Employs a recursive Kalman Filter to estimate a time-varying hedge ratio, allowing the strategy to adapt to changing market regimes.

- **Market Neutrality**  
  Maintains simultaneous long and short positions to minimize directional market exposure.

- **Out-of-Sample Validation**  
  Applies a strict **80/20 Train/Test split** to evaluate robustness and reduce overfitting.

---

## Methodology & Technical Intuition

### 1. Pair Discovery via Principal Component Analysis (PCA)

Instead of relying on simple correlation, the system identifies pairs using shared factor structure.

**Process:**
- **Standardization**: Asset returns are Z-scored to normalize volatility across the universe.
- **Eigen-Decomposition**: The covariance matrix of returns is decomposed into principal components (latent risk factors).
- **Factor Space Mapping**: Assets are represented by their factor loadings.
- **Pair Selection**: The top 5 asset pairs with the smallest **Euclidean distance** in factor space are selected as statistical "twins".

This approach improves robustness compared to raw correlation-based pairing.

---

### 2. Dynamic Modeling with the Kalman Filter

Static hedge ratios often fail as market relationships evolve.  
This system models the relationship:

Y_t = β_t · X_t + α_t

using a Kalman Filter.

**Key Properties:**
- **Recursive Updates**: Model parameters (α, β) are updated daily as new data arrives
- **Noise Filtering**: Separates the underlying relationship from short-term price noise
- **Adaptive Hedging**: Allows the hedge ratio to evolve across regimes


---

## 3. Trading Signal Generation (Z-Score)

The trading signal is derived from the **spread**, defined as the Kalman Filter residual:

Spread_t = Y_t − (β_t · X_t + α_t)

The spread is standardized into a rolling Z-score.

**Trading Rules:**

- **Entry**
  - Z > 2.0 → Short spread (Short Y, Long X)
  - Z < -2.0 → Long spread (Long Y, Short X)

- **Exit**
  - Position closed when Z-score reverts to 0

- **Risk Management**
  - Stop-loss triggered if |Z| ≥ 4.0, protecting against structural breakdowns


---

## 4. Backtesting & Return Calculation

Performance is evaluated using **percentage-based returns** to ensure comparability across assets.

**Return Logic:**

- **Long Spread (Buy Y, Sell X)**  
  Return = Return_Y − Return_X

- **Short Spread (Sell Y, Buy X)**  
  Return = Return_X − Return_Y


**Portfolio Aggregation:**
- Equal-weighted allocation across all active pairs
- Aggregated portfolio return computed daily

---

## Performance Metrics

The system reports the following metrics on the **out-of-sample test set**:

- **Cumulative Return (%)**  
  Total growth of allocated capital
- **Sharpe Ratio**  
  Risk-adjusted return measure
- **Maximum Drawdown**  
  Largest peak-to-trough decline

---

## Requirements

- Python 3.x  
- NumPy  
- Pandas  
- Matplotlib  
- yFinance  
- SciPy  
- PyKalman  

---

## How to Run

1. **Configure Asset Universe**  
   Modify the `test_tickers` list in the main script.

2. **Set Date Range**  
   Define the historical period for data collection.

3. **Execute the Pipeline**  
   Run the main script to perform:
   - Training Phase (PCA-based pair discovery)
   - Testing Phase (out-of-sample backtest)

4. **Visualization**  
   The script generates:
   - Equity curves
   - Z-score signal plots
   - PCA scree plot

---

## Notes

This project is intended for **research and educational purposes only**.  
It does not account for transaction costs, slippage, or market impact, which are critical considerations in live trading systems.

---
