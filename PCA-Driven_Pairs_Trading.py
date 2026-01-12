import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from pykalman import KalmanFilter


class DataLoader:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
    
    def download_data(self):
        print(f"Downloading data for {len(self.tickers)} tickers...")

        raw_data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date
        )

        # Use closing prices and drop assets with insufficient data
        self.data = raw_data['Close'].dropna(
            axis=1,
            thresh=int(len(raw_data) * 0.0)
        )

        print(f"Download complete. Shape: {self.data.shape}")
        return self.data
    
    def get_returns(self):
        if self.data is None:
            return None
        
        # Log returns are time-additive and scale-invariant
        return np.log(self.data / self.data.shift(1)).dropna()



class FactorModel:
    """
    Principal Component Analysis on standardized returns to extract
    common risk factors driving asset co-movements.
    """

    def __init__(self, n_comp=3):
        self.n_comp = n_comp
        self.eigenvectors = None
        self.eigenvalues = None
        self.loadings = None

    def fit(self, returns):
        # Standardize returns to remove volatility bias
        mu = returns.mean()
        sigma = returns.std()
        standardized_returns = (returns - mu) / sigma

        # Covariance matrix of standardized returns
        cov_matrix = standardized_returns.cov()

        # Eigen-decomposition (ascending order)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort components by explained variance (descending)
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        # Factor loadings for top principal components
        self.loadings = pd.DataFrame(
            self.eigenvectors[:, :self.n_comp],
            index=returns.columns,
            columns=[f'PC{i+1}' for i in range(self.n_comp)]
        )

        # Explained variance diagnostics
        total_variance = np.sum(self.eigenvalues)
        explained_variance_ratio = self.eigenvalues / total_variance

        print("\n--- PCA Factor Analysis (Training Set) ---")
        total_sum = 0
        for i in range(self.n_comp):
            var_pct = explained_variance_ratio[i] * 100
            total_sum += var_pct
            print(f"PC{i+1}: {var_pct:.2f}% variance")
        print(f"Total variance explained: {total_sum:.2f}%")

        return self.loadings


def find_best_pair(loadings, n=5):
    """
    Identify statistically similar assets by minimizing
    Euclidean distance in factor-loading space.
    """
    stocks = loadings.index
    pairs = []

    for i in range(len(stocks)):
        for j in range(i + 1, len(stocks)):
            s1, s2 = stocks[i], stocks[j]
            dist = euclidean(loadings.loc[s1], loadings.loc[s2])
            pairs.append((s1, s2, dist))

    pairs.sort(key=lambda x: x[2])
    return pairs[:n]


def run_kalman_filter(x, y):
    """
    Estimates a time-varying relationship:
        y_t = beta_t * x_t + alpha_t
    using a Kalman Filter.
    """

    delta = 1e-5
    transition_covariance = delta / (1 - delta) * np.eye(2)

    # Observation matrix includes price and intercept term
    obs_mat = np.vstack([x.values, np.ones_like(x.values)]).T
    obs_mat = np.expand_dims(obs_mat, axis=1)

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=[1, 0],
        initial_state_covariance=np.ones((2, 2)),
        observation_covariance=2,
        observation_matrices=obs_mat,
        transition_matrices=np.eye(2),
        transition_covariance=transition_covariance
    )

    state_means, _ = kf.filter(y.values)

    beta = state_means[:, 0]
    alpha = state_means[:, 1]

    # Spread (model residual)
    spread = y - (beta * x + alpha)

    # Rolling Z-score of spread
    window = 30
    mean_spread = spread.rolling(window).mean()
    std_spread = spread.rolling(window).std()
    z_score = (spread - mean_spread) / std_spread

    return z_score, state_means


def run_backtest(z_score, price_y, price_x):
    pnl = []
    in_trade = 0
    entry_y, entry_x = 0, 0
    stop_loss_threshold = 4.0

    for i in range(len(z_score)):
        current_z = z_score.iloc[i]
        today_return = 0

        if in_trade == 0:
            if current_z > 2.0:
                in_trade = -1
                entry_y, entry_x = price_y.iloc[i], price_x.iloc[i]
            elif current_z < -2.0:
                in_trade = 1
                entry_y, entry_x = price_y.iloc[i], price_x.iloc[i]

        elif in_trade == 1:
            if current_z >= 0 or current_z < -stop_loss_threshold:
                profit_y = (price_y.iloc[i] - entry_y) / entry_y
                profit_x = (entry_x - price_x.iloc[i]) / entry_x
                today_return = profit_y + profit_x
                in_trade = 0

        elif in_trade == -1:
            if current_z <= 0 or current_z > stop_loss_threshold:
                profit_y = (entry_y - price_y.iloc[i]) / entry_y
                profit_x = (price_x.iloc[i] - entry_x) / entry_x
                today_return = profit_y + profit_x
                in_trade = 0

        pnl.append(today_return)

    return pd.Series(pnl, index=z_score.index).cumsum() * 100


def calculate_metrics(cumulative_pnl):
    daily_pnl = cumulative_pnl.diff().fillna(0)

    sharpe = 0
    if daily_pnl.std() != 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)

    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    max_drawdown = drawdown.min()

    return sharpe, max_drawdown


# User-defined universe of tradable assets
# The list can be modified to include different sectors, regions, or asset classes
test_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AVGO', 'ADBE', 'CRM',
    'SHEL', 'BP', 'XOM', 'CVX', 'TTE', 'COP', 'SLB', 'EOG', 'PBR', 'VLO'
]

# Historical window (adjustable)
loader = DataLoader(test_tickers, "2020-01-01", "2024-12-31")
prices = loader.download_data()
returns = loader.get_returns()


# 2. TRAIN / TEST SPLIT
# 80% training period for factor discovery and pair selection
# 20% testing period reserved for out-of-sample evaluation
split_point = int(len(prices) * 0.8)

train_prices = prices.iloc[:split_point]
test_prices = prices.iloc[split_point:]

train_returns = returns.iloc[:split_point]
test_returns = returns.iloc[split_point:]


# 3. TRAINING PHASE (PAIR DISCOVERY)
# PCA is fit ONLY on training data to avoid look-ahead bias
mfm_train = FactorModel(n_comp=3)
train_loadings = mfm_train.fit(train_returns)

# Identify statistically similar asset pairs in factor space
top_5_pairs = find_best_pair(train_loadings, n=5)

print("\n--- Training Phase Complete ---")
print("Top 5 Factor-Based Pairs:")
for i, (s1, s2, dist) in enumerate(top_5_pairs):
    print(f"{i+1}. {s2} vs {s1} (Distance: {dist:.4f})")


# 4. TESTING PHASE (OUT-OF-SAMPLE)
plt.figure(figsize=(14, 7))
all_pair_returns = []

print("\n--- Out-of-Sample Performance ---")

for s1, s2, dist in top_5_pairs:
    # Generate trading signals using the Kalman Filter
    z_score_test, _ = run_kalman_filter(test_prices[s1], test_prices[s2])
    
    # Backtest the pair using percentage-based return differences
    pair_cum_ret = run_backtest(
        z_score_test,
        test_prices[s2],
        test_prices[s1]
    )
    
    # Performance metrics
    sharpe, mdd = calculate_metrics(pair_cum_ret)
    final_ret = pair_cum_ret.iloc[-1]
    
    print(
        f"Pair {s2}/{s1} | "
        f"Return: {final_ret:>7.2f}% | "
        f"Sharpe: {sharpe:>5.2f} | "
        f"MaxDD: {mdd:>7.2f}%"
    )
    
    # Plot individual pair equity curve
    plt.plot(
        pair_cum_ret,
        label=f"{s2}/{s1} ({final_ret:.1f}%)",
        alpha=0.8
    )
    
    all_pair_returns.append(pair_cum_ret)


# Equal-weighted portfolio across all selected pairs
portfolio_return = pd.concat(all_pair_returns, axis=1).mean(axis=1)

plt.plot(
    portfolio_return,
    color='black',
    linewidth=3,
    label='Portfolio (Equal-Weighted)'
)

plt.title("Top 5 Pairs: Cumulative Returns (Out-of-Sample)")
plt.ylabel("Cumulative Return (%)")
plt.xlabel("Date")
plt.axhline(0, color='black', lw=1, alpha=0.5)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.show()


# --- PCA Scree Plot ---
total_var = np.sum(mfm_train.eigenvalues)
var_exp = mfm_train.eigenvalues / total_var

plt.figure(figsize=(10, 4))
plt.bar(
    range(1, len(var_exp) + 1),
    var_exp,
    alpha=0.7,
    label='Individual'
)
plt.step(
    range(1, len(var_exp) + 1),
    np.cumsum(var_exp),
    where='mid',
    label='Cumulative',
    color='red'
)
plt.title('PCA Scree Plot (Training Data)')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Component')
plt.legend()
plt.show()


# --- Z-Score Signal Example ---
# Illustrates entry and exit thresholds for the top-ranked pair
best_s1, best_s2, _ = top_5_pairs[0]
best_z, _ = run_kalman_filter(
    test_prices[best_s1],
    test_prices[best_s2]
)

plt.figure(figsize=(12, 4))
plt.plot(best_z, label=f"Z-Score {best_s2}/{best_s1}", color='royalblue')
plt.axhline(2.0, color="red", linestyle='--', label="Short Spread")
plt.axhline(-2.0, color='green', linestyle='--', label="Long Spread")
plt.axhline(0, color='black', alpha=0.3)
plt.title(f"Trading Signals: {best_s2} vs {best_s1}")
plt.legend(loc='lower left')
plt.show()
