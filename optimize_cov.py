from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from backtest_utils import PerformanceMetrics, RollingWindowBacktest


TRADING_DAYS_PER_YEAR = 252
MONTHLY_REBALANCE_DAYS = 21
DEFAULT_ETF_RETURNS_PATH = "etf_returns.csv"
DEFAULT_STOCK_RETURNS_PATH = "stock_returns.csv"


def make_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(symmetric))
    if min_eig < epsilon:
        symmetric = symmetric + np.eye(symmetric.shape[0]) * (epsilon - min_eig)
    return symmetric


def load_aligned_datasets(
    etf_returns_path: str = DEFAULT_ETF_RETURNS_PATH,
    stock_returns_path: str = DEFAULT_STOCK_RETURNS_PATH,
) -> pd.DataFrame:
    etf_returns = pd.read_csv(etf_returns_path, index_col=0, parse_dates=True).sort_index()
    stock_returns = pd.read_csv(stock_returns_path, index_col=0, parse_dates=True).sort_index()

    common_dates = etf_returns.index.intersection(stock_returns.index)
    etf_returns = etf_returns.loc[common_dates].copy()
    stock_returns = stock_returns.loc[common_dates].copy()
    combined_returns = pd.concat([etf_returns, stock_returns], axis=1)
    return combined_returns


def sample_covariance(train_returns: pd.DataFrame) -> np.ndarray:
    return make_psd(train_returns.cov().values)


def shrinkage_covariance(
    train_returns: pd.DataFrame,
    shrinkage_strength: float = 0.35,
) -> np.ndarray:
    sample = sample_covariance(train_returns)
    n_assets = sample.shape[0]
    target_scale = np.trace(sample) / n_assets
    target = target_scale * np.eye(n_assets)

    sigma = cp.Variable((n_assets, n_assets), PSD=True)
    objective = cp.Minimize(
        cp.sum_squares(sigma - sample)
        + shrinkage_strength * cp.sum_squares(sigma - target)
    )
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS, verbose=False)

    if sigma.value is None:
        raise ValueError("cvxpy failed to estimate the shrinkage covariance matrix.")

    return make_psd(np.asarray(sigma.value))


def l2_regularized_covariance(
    train_returns: pd.DataFrame,
    l2_penalty: float = 0.10,
) -> np.ndarray:
    sample = sample_covariance(train_returns)
    n_assets = sample.shape[0]
    identity = np.eye(n_assets)
    sigma = cp.Variable((n_assets, n_assets), PSD=True)
    objective = cp.Minimize(
        cp.sum_squares(sigma - sample) + l2_penalty * cp.sum_squares(sigma @ identity)
    )
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS, verbose=False)

    if sigma.value is None:
        raise ValueError("cvxpy failed to estimate the L2-regularized covariance matrix.")

    sigma_l2 = np.asarray(sigma.value) + l2_penalty * identity
    return make_psd(sigma_l2)


def build_default_views(
    train_returns: pd.DataFrame,
    n_views: int = 3,
    signal_lookback: int = MONTHLY_REBALANCE_DAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recent_signal = train_returns.tail(signal_lookback).mean().values * TRADING_DAYS_PER_YEAR
    n_assets = train_returns.shape[1]
    n_views = min(n_views, n_assets)
    selected_idx = np.argsort(np.abs(recent_signal))[-n_views:]

    P = np.zeros((n_views, n_assets))
    q = np.zeros(n_views)
    train_variance = train_returns.var().values
    omega_diag = np.zeros(n_views)
    for i, idx in enumerate(selected_idx):
        P[i, idx] = 1.0
        q[i] = recent_signal[idx]
        omega_diag[i] = max(train_variance[idx] * 0.05, 1e-6)

    omega = np.diag(omega_diag)
    return P, q, omega


def compute_metrics(results: dict[str, dict[str, object]]) -> pd.DataFrame:
    metrics = {}
    for name, strategy_results in results.items():
        metrics[name] = PerformanceMetrics(
            portfolio_values=np.asarray(strategy_results["values"]),
            returns=np.asarray(strategy_results["returns"]),
            weights_history=strategy_results["weights"],
        ).get_metrics()
    return pd.DataFrame(metrics)


def plot_strategy_comparison(results: dict[str, dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 7))
    for name, strategy_results in results.items():
        plt.plot(strategy_results["values"], label=name, linewidth=2)
    plt.title("Black-Litterman Covariance Optimization Comparison", fontsize=14)
    plt.xlabel("Trading Days", fontsize=12)
    plt.ylabel("Portfolio Value", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_covariance_experiment(
    etf_returns_path: str = DEFAULT_ETF_RETURNS_PATH,
    stock_returns_path: str = DEFAULT_STOCK_RETURNS_PATH,
    train_window: int = TRADING_DAYS_PER_YEAR * 5,
    rebalance_freq: int = MONTHLY_REBALANCE_DAYS,
    tau: float = 0.05,
    shrinkage_strength: float = 0.35,
    l2_penalty: float = 0.10,
    plot: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]], str]:
    combined_returns = load_aligned_datasets(etf_returns_path, stock_returns_path)
    backtest = RollingWindowBacktest(
        returns_df=combined_returns,
        train_window=train_window,
        rebalance_freq=rebalance_freq,
    )

    def views_wrapper(train_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return build_default_views(train_data)

    def standard_params(train_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu = train_data.mean().values * TRADING_DAYS_PER_YEAR
        sigma = sample_covariance(train_data)
        return mu, sigma

    def shrinkage_params(train_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu = train_data.mean().values * TRADING_DAYS_PER_YEAR
        sigma = shrinkage_covariance(train_data, shrinkage_strength=shrinkage_strength)
        return mu, sigma

    def l2_params(train_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu = train_data.mean().values * TRADING_DAYS_PER_YEAR
        sigma = l2_regularized_covariance(train_data, l2_penalty=l2_penalty)
        return mu, sigma

    strategy_param_builders = {
        "BL Sample Covariance": standard_params,
        "BL Shrinkage Covariance": shrinkage_params,
        "BL L2-Regularized Covariance": l2_params,
    }

    results: dict[str, dict[str, object]] = {}
    for strategy_name, params_builder in strategy_param_builders.items():
        print(f"Running {strategy_name}...")
        values, returns, weights = backtest.backtest_bl(
            get_params_func=params_builder,
            get_views_func=views_wrapper,
            tau=tau,
        )
        results[strategy_name] = {
            "values": values,
            "returns": returns,
            "weights": weights,
        }

    metrics = compute_metrics(results)
    best_strategy = metrics.loc["Sharpe Ratio"].astype(float).idxmax()

    print("=" * 90)
    print("BLACK-LITTERMAN COVARIANCE OPTIMIZATION EXPERIMENT")
    print(f"Assets in portfolio: {combined_returns.shape[1]}")
    print("Train window: 5 years (1260 trading days)")
    print("Test window: next month (21 trading days), rolling forward monthly")
    print(metrics.round(4).to_string())
    print(f"\nBest strategy by Sharpe Ratio: {best_strategy}")
    print("=" * 90)

    if plot:
        plot_strategy_comparison(results)

    return metrics, results, best_strategy


if __name__ == "__main__":
    run_covariance_experiment()
