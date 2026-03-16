import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


class Markowitz:
    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 100,
    ):
        self.mu = expected_returns
        self.sigma = cov_matrix
        self.lambda_aversion = risk_aversion
        self.n_assets = len(expected_returns)

    def optimize(self, allow_shorting: bool = False) -> np.ndarray:
        def objective(weights: np.ndarray) -> float:
            portfolio_return = -np.dot(weights, self.mu)
            portfolio_variance = np.dot(weights, np.dot(self.sigma, weights))
            return portfolio_return + self.lambda_aversion * portfolio_variance / 2

        constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
        bounds = [(None, None)] * self.n_assets if allow_shorting else [(0, 1)] * self.n_assets
        x0 = np.full(self.n_assets, 1.0 / self.n_assets)

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )
        return result.x


class BlackLitterman:
    def __init__(
        self,
        market_weights: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 100,
        tau: float = 0.05,
    ):
        self.w_market = market_weights
        self.sigma = cov_matrix
        self.lambda_aversion = risk_aversion
        self.tau = tau
        self.n_assets = len(market_weights)
        self.pi = self.lambda_aversion * np.dot(self.sigma, self.w_market)

    def update_with_views(
        self,
        P: np.ndarray,
        q: np.ndarray,
        omega: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        P = np.atleast_2d(P)
        q = np.atleast_1d(q)
        prior_cov = self.tau * self.sigma

        if omega is None:
            view_variance = np.array(
                [np.dot(P[i], np.dot(self.sigma, P[i])) for i in range(len(P))]
            )
            omega = np.diag(view_variance * self.tau)

        omega_inv = np.linalg.inv(omega)
        sigma_prior_inv = np.linalg.inv(prior_cov)
        sigma_post_inv = sigma_prior_inv + np.dot(P.T, np.dot(omega_inv, P))
        sigma_post = np.linalg.inv(sigma_post_inv)

        term1 = np.dot(sigma_prior_inv, self.pi)
        term2 = np.dot(P.T, np.dot(omega_inv, q))
        mu_post = np.dot(sigma_post, term1 + term2)
        return mu_post, sigma_post

    def optimize_with_views(
        self,
        P: np.ndarray,
        q: np.ndarray,
        omega: np.ndarray | None = None,
        allow_shorting: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu_post, sigma_post = self.update_with_views(P, q, omega)
        mvo = Markowitz(mu_post, sigma_post, self.lambda_aversion)
        weights = mvo.optimize(allow_shorting)
        return weights, mu_post, sigma_post


class RollingWindowBacktest:
    def __init__(
        self,
        returns_df: pd.DataFrame,
        train_window: int = 252 * 5,
        rebalance_freq: int = 252,
    ):
        self.returns = returns_df
        self.train_window = train_window
        self.rebalance_freq = rebalance_freq
        self.n_assets = len(returns_df.columns)

    def backtest_markowitz(self, get_params_func):
        portfolio_values = [1.0]
        weights_history = []
        portfolio_returns = []

        for t in range(self.train_window, len(self.returns), self.rebalance_freq):
            train_data = self.returns.iloc[t - self.train_window : t]
            test_start = t
            test_end = min(t + self.rebalance_freq, len(self.returns))

            mu, sigma = get_params_func(train_data)
            weights = Markowitz(mu, sigma).optimize()
            weights_history.append(weights)

            test_data = self.returns.iloc[test_start:test_end]
            for ret_vec in test_data.values:
                port_ret = np.dot(weights, ret_vec)
                portfolio_returns.append(port_ret)
                portfolio_values.append(portfolio_values[-1] * (1 + port_ret))

        return np.array(portfolio_values), np.array(portfolio_returns), weights_history

    def backtest_bl(
        self,
        get_params_func,
        get_views_func,
        market_weights: np.ndarray | None = None,
        tau: float = 0.05,
    ):
        if market_weights is None:
            market_weights = np.full(self.n_assets, 1.0 / self.n_assets)

        portfolio_values = [1.0]
        weights_history = []
        portfolio_returns = []

        for t in range(self.train_window, len(self.returns), self.rebalance_freq):
            train_data = self.returns.iloc[t - self.train_window : t]
            test_start = t
            test_end = min(t + self.rebalance_freq, len(self.returns))

            _, sigma = get_params_func(train_data)
            P, q, omega = get_views_func(train_data)

            bl = BlackLitterman(market_weights, sigma, tau=tau)
            weights, _, _ = bl.optimize_with_views(P, q, omega)
            weights_history.append(weights)

            test_data = self.returns.iloc[test_start:test_end]
            for ret_vec in test_data.values:
                port_ret = np.dot(weights, ret_vec)
                portfolio_returns.append(port_ret)
                portfolio_values.append(portfolio_values[-1] * (1 + port_ret))

        return np.array(portfolio_values), np.array(portfolio_returns), weights_history


class PerformanceMetrics:
    def __init__(
        self,
        portfolio_values: np.ndarray,
        returns: np.ndarray,
        rf_rate: float = 0.02,
        weights_history: list | None = None,
    ):
        self.values = portfolio_values
        self.returns = returns
        self.rf = rf_rate
        self.weights_history = weights_history

    def annual_return(self) -> float:
        total_return = (self.values[-1] / self.values[0]) - 1
        years = len(self.returns) / 252
        return (1 + total_return) ** (1 / years) - 1

    def annual_volatility(self) -> float:
        return np.std(self.returns) * np.sqrt(252)

    def sharpe_ratio(self) -> float:
        annual_ret = self.annual_return()
        annual_vol = self.annual_volatility()
        if annual_vol == 0:
            return 0
        return (annual_ret - self.rf) / annual_vol

    def max_drawdown(self) -> float:
        cum_max = np.maximum.accumulate(self.values)
        drawdown = (self.values - cum_max) / cum_max
        return np.min(drawdown)

    def turnover(self) -> float:
        if self.weights_history is None or len(self.weights_history) < 2:
            return 0

        turnovers = []
        for i in range(1, len(self.weights_history)):
            w_old = self.weights_history[i - 1]
            w_new = self.weights_history[i]
            turnovers.append(np.sum(np.abs(w_new - w_old)) / 2)
        return np.mean(turnovers)

    def var_95(self) -> float:
        return np.percentile(self.returns, 5)

    def get_metrics(self) -> dict:
        return {
            "Annual Return": self.annual_return(),
            "Annual Volatility": self.annual_volatility(),
            "Sharpe Ratio": self.sharpe_ratio(),
            "Max Drawdown": self.max_drawdown(),
            "Turnover": self.turnover(),
            "VaR(95%)": self.var_95(),
        }


def compare_three_strategies(
    markowitz_results: dict,
    baseline_results: dict,
    experiment_results: dict,
) -> pd.DataFrame:
    markowitz_metrics = PerformanceMetrics(
        markowitz_results["values"],
        markowitz_results["returns"],
        weights_history=markowitz_results.get("weights"),
    ).get_metrics()
    baseline_metrics = PerformanceMetrics(
        baseline_results["values"],
        baseline_results["returns"],
        weights_history=baseline_results.get("weights"),
    ).get_metrics()
    experiment_metrics = PerformanceMetrics(
        experiment_results["values"],
        experiment_results["returns"],
        weights_history=experiment_results.get("weights"),
    ).get_metrics()

    return pd.DataFrame(
        {
            "Markowitz": markowitz_metrics,
            "Standard BL": baseline_metrics,
            "Improved BL": experiment_metrics,
        }
    )


def plot_cumulative_returns(
    markowitz_results: dict,
    baseline_results: dict,
    experiment_results: dict,
    title: str = "Cumulative Returns Comparison",
):
    plt.figure(figsize=(14, 7))
    plt.plot(markowitz_results["values"], label="Markowitz", linewidth=2, alpha=0.8)
    plt.plot(baseline_results["values"], label="Standard BL", linewidth=2, alpha=0.8)
    plt.plot(experiment_results["values"], label="Improved BL", linewidth=2, alpha=0.8)
    plt.xlabel("Trading Days", fontsize=12)
    plt.ylabel("Cumulative Return (Portfolio Value)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=11, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def get_standard_params(train_returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu = train_returns.mean().values * 252
    sigma = train_returns.cov().values
    return mu, sigma


def get_standard_views(train_returns: pd.DataFrame):
    n_assets = len(train_returns.columns)
    recent_returns = train_returns.tail(20).mean() * 252
    top_indices = np.argsort(recent_returns.values)[-2:]

    P = np.zeros((2, n_assets))
    for i, idx in enumerate(top_indices):
        P[i, idx] = 1.0

    q = np.array([0.05, 0.05])
    omega = None
    return P, q, omega


def get_custom_params(
    train_returns: pd.DataFrame,
    custom_sigma: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mu = train_returns.mean().values * 252
    sigma = custom_sigma if custom_sigma is not None else train_returns.cov().values
    return mu, sigma


def get_custom_views(
    train_returns: pd.DataFrame,
    custom_P: np.ndarray | None = None,
    custom_q: np.ndarray | None = None,
    custom_omega: np.ndarray | None = None,
):
    if custom_P is not None and custom_q is not None:
        P = custom_P
        q = custom_q
    else:
        P, q, _ = get_standard_views(train_returns)

    omega = custom_omega
    return P, q, omega


def run_experiment(
    returns_data: pd.DataFrame,
    custom_sigma: np.ndarray | None = None,
    custom_P: np.ndarray | None = None,
    custom_q: np.ndarray | None = None,
    custom_omega: np.ndarray | None = None,
    tau: float = 0.05,
    train_window: int = 252 * 5,
    rebalance_freq: int = 252,
    plot: bool = True,
):
    backtest = RollingWindowBacktest(
        returns_data,
        train_window=train_window,
        rebalance_freq=rebalance_freq,
    )

    print("=" * 80)
    print("BACKTEST RUNNING...")
    print("=" * 80)

    print("-> Strategy 1: Markowitz...")
    markowitz_values, markowitz_returns, markowitz_weights = backtest.backtest_markowitz(
        get_params_func=get_standard_params
    )
    markowitz = {
        "values": markowitz_values,
        "returns": markowitz_returns,
        "weights": markowitz_weights,
    }

    print("-> Strategy 2: Standard BL...")
    baseline_values, baseline_returns, baseline_weights = backtest.backtest_bl(
        get_params_func=get_standard_params,
        get_views_func=get_standard_views,
        tau=tau,
    )
    baseline = {
        "values": baseline_values,
        "returns": baseline_returns,
        "weights": baseline_weights,
    }

    print("-> Strategy 3: Improved BL...")

    def custom_params_wrapper(train_data: pd.DataFrame):
        return get_custom_params(train_data, custom_sigma=custom_sigma)

    def custom_views_wrapper(train_data: pd.DataFrame):
        return get_custom_views(
            train_data,
            custom_P=custom_P,
            custom_q=custom_q,
            custom_omega=custom_omega,
        )

    experiment_values, experiment_returns, experiment_weights = backtest.backtest_bl(
        get_params_func=custom_params_wrapper,
        get_views_func=custom_views_wrapper,
        tau=tau,
    )
    experiment = {
        "values": experiment_values,
        "returns": experiment_returns,
        "weights": experiment_weights,
    }

    comparison = compare_three_strategies(markowitz, baseline, experiment)
    print("\nCOMPARISON RESULTS:")
    print(comparison.round(4).to_string())
    print("=" * 80)

    if plot:
        plot_cumulative_returns(markowitz, baseline, experiment)

    return comparison, {
        "markowitz": markowitz,
        "standard_bl": baseline,
        "improved_bl": experiment,
    }


def run_experiment_improve_sigma_only(
    returns_data: pd.DataFrame,
    improved_sigma: np.ndarray,
    tau: float = 0.05,
    train_window: int = 252 * 5,
    rebalance_freq: int = 252,
    plot: bool = True,
):
    print("\n" + "=" * 80)
    print("TEST: Improved Sigma ONLY")
    print("P, q, omega use standard BL settings")
    print("=" * 80 + "\n")
    return run_experiment(
        returns_data,
        custom_sigma=improved_sigma,
        tau=tau,
        train_window=train_window,
        rebalance_freq=rebalance_freq,
        plot=plot,
    )


def run_experiment_improve_omega_only(
    returns_data: pd.DataFrame,
    improved_omega: np.ndarray,
    tau: float = 0.05,
    train_window: int = 252 * 5,
    rebalance_freq: int = 252,
    plot: bool = True,
):
    print("\n" + "=" * 80)
    print("TEST: Improved Omega ONLY")
    print("Sigma, P, q use standard BL settings")
    print("=" * 80 + "\n")
    return run_experiment(
        returns_data,
        custom_omega=improved_omega,
        tau=tau,
        train_window=train_window,
        rebalance_freq=rebalance_freq,
        plot=plot,
    )
