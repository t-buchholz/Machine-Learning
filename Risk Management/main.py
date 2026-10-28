"""credit-migration-early-warning: ML early warning system for bond downgrades.

This single file holds the complete project:

    1. Configuration
    2. Synthetic data generator
    3. Risk metrics (EL, duration, VaR, ES, cost matrix)
    4. Machine learning pipeline (scikit-learn)
    5. Visualizations (matplotlib / seaborn)
    6. Command line interface and self-checks

All data is synthetic. This project is not a regulatory model.

Usage:
    python main.py --seed 42 --theme light --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402
from numpy.typing import ArrayLike  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

LOGGER = logging.getLogger("credit_ews")


# =============================================================================
# 1. CONFIGURATION AND CONSTANTS
# =============================================================================

RATINGS: list[str] = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC"]
STATES: list[str] = RATINGS + ["D"]
SECTORS: list[str] = [
    "Industrials",
    "Utilities",
    "Financials",
    "Technology",
    "Energy",
    "Consumer",
    "Healthcare",
    "Telecom",
]
NUMERIC_FEATURES: list[str] = [
    "coupon",
    "maturity_years",
    "mod_duration",
    "spread_bps",
    "debt_to_ebitda",
    "interest_coverage",
    "fcf_margin",
    "leverage_trend",
    "equity_return_3m",
    "equity_vol_3m",
    "spread_change_3m",
]
CATEGORICAL_FEATURES: list[str] = ["sector", "rating"]
TARGET: str = "downgrade_12m"

# Share of issuers per rating class in the synthetic universe.
RATING_WEIGHTS: list[float] = [0.03, 0.08, 0.20, 0.30, 0.20, 0.12, 0.05, 0.02]

# Assumed 1-year default probability per rating class (illustrative values).
PD_BY_RATING: dict[str, float] = dict(
    zip(RATINGS, [0.0001, 0.0003, 0.0007, 0.0020, 0.0090, 0.0350, 0.1400, 0.2800])
)

# Assumed spread jump in basis points after a 1-notch downgrade.
SHOCK_BPS_BY_RATING: dict[str, float] = dict(
    zip(RATINGS, [10.0, 15.0, 25.0, 40.0, 70.0, 120.0, 250.0, 400.0])
)

# Typical credit spread in basis points per rating class.
BASE_SPREAD_BPS: list[float] = [40, 55, 80, 130, 280, 480, 900, 1500]

# Assumed Loss Given Default per sector (senior unsecured).
LGD_BY_SECTOR: dict[str, float] = {
    "Industrials": 0.45,
    "Utilities": 0.35,
    "Financials": 0.55,
    "Technology": 0.50,
    "Energy": 0.50,
    "Consumer": 0.50,
    "Healthcare": 0.45,
    "Telecom": 0.45,
}

# Annual transition probabilities used to simulate rating paths.
_P_UP: list[float] = [0.00, 0.04, 0.05, 0.05, 0.05, 0.05, 0.06, 0.08]
_P_DOWN1: list[float] = [0.06, 0.07, 0.07, 0.07, 0.08, 0.10, 0.14, 0.00]
_P_DOWN2: list[float] = [0.005, 0.006, 0.007, 0.010, 0.015, 0.020, 0.00, 0.00]


@dataclass(frozen=True)
class Config:
    """Hold all settings for the data, model, risk, and plot steps.

    Attributes:
        seed: Seed for every random number generator.
        n_bonds: Number of bonds in the synthetic portfolio.
        n_issuers: Number of issuers in the synthetic portfolio.
        n_years: Length of the observation window in years (at least 2).
        target_base_rate: Target share of bonds with a downgrade.
        test_fraction: Share of issuers in the test set.
        val_fraction: Share of issuers in the validation set.
        alpha_levels: Confidence levels for VaR and Expected Shortfall.
        recall_target: Minimum recall for the threshold search.
        n_simulations: Number of Monte Carlo simulations.
        asset_correlation: Asset correlation in the Vasicek model.
        fn_loss_share: Share of EAD x LGD lost if a downgrade is missed.
        fp_review_cost: Fixed cost in EUR of one review or de-risking action.
        bid_ask_bps: Bid-ask cost in basis points of the position value.
        theme: Plot theme, either "light" or "dark".
        output_dir: Directory for all output artifacts.
        dpi: Resolution of saved figures.
    """

    seed: int = 42
    n_bonds: int = 5000
    n_issuers: int = 300
    n_years: int = 5
    target_base_rate: float = 0.07
    test_fraction: float = 0.20
    val_fraction: float = 0.20
    alpha_levels: tuple[float, ...] = (0.95, 0.99, 0.995)
    recall_target: float = 0.80
    n_simulations: int = 50_000
    asset_correlation: float = 0.20
    fn_loss_share: float = 0.15
    fp_review_cost: float = 2_000.0
    bid_ask_bps: float = 25.0
    theme: str = "light"
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    dpi: int = 300

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            ValueError: If a setting is outside its allowed range.
        """
        if self.theme not in ("light", "dark"):
            raise ValueError("theme must be 'light' or 'dark'.")
        if self.n_bonds < self.n_issuers:
            raise ValueError("n_bonds must be at least n_issuers.")
        if self.n_years < 2:
            raise ValueError("n_years must be at least 2.")
        if not 0.0 < self.target_base_rate < 0.5:
            raise ValueError("target_base_rate must be in (0, 0.5).")
        if not 0.0 < self.test_fraction + self.val_fraction < 1.0:
            raise ValueError("test_fraction + val_fraction must be in (0, 1).")
        if not 0.0 < self.recall_target <= 1.0:
            raise ValueError("recall_target must be in (0, 1].")
        if not 0.0 <= self.asset_correlation < 1.0:
            raise ValueError("asset_correlation must be in [0, 1).")
        if any(not 0.0 < a < 1.0 for a in self.alpha_levels):
            raise ValueError("Each alpha level must be in (0, 1).")
        if self.n_simulations < 1_000:
            raise ValueError("n_simulations must be at least 1,000.")

    def ensure_output_dir(self) -> Path:
        """Create the output directory if it does not exist.

        Returns:
            The output directory.
        """
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


# =============================================================================
# 2. SYNTHETIC DATA GENERATOR
# =============================================================================


def _solve_intercept(core: np.ndarray, base_rate: float) -> float:
    """Find the logistic intercept that gives a target mean probability.

    Args:
        core: Latent linear predictor without intercept.
        base_rate: Target mean downgrade probability.

    Returns:
        The intercept value.
    """
    low, high = -25.0, 10.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        mean_p = float(np.mean(1.0 / (1.0 + np.exp(-(mid + core)))))
        if mean_p < base_rate:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _price_and_duration(
    coupon: np.ndarray, yld: np.ndarray, maturity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute price and modified duration of annual-coupon bonds.

    Args:
        coupon: Annual coupon per 100 of par.
        yld: Annual yield as a decimal.
        maturity: Integer maturity in years (1 to 12).

    Returns:
        A tuple with the price (percent of par) and the modified duration.
    """
    t = np.arange(1, 13, dtype=float)[None, :]
    n = maturity[:, None]
    cash = np.where(t <= n, coupon[:, None], 0.0) + np.where(t == n, 100.0, 0.0)
    pv = cash / (1.0 + yld[:, None]) ** t
    price = pv.sum(axis=1)
    macaulay = (t * pv).sum(axis=1) / price
    return price, macaulay / (1.0 + yld)


def generate_portfolio(config: Config) -> pd.DataFrame:
    """Generate a synthetic corporate bond portfolio with a downgrade target.

    The target comes from a logistic latent model. A latent issuer stress
    variable and a quarterly systematic factor drive the downgrade
    probability. The observed features are noisy proxies of these variables.
    Each issuer enters the data in one 12-month cohort. This makes an
    issuer-level time split possible.

    Args:
        config: Project configuration.

    Returns:
        A DataFrame with one row per bond observation, sorted by date.
    """
    rng = np.random.default_rng(config.seed)
    n_iss, n = config.n_issuers, config.n_bonds

    rating_idx = rng.choice(len(RATINGS), size=n_iss, p=RATING_WEIGHTS)
    sector_idx = rng.integers(0, len(SECTORS), size=n_iss)
    issuer_stress = 0.25 * (rating_idx - 3) + rng.normal(0.0, 0.8, n_iss)
    horizon = config.n_years * 365
    entry_day = rng.integers(0, horizon - 365, size=n_iss)

    owner = np.concatenate([np.arange(n_iss), rng.integers(0, n_iss, n - n_iss)])
    obs_day = entry_day[owner] + rng.integers(0, 365, size=n)
    stress = issuer_stress[owner] + rng.normal(0.0, 0.3, n)
    ridx = rating_idx[owner]

    # Systematic factor per quarter: AR(1) process. High value = credit stress.
    n_q = horizon // 91 + 2
    z_q = np.empty(n_q)
    z_q[0] = rng.normal()
    for t in range(1, n_q):
        z_q[t] = 0.7 * z_q[t - 1] + np.sqrt(1.0 - 0.7**2) * rng.normal()
    z = z_q[obs_day // 91]

    # Market data.
    base_spread = np.asarray(BASE_SPREAD_BPS, dtype=float)[ridx]
    spread = base_spread * np.exp(0.12 * stress + 0.10 * z + rng.normal(0, 0.12, n))
    maturity = rng.integers(1, 13, size=n).astype(float)
    coupon = np.clip(2.0 + spread / 100.0 + rng.normal(0.0, 0.3, n), 0.5, None)
    price, mod_duration = _price_and_duration(
        coupon, 0.025 + spread / 1e4, maturity
    )
    notional = np.clip(np.round(rng.lognormal(np.log(2e6), 0.6, n), -5), 5e5, 2e7)

    # Fundamental and market signals: noisy proxies of the latent stress.
    debt_to_ebitda = np.clip(
        np.exp(0.9 + 0.30 * stress + rng.normal(0, 0.30, n)), 0.3, 15.0
    )
    interest_coverage = np.clip(
        np.exp(1.9 - 0.35 * stress + rng.normal(0, 0.35, n)), 0.3, 40.0
    )
    fcf_margin = 0.07 - 0.03 * stress + rng.normal(0, 0.03, n)
    leverage_trend = 0.30 * stress + rng.normal(0, 0.60, n)
    equity_return = -0.04 * z - 0.03 * stress + rng.normal(0, 0.07, n)
    equity_vol = np.clip(
        0.22 + 0.04 * stress + 0.03 * z + rng.normal(0, 0.04, n), 0.05, 1.0
    )
    spread_change = 10.0 * stress + 15.0 * z + rng.normal(0, 15.0, n)

    # Target: logistic latent model with unobservable noise.
    core = 2.1 * stress + 0.4 * z + rng.normal(0.0, 0.3, n)
    intercept = _solve_intercept(core, config.target_base_rate)
    prob = 1.0 / (1.0 + np.exp(-(intercept + core)))
    target = (rng.random(n) < prob).astype(int)

    start = pd.Timestamp("2019-01-01")
    df = pd.DataFrame(
        {
            "issuer_id": np.array([f"ISS{i:04d}" for i in range(n_iss)])[owner],
            "observation_date": start + pd.to_timedelta(obs_day, unit="D"),
            "sector": np.asarray(SECTORS)[sector_idx[owner]],
            "rating": np.asarray(RATINGS)[ridx],
            "notional": notional,
            "price": np.round(price, 2),
            "coupon": np.round(coupon, 3),
            "maturity_years": maturity,
            "mod_duration": np.round(mod_duration, 3),
            "spread_bps": np.round(spread, 1),
            "debt_to_ebitda": debt_to_ebitda,
            "interest_coverage": interest_coverage,
            "fcf_margin": fcf_margin,
            "leverage_trend": leverage_trend,
            "equity_return_3m": equity_return,
            "equity_vol_3m": equity_vol,
            "spread_change_3m": spread_change,
            TARGET: target,
        }
    )
    for col in ["debt_to_ebitda", "interest_coverage", "fcf_margin", "equity_vol_3m"]:
        rate = rng.uniform(0.03, 0.05)
        df.loc[rng.random(n) < rate, col] = np.nan

    df = df.sort_values("observation_date", kind="stable").reset_index(drop=True)
    df.insert(0, "bond_id", [f"BND{i:05d}" for i in range(n)])
    return df


def _base_transition_matrix() -> np.ndarray:
    """Build the annual transition matrix used to simulate rating paths.

    Returns:
        A 9 x 9 row-stochastic matrix. The last state is Default.
    """
    p = np.zeros((9, 9))
    for i, rating in enumerate(RATINGS):
        if i > 0:
            p[i, i - 1] += _P_UP[i]
        if i + 1 <= 7:
            p[i, i + 1] += _P_DOWN1[i]
        if i + 2 <= 7:
            p[i, i + 2] += _P_DOWN2[i]
        p[i, 8] = PD_BY_RATING[rating]
        p[i, i] = 1.0 - (p[i].sum() - p[i, i])
    p[8, 8] = 1.0
    return p


def build_transition_matrix(config: Config) -> pd.DataFrame:
    """Estimate an empirical rating transition matrix from simulated paths.

    The function simulates yearly Markov rating paths for many issuers. It
    counts the observed transitions and normalizes each row.

    Args:
        config: Project configuration.

    Returns:
        A 9 x 9 DataFrame of transition frequencies. Rows are the start
        state. Columns are the end state. The last state is Default.
    """
    rng = np.random.default_rng(config.seed + 1)
    cum = np.cumsum(_base_transition_matrix(), axis=1)
    cum[:, -1] = 1.0
    n_paths = 20_000
    state = rng.choice(len(RATINGS), size=n_paths, p=RATING_WEIGHTS)
    counts = np.zeros((9, 9))
    for _ in range(config.n_years):
        u = rng.random(n_paths)
        nxt = np.minimum((u[:, None] > cum[state]).sum(axis=1), 8)
        np.add.at(counts, (state, nxt), 1.0)
        state = nxt
    sums = counts.sum(axis=1, keepdims=True)
    matrix = np.divide(counts, sums, out=np.zeros_like(counts), where=sums > 0)
    for i in np.flatnonzero(sums.ravel() == 0):
        matrix[i, i] = 1.0
    return pd.DataFrame(matrix, index=STATES, columns=STATES)


# =============================================================================
# 3. RISK METRICS
# =============================================================================


def rating_to_pd(ratings: pd.Series) -> pd.Series:
    """Map ratings to 1-year Probability of Default.

    This is a default probability from a fixed lookup. It is not the
    model's downgrade probability.

    Args:
        ratings: Series of rating labels.

    Returns:
        Series of default probabilities.

    Raises:
        ValueError: If a rating is not in the lookup.
    """
    out = ratings.map(PD_BY_RATING).astype(float)
    if out.isna().any():
        raise ValueError("Unknown rating label found.")
    return out


def lgd_by_sector(sectors: pd.Series) -> pd.Series:
    """Map sectors to an assumed Loss Given Default.

    Args:
        sectors: Series of sector labels.

    Returns:
        Series of LGD values between 0 and 1.

    Raises:
        ValueError: If a sector is not in the lookup.
    """
    out = sectors.map(LGD_BY_SECTOR).astype(float)
    if out.isna().any():
        raise ValueError("Unknown sector label found.")
    return out


def exposure_at_default(df: pd.DataFrame, accrued_fraction: float = 0.5) -> pd.Series:
    """Compute Exposure at Default.

    EAD equals the market value plus accrued interest. The accrued interest
    equals the annual coupon amount times ``accrued_fraction``.

    Args:
        df: Portfolio with ``notional``, ``price``, and ``coupon``.
        accrued_fraction: Fraction of one annual coupon that is accrued.

    Returns:
        Series of EAD values in EUR.
    """
    market_value = df["notional"] * df["price"] / 100.0
    accrued = df["notional"] * df["coupon"] / 100.0 * accrued_fraction
    return market_value + accrued


def expected_loss(ead: ArrayLike, pd_: ArrayLike, lgd: ArrayLike) -> np.ndarray:
    """Compute Expected Loss: EL = EAD x PD x LGD.

    Args:
        ead: Exposure at Default.
        pd_: Probability of Default.
        lgd: Loss Given Default.

    Returns:
        Array of expected losses.
    """
    return (
        np.asarray(ead, dtype=float)
        * np.asarray(pd_, dtype=float)
        * np.asarray(lgd, dtype=float)
    )


def duration_price_change(
    price: ArrayLike,
    mod_duration: ArrayLike,
    dy: ArrayLike,
    convexity: ArrayLike | None = None,
) -> np.ndarray:
    """Approximate the price change for a yield change.

    The formula is dP = -D_mod x P x dy, with an optional convexity term
    0.5 x C x P x dy^2.

    Args:
        price: Price (any unit). The result has the same unit.
        mod_duration: Modified duration in years.
        dy: Yield or spread change as a decimal (0.01 = 100 bp).
        convexity: Optional convexity.

    Returns:
        Array of price changes. A positive ``dy`` gives a negative change.
    """
    price_a = np.asarray(price, dtype=float)
    dy_a = np.asarray(dy, dtype=float)
    change = -np.asarray(mod_duration, dtype=float) * price_a * dy_a
    if convexity is not None:
        change = change + 0.5 * np.asarray(convexity, dtype=float) * price_a * dy_a**2
    return change


def downgrade_spread_shock(ratings: pd.Series) -> pd.Series:
    """Return the assumed spread jump after a downgrade.

    Args:
        ratings: Series of current ratings.

    Returns:
        Series of spread shocks in basis points.
    """
    return ratings.map(SHOCK_BPS_BY_RATING).astype(float)


def migration_loss_given_downgrade(df: pd.DataFrame) -> np.ndarray:
    """Compute the position loss if a bond is downgraded.

    The loss uses the duration approximation and the rating-dependent
    spread shock.

    Args:
        df: Portfolio with ``rating``, ``price``, ``mod_duration``, and
            ``notional``.

    Returns:
        Array of positive losses in EUR.
    """
    shock = downgrade_spread_shock(df["rating"]).to_numpy() / 1e4
    d_price = duration_price_change(df["price"], df["mod_duration"], shock)
    return -(df["notional"].to_numpy() / 100.0) * d_price


def simulate_portfolio_losses(
    df: pd.DataFrame, pd_downgrade: ArrayLike, config: Config
) -> np.ndarray:
    """Simulate 1-year portfolio losses with a Vasicek one-factor model.

    Assumptions:
        * Asset return X = sqrt(rho) Z + sqrt(1 - rho) e. Z and e are
          standard normal.
        * A bond defaults if X is below the default threshold. A bond is
          downgraded (without default) if X is between the default and the
          downgrade threshold.
        * The thresholds come from the unconditional probabilities.
        * Default loss is EAD x LGD. Downgrade loss is the duration-based
          spread shock loss.
        * Recovery rates and rating paths are fixed. Horizon is 1 year.

    Args:
        df: Portfolio with the columns needed for EAD, LGD, and migration loss.
        pd_downgrade: Calibrated downgrade probability for each bond.
        config: Project configuration.

    Returns:
        Array of simulated portfolio losses with ``n_simulations`` entries.
    """
    rng = np.random.default_rng(config.seed + 2)
    n = len(df)
    pd_def = rating_to_pd(df["rating"]).to_numpy()
    p_down = np.clip(np.asarray(pd_downgrade, dtype=float), 1e-6, 1.0 - 1e-6)
    p_down = np.maximum(p_down, pd_def)
    thr_def = norm.ppf(pd_def)
    thr_down = norm.ppf(p_down)
    loss_default = (exposure_at_default(df) * lgd_by_sector(df["sector"])).to_numpy()
    loss_migration = migration_loss_given_downgrade(df)

    rho = config.asset_correlation
    losses = np.empty(config.n_simulations)
    chunk = max(1, int(2_000_000 // n))
    done = 0
    while done < config.n_simulations:
        m = min(chunk, config.n_simulations - done)
        z = rng.standard_normal((m, 1))
        eps = rng.standard_normal((m, n))
        x = np.sqrt(rho) * z + np.sqrt(1.0 - rho) * eps
        defaulted = x < thr_def
        downgraded = (x < thr_down) & ~defaulted
        losses[done : done + m] = defaulted @ loss_default + downgraded @ loss_migration
        done += m
    return losses


def value_at_risk(losses: ArrayLike, alpha: float) -> float:
    """Compute Value at Risk as the alpha-quantile of the loss distribution.

    Args:
        losses: Simulated portfolio losses (positive = loss).
        alpha: Confidence level, for example 0.99.

    Returns:
        The VaR value.
    """
    return float(np.quantile(np.asarray(losses, dtype=float), alpha))


def expected_shortfall(losses: ArrayLike, alpha: float) -> float:
    """Compute Expected Shortfall: the mean loss at or above the VaR.

    Args:
        losses: Simulated portfolio losses (positive = loss).
        alpha: Confidence level, for example 0.99.

    Returns:
        The ES value.
    """
    arr = np.asarray(losses, dtype=float)
    var = np.quantile(arr, alpha)
    return float(arr[arr >= var].mean())


def var_es_table(
    losses: ArrayLike, alphas: tuple[float, ...]
) -> dict[float, dict[str, float]]:
    """Compute VaR and ES for several confidence levels.

    Args:
        losses: Simulated portfolio losses.
        alphas: Confidence levels.

    Returns:
        A dictionary from alpha to a dictionary with keys ``VaR`` and ``ES``.
    """
    return {
        a: {"VaR": value_at_risk(losses, a), "ES": expected_shortfall(losses, a)}
        for a in alphas
    }


def cost_matrix(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Build asymmetric misclassification costs for each bond.

    False negative cost = EAD x LGD x ``fn_loss_share``. This is a missed
    downgrade. False positive cost = review cost + bid-ask cost on the
    market value. This is an unnecessary de-risking action.

    Args:
        df: Portfolio with the columns needed for EAD and LGD.
        config: Project configuration.

    Returns:
        A DataFrame with the columns ``cost_fn`` and ``cost_fp``.
    """
    ead = exposure_at_default(df)
    lgd = lgd_by_sector(df["sector"])
    market_value = df["notional"] * df["price"] / 100.0
    return pd.DataFrame(
        {
            "cost_fn": ead * lgd * config.fn_loss_share,
            "cost_fp": config.fp_review_cost
            + market_value * config.bid_ask_bps / 1e4,
        },
        index=df.index,
    )


def cost_weighted_confusion(
    y_true: ArrayLike, y_pred: ArrayLike, costs: pd.DataFrame
) -> np.ndarray:
    """Build a confusion matrix in EUR instead of counts.

    Correct decisions cost zero. A false positive costs ``cost_fp``. A
    false negative costs ``cost_fn``.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        costs: Output of :func:`cost_matrix`.

    Returns:
        A 2 x 2 array in the order [[TN, FP], [FN, TP]].
    """
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    fp = float(costs["cost_fp"].to_numpy()[(yt == 0) & (yp == 1)].sum())
    fn = float(costs["cost_fn"].to_numpy()[(yt == 1) & (yp == 0)].sum())
    return np.array([[0.0, fp], [fn, 0.0]])


def expected_cost(y_true: ArrayLike, y_pred: ArrayLike, costs: pd.DataFrame) -> float:
    """Compute the total misclassification cost.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        costs: Output of :func:`cost_matrix`.

    Returns:
        Total cost in EUR.
    """
    return float(cost_weighted_confusion(y_true, y_pred, costs).sum())


def rating_summary(df: pd.DataFrame, pd_downgrade: ArrayLike) -> pd.DataFrame:
    """Summarize exposure and expected loss per rating class.

    Args:
        df: Portfolio.
        pd_downgrade: Calibrated downgrade probability for each bond.

    Returns:
        A DataFrame with one row per rating class and the columns
        ``n_bonds``, ``exposure``, ``mean_downgrade_prob``, ``el_default``,
        ``el_migration``, ``el_total``, and ``el_to_exposure_bps``.
    """
    ead = exposure_at_default(df)
    p = np.asarray(pd_downgrade, dtype=float)
    tmp = pd.DataFrame(
        {
            "rating": df["rating"].to_numpy(),
            "exposure": ead.to_numpy(),
            "p_down": p,
            "el_default": expected_loss(
                ead, rating_to_pd(df["rating"]), lgd_by_sector(df["sector"])
            ),
            "el_migration": p * migration_loss_given_downgrade(df),
        }
    )
    grp = tmp.groupby("rating").agg(
        n_bonds=("exposure", "size"),
        exposure=("exposure", "sum"),
        mean_downgrade_prob=("p_down", "mean"),
        el_default=("el_default", "sum"),
        el_migration=("el_migration", "sum"),
    )
    grp = grp.reindex(RATINGS).fillna(0.0)
    grp["n_bonds"] = grp["n_bonds"].astype(int)
    grp["el_total"] = grp["el_default"] + grp["el_migration"]
    grp["el_to_exposure_bps"] = np.where(
        grp["exposure"] > 0, grp["el_total"] / grp["exposure"].replace(0, np.nan) * 1e4, 0.0
    )
    grp.index.name = "rating"
    return grp.reset_index()


def build_risk_report(
    df: pd.DataFrame, pd_downgrade: ArrayLike, config: Config
) -> tuple[dict[str, Any], np.ndarray]:
    """Build the portfolio risk report.

    Args:
        df: Portfolio.
        pd_downgrade: Calibrated downgrade probability for each bond.
        config: Project configuration.

    Returns:
        A tuple with the report dictionary and the simulated losses.
    """
    p = np.asarray(pd_downgrade, dtype=float)
    ead = exposure_at_default(df)
    el_default = float(
        expected_loss(
            ead, rating_to_pd(df["rating"]), lgd_by_sector(df["sector"])
        ).sum()
    )
    el_migration = float((p * migration_loss_given_downgrade(df)).sum())
    losses = simulate_portfolio_losses(df, p, config)
    table = var_es_table(losses, config.alpha_levels)
    total_ead = float(ead.sum())
    report: dict[str, Any] = {
        "n_bonds": int(len(df)),
        "total_ead_eur": total_ead,
        "expected_downgrades": float(p.sum()),
        "el_default_eur": el_default,
        "el_migration_eur": el_migration,
        "el_total_eur": el_default + el_migration,
        "simulation": {
            "n_simulations": config.n_simulations,
            "asset_correlation": config.asset_correlation,
            "horizon_years": 1,
            "mean_loss_eur": float(losses.mean()),
        },
        "var_es": {
            f"{a * 100:g}%": {
                "VaR_eur": v["VaR"],
                "ES_eur": v["ES"],
                "VaR_pct_of_ead": v["VaR"] / total_ead * 100.0,
                "ES_pct_of_ead": v["ES"] / total_ead * 100.0,
            }
            for a, v in table.items()
        },
    }
    return report, losses


# =============================================================================
# 4. MACHINE LEARNING PIPELINE
# =============================================================================


@dataclass
class PipelineResults:
    """Hold the outputs of the training and evaluation step.

    Attributes:
        model: Fitted scikit-learn pipeline (raw scores).
        calibrator: Isotonic calibrator fitted on validation data.
        baseline_model: Fitted logistic regression baseline.
        threshold_recall: Threshold with max precision at the recall target.
        threshold_cost: Threshold with the lowest expected cost.
        metrics: Nested dictionary of metrics.
        predictions: Test-set predictions.
        splits: Train, validation, and test DataFrames.
        importance: Permutation importance on the test set.
        best_params: Selected hyperparameters.
    """

    model: Pipeline
    calibrator: IsotonicRegression
    baseline_model: Pipeline
    threshold_recall: float
    threshold_cost: float
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    importance: pd.DataFrame
    best_params: dict[str, Any] = field(default_factory=dict)


def time_grouped_split(df: pd.DataFrame, config: Config) -> dict[str, pd.DataFrame]:
    """Split data by issuer in time order.

    The function sorts issuers by their first observation date. The earliest
    issuers go to train, the next to validation, and the latest to test.
    One issuer appears in one split only. This prevents issuer leakage.

    Args:
        df: Portfolio with ``issuer_id`` and ``observation_date``.
        config: Project configuration.

    Returns:
        A dictionary with the keys ``train``, ``val``, and ``test``.

    Raises:
        ValueError: If a split is empty or has no positive case.
    """
    entry = df.groupby("issuer_id")["observation_date"].min()
    issuers = entry.sort_values(kind="stable").index.to_numpy()
    n = len(issuers)
    n_test = max(1, int(round(config.test_fraction * n)))
    n_val = max(1, int(round(config.val_fraction * n)))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough issuers for the requested split.")
    groups = {
        "train": issuers[:n_train],
        "val": issuers[n_train : n_train + n_val],
        "test": issuers[n_train + n_val :],
    }
    splits = {
        name: df[df["issuer_id"].isin(ids)].reset_index(drop=True)
        for name, ids in groups.items()
    }
    for name, part in splits.items():
        if part.empty or part[TARGET].nunique() < 2:
            raise ValueError(f"Split '{name}' needs both classes.")
    return splits


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing step.

    Numeric features use median imputation and standard scaling.
    Categorical features use one-hot encoding.

    Returns:
        A ``ColumnTransformer``.
    """
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_model(kind: str, seed: int, **params: Any) -> Pipeline:
    """Build a full model pipeline.

    Both classifiers use ``class_weight="balanced"`` to handle the class
    imbalance.

    Args:
        kind: Either ``"hgb"`` (gradient boosting) or ``"logreg"``.
        seed: Random seed.
        **params: Extra classifier parameters for ``"hgb"``.

    Returns:
        A scikit-learn ``Pipeline``.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind == "hgb":
        clf: Any = HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=seed,
            max_iter=200,
            min_samples_leaf=40,
            l2_regularization=1.0,
            **params,
        )
    elif kind == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def find_recall_threshold(
    y_true: ArrayLike, proba: ArrayLike, recall_target: float
) -> float:
    """Find the threshold with maximum precision at a minimum recall.

    Args:
        y_true: True labels.
        proba: Predicted downgrade probabilities.
        recall_target: Minimum recall, for example 0.80.

    Returns:
        The decision threshold. Predict a downgrade if proba >= threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = np.flatnonzero(recall[:-1] >= recall_target)
    if ok.size == 0:
        return float(thresholds.min())
    best = ok[np.argmax(precision[:-1][ok])]
    return float(thresholds[best])


def find_cost_threshold(
    y_true: ArrayLike, proba: ArrayLike, costs: pd.DataFrame
) -> float:
    """Find the threshold with the lowest total misclassification cost.

    Args:
        y_true: True labels.
        proba: Predicted downgrade probabilities.
        costs: Output of :func:`cost_matrix`.

    Returns:
        The decision threshold. Predict a downgrade if proba >= threshold.
    """
    yt = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    candidates = np.unique(np.concatenate([[0.0], p]))
    best_t, best_cost = float(candidates[0]), np.inf
    for t in candidates:
        cost = expected_cost(yt, (p >= t).astype(int), costs)
        if cost < best_cost:
            best_t, best_cost = float(t), cost
    return best_t


def _operating_point(
    y_true: np.ndarray, proba: np.ndarray, threshold: float, costs: pd.DataFrame
) -> dict[str, float]:
    """Compute metrics for one decision threshold.

    Args:
        y_true: True labels.
        proba: Predicted probabilities.
        threshold: Decision threshold.
        costs: Output of :func:`cost_matrix`.

    Returns:
        A dictionary with precision, recall, counts, and cost.
    """
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {
        "threshold": float(threshold),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "alert_rate": float(pred.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "expected_cost_eur": expected_cost(y_true, pred, costs),
    }


def train_and_evaluate(df: pd.DataFrame, config: Config) -> PipelineResults:
    """Train, calibrate, tune thresholds, and evaluate the model.

    Steps:
        1. Split the data by issuer in time order.
        2. Tune gradient boosting hyperparameters on validation PR-AUC.
        3. Calibrate probabilities with isotonic regression (validation).
        4. Choose two thresholds on validation data only.
        5. Evaluate on the test set once.

    The calibrator and the thresholds use the same validation set. This can
    make validation metrics slightly optimistic. Test metrics are clean.

    Args:
        df: Portfolio from :func:`generate_portfolio`.
        config: Project configuration.

    Returns:
        A :class:`PipelineResults` object.
    """
    splits = time_grouped_split(df, config)
    feats = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    train, val, test = splits["train"], splits["val"], splits["test"]
    x_tr, y_tr = train[feats], train[TARGET].to_numpy()
    x_val, y_val = val[feats], val[TARGET].to_numpy()
    x_te, y_te = test[feats], test[TARGET].to_numpy()

    # Hyperparameter search on train / validation.
    best_score, best_params, model = -np.inf, {}, None
    for lr in (0.03, 0.08):
        for depth in (2, 3, 4):
            candidate = build_model("hgb", config.seed, learning_rate=lr, max_depth=depth)
            candidate.fit(x_tr, y_tr)
            score = average_precision_score(y_val, candidate.predict_proba(x_val)[:, 1])
            LOGGER.info("Grid lr=%.2f depth=%d val PR-AUC=%.4f", lr, depth, score)
            if score > best_score:
                best_score = score
                best_params = {"learning_rate": lr, "max_depth": depth}
                model = candidate
    assert model is not None
    baseline = build_model("logreg", config.seed).fit(x_tr, y_tr)

    # Calibration and thresholds on validation data.
    val_raw = model.predict_proba(x_val)[:, 1]
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(val_raw, y_val)
    val_cal = calibrator.predict(val_raw)
    val_costs = cost_matrix(val, config)
    thr_recall = find_recall_threshold(y_val, val_cal, config.recall_target)
    thr_cost = find_cost_threshold(y_val, val_cal, val_costs)

    # Single evaluation on test data.
    te_raw = model.predict_proba(x_te)[:, 1]
    te_cal = calibrator.predict(te_raw)
    te_costs = cost_matrix(test, config)
    base_te = baseline.predict_proba(x_te)[:, 1]

    metrics: dict[str, Any] = {
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "n_issuers": {k: int(v["issuer_id"].nunique()) for k, v in splits.items()},
        "base_rate": {k: float(v[TARGET].mean()) for k, v in splits.items()},
        "best_params": best_params,
        "thresholds": {
            "recall_target_threshold": thr_recall,
            "cost_minimizing_threshold": thr_cost,
        },
        "validation": {
            "roc_auc": float(roc_auc_score(y_val, val_cal)),
            "pr_auc": float(average_precision_score(y_val, val_cal)),
            "brier_raw": float(brier_score_loss(y_val, val_raw)),
            "brier_calibrated": float(brier_score_loss(y_val, val_cal)),
            "recall_threshold": _operating_point(y_val, val_cal, thr_recall, val_costs),
            "cost_threshold": _operating_point(y_val, val_cal, thr_cost, val_costs),
        },
        "test": {
            "roc_auc": float(roc_auc_score(y_te, te_cal)),
            "pr_auc": float(average_precision_score(y_te, te_cal)),
            "brier_raw": float(brier_score_loss(y_te, te_raw)),
            "brier_calibrated": float(brier_score_loss(y_te, te_cal)),
            "recall_threshold": _operating_point(y_te, te_cal, thr_recall, te_costs),
            "cost_threshold": _operating_point(y_te, te_cal, thr_cost, te_costs),
            "baseline_logreg": {
                "roc_auc": float(roc_auc_score(y_te, base_te)),
                "pr_auc": float(average_precision_score(y_te, base_te)),
            },
        },
    }

    perm = permutation_importance(
        model,
        x_te,
        y_te,
        scoring="average_precision",
        n_repeats=10,
        random_state=config.seed,
    )
    importance = (
        pd.DataFrame(
            {
                "feature": feats,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    predictions = pd.DataFrame(
        {
            "bond_id": test["bond_id"],
            "issuer_id": test["issuer_id"],
            "observation_date": test["observation_date"],
            "rating": test["rating"],
            "y_true": y_te,
            "proba_raw": te_raw,
            "proba_calibrated": te_cal,
            "pred_recall_threshold": (te_cal >= thr_recall).astype(int),
            "pred_cost_threshold": (te_cal >= thr_cost).astype(int),
        }
    )
    return PipelineResults(
        model=model,
        calibrator=calibrator,
        baseline_model=baseline,
        threshold_recall=thr_recall,
        threshold_cost=thr_cost,
        metrics=metrics,
        predictions=predictions,
        splits=splits,
        importance=importance,
        best_params=best_params,
    )


# =============================================================================
# 5. VISUALIZATIONS
# =============================================================================


def apply_theme(theme: str) -> dict[str, str]:
    """Apply the light or dark matplotlib theme.

    Args:
        theme: Either ``"light"`` or ``"dark"``.

    Returns:
        The color palette as a dictionary.

    Raises:
        ValueError: If ``theme`` is unknown.
    """
    palettes = {
        "light": {
            "bg": "#FFFFFF", "fg": "#1F2933", "grid": "#D9DEE3",
            "primary": "#1F4E79", "accent": "#C0392B", "muted": "#7B8794",
            "good": "#2E7D5B", "warn": "#D98E04",
        },
        "dark": {
            "bg": "#0F1720", "fg": "#E6EDF3", "grid": "#2D3946",
            "primary": "#5AA9E6", "accent": "#FF6B6B", "muted": "#9AA5B1",
            "good": "#4CC38A", "warn": "#F2B84B",
        },
    }
    if theme not in palettes:
        raise ValueError("theme must be 'light' or 'dark'.")
    pal = palettes[theme]
    plt.rcParams.update(
        {
            "figure.facecolor": pal["bg"],
            "axes.facecolor": pal["bg"],
            "savefig.facecolor": pal["bg"],
            "axes.edgecolor": pal["fg"],
            "axes.labelcolor": pal["fg"],
            "text.color": pal["fg"],
            "xtick.color": pal["fg"],
            "ytick.color": pal["fg"],
            "grid.color": pal["grid"],
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )
    return pal


def _save(fig: plt.Figure, path: Path, config: Config) -> Path:
    """Save a figure and close it.

    Args:
        fig: Figure to save.
        path: Output file path.
        config: Project configuration.

    Returns:
        The output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


FEATURE_LABELS: dict[str, str] = {
    "coupon": "Coupon",
    "maturity_years": "Maturity",
    "mod_duration": "Modified duration",
    "spread_bps": "Credit spread",
    "debt_to_ebitda": "Debt / EBITDA",
    "interest_coverage": "Interest coverage",
    "fcf_margin": "Free cash flow margin",
    "leverage_trend": "Leverage trend",
    "equity_return_3m": "Equity return (3m)",
    "equity_vol_3m": "Equity volatility (3m)",
    "spread_change_3m": "Spread change (3m)",
    "sector": "Sector",
    "rating": "Current rating",
}


def _headline(fig: plt.Figure, text: str) -> None:
    """Add a left-aligned headline with the key finding.

    Args:
        fig: Figure to change.
        text: Headline text. Use a line break for two lines.
    """
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, fontweight="bold")


def _footer(fig: plt.Figure, text: str, pal: dict[str, str]) -> None:
    """Add a small explanatory note below the figure.

    Args:
        fig: Figure to change.
        text: Note text.
        pal: Color palette from :func:`apply_theme`.
    """
    fig.text(0.01, -0.03, text, ha="left", va="top", fontsize=8, color=pal["muted"])


def plot_model_performance(
    y_true: ArrayLike,
    proba_raw: ArrayLike,
    proba: ArrayLike,
    thresholds: dict[str, float],
    costs: pd.DataFrame,
    config: Config,
    path: Path,
) -> Path:
    """Plot detection quality, probability quality, and decisions.

    Panel A shows the precision-recall curve with both operating points.
    Panel B shows the calibration curve. Panel C shows the decisions at the
    first threshold in ``thresholds``, with counts and costs.

    Args:
        y_true: True labels (test set).
        proba_raw: Uncalibrated probabilities.
        proba: Calibrated probabilities.
        thresholds: Names and values of two thresholds. The first one is
            used for panel C and for the headline.
        costs: Output of :func:`cost_matrix` for the test set.
        config: Project configuration.
        path: Output file path.

    Returns:
        The output path.
    """
    pal = apply_theme(config.theme)
    yt = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    p_raw = np.asarray(proba_raw, dtype=float)
    names = list(thresholds)
    base = float(yt.mean())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), layout="constrained")

    # Panel A: precision-recall curve with operating points.
    ax = axes[0]
    precision, recall, _ = precision_recall_curve(yt, p)
    ax.plot(recall, precision, color=pal["primary"], lw=2.2)
    ax.axhline(base, color=pal["muted"], ls=":")
    ax.text(0.02, base + 0.015, f"Random guessing: {base:.0%}", ha="left",
            va="bottom", fontsize=8, color=pal["muted"])
    stats: dict[str, tuple[float, float, float]] = {}
    for name, thr, color, marker, y_box in zip(
        names, thresholds.values(), [pal["accent"], pal["warn"]], ["o", "s"], [0.66, 0.42]
    ):
        pred = p >= thr
        tp = int((pred & (yt == 1)).sum())
        fp = int((pred & (yt == 0)).sum())
        rec = tp / max(int(yt.sum()), 1)
        prec = tp / max(tp + fp, 1)
        stats[name] = (float(pred.mean()), rec, prec)
        ax.scatter(rec, prec, s=90, color=color, marker=marker, zorder=5,
                   edgecolor=pal["bg"])
        ax.annotate(
            f"{name}\nalerts on {pred.mean():.0%} of bonds\n"
            f"catches {rec:.0%} of downgrades\n{prec:.0%} of alerts are right",
            xy=(rec, prec), xytext=(0.40, y_box), textcoords="axes fraction",
            fontsize=8, va="bottom", color=pal["fg"],
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.2},
            bbox={"boxstyle": "round,pad=0.35", "fc": pal["bg"], "ec": color, "lw": 1.2},
        )
    ax.set(xlim=(0, 1.02), ylim=(0, 1.05),
           xlabel="Recall: share of real downgrades caught",
           ylabel="Precision: share of alerts that are right",
           title=f"A. Detection quality (ROC-AUC {roc_auc_score(yt, p):.2f})")

    # Panel B: calibration.
    ax = axes[1]
    top = 0.0
    ax.plot([0, 1], [0, 1], color=pal["muted"], ls="--", label="Perfect calibration")
    for label, prob, color in [
        ("Before calibration (class-weighted)", p_raw, pal["accent"]),
        ("After calibration", p, pal["primary"]),
    ]:
        frac, mean_pred = calibration_curve(yt, prob, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", lw=2, color=color, label=label)
        top = max(top, float(mean_pred.max()), float(frac.max()))
    lim = min(1.0, top * 1.1)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Predicted downgrade probability",
           ylabel="Observed downgrade rate",
           title="B. Can the probabilities be trusted?")
    ax.legend(loc="upper left", fontsize=8)

    # Panel C: decisions at the first threshold.
    ax = axes[2]
    pred0 = (p >= thresholds[names[0]]).astype(int)
    cm = confusion_matrix(yt, pred0, labels=[0, 1])
    rates = cm / cm.sum(axis=1, keepdims=True)
    cost_cm = cost_weighted_confusion(yt, pred0, costs)
    cell = [["Correct: no alert", "False alarm"], ["Missed downgrade", "Caught downgrade"]]
    rows = ["non-downgrades", "downgrades"]
    annot = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            extra = f"\nEUR {cost_cm[i, j] / 1e3:,.0f}k cost" if cost_cm[i, j] > 0 else ""
            annot[i, j] = (
                f"{cell[i][j]}\n{cm[i, j]:,} bonds\n{rates[i, j]:.0%} of {rows[i]}{extra}"
            )
    cmap = LinearSegmentedColormap.from_list("perf", [pal["bg"], pal["primary"]])
    sns.heatmap(rates, annot=annot, fmt="", cmap=cmap, vmin=0, vmax=1, cbar=False,
                linewidths=2, linecolor=pal["bg"], ax=ax, annot_kws={"fontsize": 9},
                xticklabels=["No alert", "Alert"],
                yticklabels=["Really not downgraded", "Really downgraded"])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center")
    ax.grid(False)
    ax.set_title(f"C. Decisions at '{names[0]}'")

    alert, rec, _ = stats[names[0]]
    cost_alert = float(cost_cm.sum())
    cost_none = float(costs["cost_fn"].to_numpy()[yt == 1].sum())
    _headline(
        fig,
        f"Threshold '{names[0]}' (set on validation data): on the unseen test set the "
        f"model alerts on {alert:.0%} of bonds and catches {rec:.0%} of downgrades.\n"
        f"Error cost: EUR {cost_alert / 1e6:.1f}m, versus EUR {cost_none / 1e6:.1f}m "
        "with no early warning.",
    )
    _footer(
        fig,
        f"Synthetic data | test set: {len(yt):,} bonds, {int(yt.sum())} downgrades | "
        "thresholds were chosen on validation data. Cost = missed downgrades "
        "(EAD x LGD x loss share) + false alarms (review + bid-ask cost).",
        pal,
    )
    return _save(fig, path, config)


def plot_feature_importance(
    importance: pd.DataFrame, config: Config, path: Path
) -> Path:
    """Plot permutation feature importance with plain-language labels.

    Args:
        importance: DataFrame with ``feature``, ``importance_mean``, and
            ``importance_std``.
        config: Project configuration.
        path: Output file path.

    Returns:
        The output path.
    """
    pal = apply_theme(config.theme)
    ranked = importance.sort_values("importance_mean", ascending=False)
    top = list(ranked["feature"].head(3))
    data = ranked.iloc[::-1].copy()
    labels = data["feature"].map(FEATURE_LABELS).fillna(data["feature"])
    colors = [pal["accent"] if f in top else pal["primary"] for f in data["feature"]]
    fig, ax = plt.subplots(figsize=(9, 6.2), layout="constrained")
    ax.barh(labels, data["importance_mean"], xerr=data["importance_std"],
            color=colors, ecolor=pal["muted"], capsize=3)
    ax.axvline(0, color=pal["fg"], lw=0.8)
    ax.set_xlabel("Loss of average precision when the feature is shuffled\n"
                  "(larger = the model relies on it more)")
    ax.grid(axis="y", visible=False)
    names = [FEATURE_LABELS.get(f, f) for f in top]
    _headline(fig, f"What drives downgrade risk?\n{names[0]}, {names[1]} and "
                   f"{names[2]} carry most of the signal.")
    _footer(fig, "Synthetic data | permutation importance on the test set, "
                 "10 repeats, error bars = 1 standard deviation. Bars at or below zero mean "
                 "no reliable signal.", pal)
    return _save(fig, path, config)


def plot_portfolio_risk(
    summary: pd.DataFrame,
    losses: ArrayLike,
    var_es: dict[float, dict[str, float]],
    config: Config,
    path: Path,
) -> Path:
    """Plot where the risk sits and how bad a bad year can get.

    Panel A compares the share of exposure with the share of expected loss
    per rating class. Panel B shows the simulated loss distribution.

    Args:
        summary: Output of :func:`rating_summary`.
        losses: Simulated portfolio losses.
        var_es: Output of :func:`var_es_table`.
        config: Project configuration.
        path: Output file path.

    Returns:
        The output path.
    """
    pal = apply_theme(config.theme)
    arr = np.asarray(losses, dtype=float)
    total_ead = float(summary["exposure"].sum())
    exp_share = summary["exposure"].to_numpy() / total_ead * 100.0
    el_share = summary["el_total"].to_numpy() / summary["el_total"].sum() * 100.0
    x = np.arange(len(summary))
    width = 0.4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.4), layout="constrained")

    ax1.bar(x - width / 2, exp_share, width, color=pal["primary"],
            label="Share of exposure")
    ax1.bar(x + width / 2, el_share, width, color=pal["accent"],
            label="Share of expected loss")
    for xi, v in zip(x + width / 2, el_share):
        if v >= 1.0:
            ax1.text(xi, v + 0.6, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x, list(summary["rating"]))
    ax1.set(xlabel="Rating class", ylabel="% of portfolio",
            title="A. Where does the risk sit?")
    ax1.legend(loc="upper left")

    alpha = 0.99 if 0.99 in var_es else max(var_es)
    var, es, mean = var_es[alpha]["VaR"], var_es[alpha]["ES"], float(arr.mean())
    a_txt = f"{alpha * 100:g}%"
    ax2.hist(arr / 1e6, bins=80, color=pal["primary"], alpha=0.85)
    right = max(es * 1.15, float(np.quantile(arr, 0.9995))) / 1e6
    ax2.set_xlim(left=-0.02 * right, right=right)
    ax2.axvspan(var / 1e6, right, color=pal["accent"], alpha=0.15)
    ax2.axvline(mean / 1e6, color=pal["good"], lw=2,
                label=f"Mean loss: EUR {mean / 1e6:,.0f}m")
    ax2.axvline(var / 1e6, color=pal["warn"], ls="--", lw=2,
                label=f"VaR {a_txt}: EUR {var / 1e6:,.0f}m")
    ax2.axvline(es / 1e6, color=pal["accent"], ls=":", lw=2,
                label=f"Expected shortfall {a_txt}: EUR {es / 1e6:,.0f}m")
    ax2.set(xlabel="Simulated 1-year portfolio loss (EUR million)",
            ylabel="Number of simulations", title="B. How bad can one year get?")
    ax2.legend(loc="upper right", fontsize=9)

    hy = slice(RATINGS.index("BB"), None)
    hy_exp = float(exp_share[hy].sum())
    hy_el = float(el_share[hy].sum())
    _headline(
        fig,
        f"{hy_exp:.0f}% of exposure is rated BB or lower, but it causes {hy_el:.0f}% of "
        f"expected loss.\nIn the worst {(1 - alpha) * 100:g}% of years the loss exceeds "
        f"EUR {var / 1e6:,.0f}m ({var / total_ead:.1%} of exposure).",
    )
    _footer(
        fig,
        f"Synthetic data | test-set portfolio | expected loss = default loss + "
        f"downgrade (spread-widening) loss | Vasicek one-factor Monte Carlo, "
        f"{len(arr):,} simulations, asset correlation {config.asset_correlation:.2f} | "
        f"shaded area = worst {(1 - alpha) * 100:g}% of outcomes.",
        pal,
    )
    return _save(fig, path, config)


def plot_transition_matrix(matrix: pd.DataFrame, config: Config, path: Path) -> Path:
    """Plot the annual rating transition (migration) matrix.

    Args:
        matrix: Output of :func:`build_transition_matrix`.
        config: Project configuration.
        path: Output file path.

    Returns:
        The output path.
    """
    pal = apply_theme(config.theme)
    values = matrix.to_numpy()
    n = len(matrix)
    cmap = LinearSegmentedColormap.from_list("mig", [pal["bg"], pal["primary"]])
    norm_ = PowerNorm(0.4, vmin=0.0, vmax=1.0)
    fig, ax = plt.subplots(figsize=(10.5, 7.6), layout="constrained")
    ax.pcolormesh(values, cmap=cmap, norm=norm_, edgecolors=pal["bg"], linewidth=1)
    for i in range(n):
        for j in range(n):
            v = values[i, j]
            if v < 0.0005:
                continue
            r, g, b, _ = cmap(norm_(v))
            color = "#111111" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "#FFFFFF"
            ax.text(j + 0.5, i + 0.5, f"{v:.1%}", ha="center", va="center",
                    fontsize=8, color=color)
    ax.plot([0, n], [0, n], color=pal["fg"], lw=1, ls="--", alpha=0.6)
    ax.set_xticks(np.arange(n) + 0.5, list(matrix.columns))
    ax.set_yticks(np.arange(n) + 0.5, list(matrix.index))
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set(xlabel="Rating after 1 year  (D = default)",
           ylabel="Rating at start of year")
    idx = RATINGS.index("BBB")
    down = float(values[idx, idx + 1 :].sum())
    _headline(
        fig,
        f"A BBB bond is downgraded within a year with {down:.0%} probability "
        f"({values[idx, -1]:.1%} default).\nIt stays BBB with {values[idx, idx]:.0%}.",
    )
    _footer(
        fig,
        "Simulated yearly rating paths (synthetic) | read each row from left to right | "
        "right of the dashed diagonal = downgrade, left = upgrade | colors are "
        "stretched so that small probabilities stay visible.",
        pal,
    )
    return _save(fig, path, config)


# =============================================================================
# 6. COMMAND LINE INTERFACE AND SELF-CHECKS
# =============================================================================


def _to_python(obj: Any) -> Any:
    """Convert numpy and pandas objects to JSON-safe Python objects.

    Args:
        obj: Any object.

    Returns:
        A JSON-safe object. NaN becomes ``None``.
    """
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_python(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_self_checks(
    config: Config, df: pd.DataFrame, results: PipelineResults
) -> list[str]:
    """Run built-in sanity checks.

    Args:
        config: Project configuration.
        df: Portfolio used for training.
        results: Output of :func:`train_and_evaluate`.

    Returns:
        A list of failure messages. The list is empty if all checks pass.
    """
    failures: list[str] = []

    def check(name: str, ok: bool) -> None:
        LOGGER.info("Check %-34s %s", name, "OK" if ok else "FAILED")
        if not ok:
            failures.append(name)

    check("EL identity", bool(np.isclose(expected_loss(1000.0, 0.02, 0.45), 9.0)))
    check(
        "Duration sign",
        bool(
            duration_price_change(100.0, 5.0, 0.01) < 0
            and duration_price_change(100.0, 5.0, -0.01) > 0
        ),
    )
    sample = np.random.default_rng(0).lognormal(0.0, 1.0, 10_000)
    check(
        "VaR <= ES",
        all(
            value_at_risk(sample, a) <= expected_shortfall(sample, a)
            for a in config.alpha_levels
        ),
    )
    check("Seed reproducibility", generate_portfolio(config).equals(df))
    check("Base rate in range", 0.03 <= float(df[TARGET].mean()) <= 0.12)
    ids = {k: set(v["issuer_id"]) for k, v in results.splits.items()}
    check(
        "No issuer leakage",
        not (ids["train"] & ids["val"] or ids["train"] & ids["test"]
             or ids["val"] & ids["test"]),
    )
    val_recall = results.metrics["validation"]["recall_threshold"]["recall"]
    check("Recall target met (validation)", val_recall >= config.recall_target - 1e-9)
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list. If ``None``, use ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Credit migration early warning system (synthetic data)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--theme", choices=["light", "dark"], default="light",
                        help="Plot theme.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory.")
    parser.add_argument("--n-simulations", type=int, default=50_000,
                        help="Monte Carlo simulations for VaR and ES.")
    parser.add_argument("--skip-checks", action="store_true",
                        help="Skip the built-in self-checks.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the full workflow and save all artifacts.

    Args:
        argv: Argument list. If ``None``, use ``sys.argv``.

    Returns:
        Exit code. Zero means success.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = Config(
        seed=args.seed,
        theme=args.theme,
        output_dir=Path(args.output_dir),
        n_simulations=args.n_simulations,
    )
    out = config.ensure_output_dir()

    LOGGER.info("Generate synthetic portfolio (seed=%d).", config.seed)
    df = generate_portfolio(config)
    transition = build_transition_matrix(config)

    LOGGER.info("Train and evaluate the model.")
    results = train_and_evaluate(df, config)

    test = results.splits["test"]
    preds = results.predictions
    p_down = preds["proba_calibrated"].to_numpy()
    LOGGER.info("Simulate portfolio losses (%d simulations).", config.n_simulations)
    report, losses = build_risk_report(test, p_down, config)
    summary = rating_summary(test, p_down)
    var_es = var_es_table(losses, config.alpha_levels)
    costs = cost_matrix(test, config)
    y_te = preds["y_true"].to_numpy()

    LOGGER.info("Create figures.")
    thresholds = {
        f"Recall target ({config.recall_target:.0%})": results.threshold_recall,
        "Lowest total cost": results.threshold_cost,
    }
    plot_model_performance(y_te, preds["proba_raw"], p_down, thresholds, costs,
                           config, out / "1_model_performance.png")
    plot_feature_importance(results.importance, config, out / "2_risk_drivers.png")
    plot_portfolio_risk(summary, losses, var_es, config, out / "3_portfolio_risk.png")
    plot_transition_matrix(transition, config, out / "4_rating_migration.png")

    LOGGER.info("Save results.")
    payload = {
        "seed": config.seed,
        "model": results.metrics,
        "portfolio_risk": report,
        "rating_summary": summary.to_dict("records"),
        "transition_matrix": transition.round(6).to_dict(),
    }
    with open(out / "results.json", "w", encoding="utf-8") as fh:
        json.dump(_to_python(payload), fh, indent=2)
    preds.to_csv(out / "predictions.csv", index=False)
    joblib.dump(
        {
            "pipeline": results.model,
            "calibrator": results.calibrator,
            "threshold_recall": results.threshold_recall,
            "threshold_cost": results.threshold_cost,
            "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        },
        out / "model.joblib",
    )

    test_m = results.metrics["test"]
    rt = test_m["recall_threshold"]
    key = 0.99 if 0.99 in var_es else max(var_es)
    LOGGER.info("Test ROC-AUC %.3f | PR-AUC %.3f", test_m["roc_auc"], test_m["pr_auc"])
    LOGGER.info(
        "Recall-target threshold: alerts on %.0f%% of bonds, catches %.0f%% of "
        "downgrades, %.0f%% of alerts are right.",
        rt["alert_rate"] * 100, rt["recall"] * 100, rt["precision"] * 100,
    )
    LOGGER.info(
        "Portfolio: expected loss EUR %.1fm | VaR %g%% EUR %.1fm | ES %g%% EUR %.1fm",
        report["el_total_eur"] / 1e6, key * 100, var_es[key]["VaR"] / 1e6,
        key * 100, var_es[key]["ES"] / 1e6,
    )

    if not args.skip_checks:
        failures = run_self_checks(config, df, results)
        if failures:
            LOGGER.error("Self-checks failed: %s", ", ".join(failures))
            return 1
        LOGGER.info("All self-checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())