"""V4 dynamic Bayesian team-state and hierarchical goal models.

This module deliberately keeps the first V4 prototype self contained.  It uses
assumed-density filtering (a Gaussian approximation to the posterior) rather
than a heavyweight MCMC dependency.  The important property for deployment is
that the feature replay is strictly pregame and deterministic: a game's result
is applied only after every game on that calendar date has received its
forecast.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


V4_FEATURE_COLUMNS: list[str] = [
    "v4_rating_home_mean",
    "v4_rating_away_mean",
    "v4_rating_diff_mean",
    "v4_rating_home_variance",
    "v4_rating_away_variance",
    "v4_rating_diff_variance",
    "v4_rating_home_prob",
    "v4_goal_lambda_home",
    "v4_goal_lambda_away",
    "v4_goal_log_rate_diff",
    "v4_goal_home_prob",
    "v4_goal_home_attack_mean",
    "v4_goal_away_attack_mean",
    "v4_goal_home_defense_mean",
    "v4_goal_away_defense_mean",
    "v4_goal_home_variance",
    "v4_goal_away_variance",
]

V4_PROBABILITY_COLUMNS: list[str] = [
    "v4_rating_home_prob",
    "v4_goal_home_prob",
]


def _clip_probability(value: float) -> float:
    return float(np.clip(float(value), 1e-6, 1.0 - 1e-6))


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-value))


def _as_date(value: Any) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


def _poisson_pmf(mean: float, max_goals: int = 20) -> np.ndarray:
    """Return a stable, truncated Poisson PMF for a hockey goal rate."""
    rate = float(np.clip(mean, 1e-6, 40.0))
    values = np.zeros(max_goals + 1, dtype=float)
    values[0] = math.exp(-rate)
    for goal in range(1, max_goals + 1):
        values[goal] = values[goal - 1] * rate / float(goal)
    total = values.sum()
    return values / total if total > 0 else values


def _negative_binomial_pmf(mean: float, overdispersion: float, max_goals: int = 20) -> np.ndarray:
    """Mean-parameterized negative-binomial PMF for overdispersed goals."""
    rate = float(np.clip(mean, 1e-6, 40.0))
    size = 1.0 / max(float(overdispersion), 1e-6)
    success = size / (size + rate)
    values = np.zeros(max_goals + 1, dtype=float)
    values[0] = success**size
    for goal in range(1, max_goals + 1):
        values[goal] = values[goal - 1] * (size + goal - 1.0) / goal * (1.0 - success)
    total = values.sum()
    return values / total if total > 0 else values


def poisson_home_win_probability(
    home_mean: float,
    away_mean: float,
    overdispersion: float = 0.0,
) -> float:
    """Convert two predictive goal rates into P(home wins)."""
    pmf = _negative_binomial_pmf if overdispersion > 0 else _poisson_pmf
    home = pmf(home_mean, overdispersion) if overdispersion > 0 else pmf(home_mean)
    away = pmf(away_mean, overdispersion) if overdispersion > 0 else pmf(away_mean)
    cumulative_away_below = np.cumsum(away) - away
    return _clip_probability(float(np.dot(home, cumulative_away_below)))


@dataclass
class GaussianTeamState:
    mean: float = 0.0
    variance: float = 1.0
    games: int = 0
    last_date: dt.date | None = None


@dataclass(frozen=True)
class V4ModelConfig:
    """Tunable filtering hyperparameters.

    Values are on the natural-logit scale for ratings and natural-log scale
    for goal rates.  They are intentionally explicit so a walk-forward search
    can record every trial instead of hiding tuning inside the model code.
    """

    rating_prior_variance: float = 1.0
    rating_drift_variance_per_day: float = 0.0025
    rating_home_advantage: float = 0.14
    rating_home_advantage_variance: float = 0.25
    goal_prior_variance: float = 0.35
    goal_drift_variance_per_day: float = 0.0015
    goal_prior_log_rate: float = math.log(2.8)
    goal_home_advantage: float = 0.07
    goal_overdispersion: float = 0.0
    hierarchy_strength: float = 8.0


class DynamicBayesianRating:
    """Approximate Bayesian dynamic Bradley-Terry rating filter."""

    def __init__(self, config: V4ModelConfig | None = None) -> None:
        self.config = config or V4ModelConfig()
        self.states: dict[int, GaussianTeamState] = {}
        self.home_advantage = GaussianTeamState(
            mean=self.config.rating_home_advantage,
            variance=self.config.rating_home_advantage_variance,
        )

    def _state(self, team_id: int) -> GaussianTeamState:
        if team_id not in self.states:
            self.states[team_id] = GaussianTeamState(
                variance=self.config.rating_prior_variance
            )
        return self.states[team_id]

    def _advance(self, state: GaussianTeamState, game_date: dt.date) -> None:
        if state.last_date is not None:
            gap = max((game_date - state.last_date).days, 1)
            state.variance += self.config.rating_drift_variance_per_day * gap
        state.last_date = game_date

    def pregame(self, home_id: int, away_id: int, game_date: dt.date) -> dict[str, float]:
        home = self._state(home_id)
        away = self._state(away_id)
        self._advance(home, game_date)
        self._advance(away, game_date)

        diff = home.mean - away.mean + self.home_advantage.mean
        diff_variance = home.variance + away.variance + self.home_advantage.variance
        # Logistic-normal moment approximation; uncertainty pulls probabilities
        # toward 0.5, which is preferable to overconfident cold-start ratings.
        scaled_diff = diff / math.sqrt(1.0 + math.pi * diff_variance / 8.0)
        return {
            "v4_rating_home_mean": float(home.mean),
            "v4_rating_away_mean": float(away.mean),
            "v4_rating_diff_mean": float(home.mean - away.mean),
            "v4_rating_home_variance": float(home.variance),
            "v4_rating_away_variance": float(away.variance),
            "v4_rating_diff_variance": float(diff_variance),
            "v4_rating_home_prob": _clip_probability(_sigmoid(scaled_diff)),
        }

    def update(
        self,
        home_id: int,
        away_id: int,
        home_win: int | None,
        game_date: dt.date,
    ) -> None:
        home = self._state(home_id)
        away = self._state(away_id)
        self._advance(home, game_date)
        self._advance(away, game_date)
        if home_win is not None:
            p = _sigmoid(home.mean - away.mean + self.home_advantage.mean)
            residual = float(home_win) - p
            information = max(p * (1.0 - p), 1e-5)

            for state, sign in ((home, 1.0), (away, -1.0)):
                posterior_precision = 1.0 / max(state.variance, 1e-6) + information
                posterior_variance = 1.0 / posterior_precision
                state.mean += posterior_variance * sign * residual
                state.variance = posterior_variance

            home_precision = 1.0 / max(self.home_advantage.variance, 1e-6) + information
            self.home_advantage.mean += (1.0 / home_precision) * residual
            self.home_advantage.variance = 1.0 / home_precision

        home.games += 1
        away.games += 1
        home.last_date = game_date
        away.last_date = game_date


class HierarchicalGoalDistribution:
    """Dynamic hierarchical attack/defense model with a Poisson predictive law."""

    def __init__(self, config: V4ModelConfig | None = None) -> None:
        self.config = config or V4ModelConfig()
        self.attack: dict[int, GaussianTeamState] = {}
        self.defense: dict[int, GaussianTeamState] = {}
        self.log_rate = self.config.goal_prior_log_rate
        self.log_rate_observations = self.config.hierarchy_strength
        self.home_advantage = self.config.goal_home_advantage

    def _state(self, states: dict[int, GaussianTeamState], team_id: int) -> GaussianTeamState:
        if team_id not in states:
            states[team_id] = GaussianTeamState(
                variance=self.config.goal_prior_variance
            )
        return states[team_id]

    def _advance(self, state: GaussianTeamState, game_date: dt.date) -> None:
        if state.last_date is not None:
            gap = max((game_date - state.last_date).days, 1)
            state.variance += self.config.goal_drift_variance_per_day * gap
        state.last_date = game_date

    def _linear_predictor(self, home_id: int, away_id: int, game_date: dt.date) -> tuple[float, float]:
        home_attack = self._state(self.attack, home_id)
        away_attack = self._state(self.attack, away_id)
        home_defense = self._state(self.defense, home_id)
        away_defense = self._state(self.defense, away_id)
        for state in (home_attack, away_attack, home_defense, away_defense):
            self._advance(state, game_date)

        home_log_rate = self.log_rate + self.home_advantage + home_attack.mean - away_defense.mean
        away_log_rate = self.log_rate + away_attack.mean - home_defense.mean
        return home_log_rate, away_log_rate

    def pregame(self, home_id: int, away_id: int, game_date: dt.date) -> dict[str, float]:
        home_log_rate, away_log_rate = self._linear_predictor(home_id, away_id, game_date)
        home_attack = self._state(self.attack, home_id)
        away_attack = self._state(self.attack, away_id)
        home_defense = self._state(self.defense, home_id)
        away_defense = self._state(self.defense, away_id)
        home_mean = float(np.exp(np.clip(home_log_rate, -5.0, 4.0)))
        away_mean = float(np.exp(np.clip(away_log_rate, -5.0, 4.0)))
        return {
            "v4_goal_lambda_home": home_mean,
            "v4_goal_lambda_away": away_mean,
            "v4_goal_log_rate_diff": float(home_log_rate - away_log_rate),
            "v4_goal_home_prob": poisson_home_win_probability(
                home_mean,
                away_mean,
                overdispersion=self.config.goal_overdispersion,
            ),
            "v4_goal_home_attack_mean": float(home_attack.mean),
            "v4_goal_away_attack_mean": float(away_attack.mean),
            "v4_goal_home_defense_mean": float(home_defense.mean),
            "v4_goal_away_defense_mean": float(away_defense.mean),
            "v4_goal_home_variance": float(home_attack.variance + away_defense.variance),
            "v4_goal_away_variance": float(away_attack.variance + home_defense.variance),
        }

    def _update_component(
        self,
        state: GaussianTeamState,
        residual: float,
        information: float,
        sign: float,
        game_date: dt.date,
    ) -> None:
        posterior_precision = 1.0 / max(state.variance, 1e-6) + max(information, 1e-5)
        posterior_variance = 1.0 / posterior_precision
        state.mean += posterior_variance * sign * residual
        state.variance = posterior_variance
        state.games += 1
        state.last_date = game_date

    def update(
        self,
        home_id: int,
        away_id: int,
        home_goals: float | None,
        away_goals: float | None,
        game_date: dt.date,
    ) -> None:
        if home_goals is None or away_goals is None:
            return
        home_log_rate, away_log_rate = self._linear_predictor(home_id, away_id, game_date)
        home_mean = float(np.exp(np.clip(home_log_rate, -5.0, 4.0)))
        away_mean = float(np.exp(np.clip(away_log_rate, -5.0, 4.0)))
        home_residual = float(home_goals) - home_mean
        away_residual = float(away_goals) - away_mean

        home_attack = self._state(self.attack, home_id)
        away_attack = self._state(self.attack, away_id)
        home_defense = self._state(self.defense, home_id)
        away_defense = self._state(self.defense, away_id)
        self._update_component(home_attack, home_residual, home_mean, 1.0, game_date)
        self._update_component(away_defense, home_residual, home_mean, -1.0, game_date)
        self._update_component(away_attack, away_residual, away_mean, 1.0, game_date)
        self._update_component(home_defense, away_residual, away_mean, -1.0, game_date)

        # Conjugate-style empirical Bayes update for the population scoring
        # level.  The pseudo-observations supply the hierarchical shrinkage.
        observed_log_rate = math.log((float(home_goals) + float(away_goals) + 1.0) / 2.0)
        self.log_rate = (
            self.log_rate * self.log_rate_observations + observed_log_rate
        ) / (self.log_rate_observations + 1.0)
        self.log_rate_observations += 1.0


def is_completed_game(status: Any, league_code: str | None = None) -> bool:
    """Return whether a provider status is safe to use as a final result."""
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return False
    normalized = str(status).strip().lower()
    if normalized.startswith("final"):
        return True
    # One historical WHL feed uses the numeric final status "4".
    return (league_code or "").strip().lower() == "whl" and normalized == "4"


def _regulation_goals_from_breakdown(value: Any, side: str) -> float | None:
    """Extract regulation-period goals from a provider scoring breakdown."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    raw = value
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None
    if not isinstance(raw, Mapping):
        return None
    side_value = raw.get(side) or raw.get("away" if side == "visiting" else "visiting")
    if not isinstance(side_value, Mapping):
        return None
    period_values: list[float] = []
    for period in ("1", "2", "3"):
        if period not in side_value:
            return None
        try:
            period_values.append(float(side_value[period]))
        except (TypeError, ValueError):
            return None
    return float(sum(period_values))


def prepare_v4_games(games: pd.DataFrame, league_code: str | None = None) -> pd.DataFrame:
    """Normalize and filter raw games for point-in-time V4 replay.

    Rows with no status are accepted for synthetic/unit-test frames.  Real
    provider frames should always carry status; non-final 0-0 rows are rejected
    so scheduled/postponed records cannot masquerade as completed games.
    """
    required = {"game_id", "game_date", "home_team_id", "away_team_id", "home_goal_count", "away_goal_count"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Missing V4 game columns: {sorted(missing)}")

    frame = games.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.date
    frame = frame.dropna(subset=list(required))
    if "status" in frame.columns:
        status_present = frame["status"].notna()
        completed = frame["status"].map(lambda value: is_completed_game(value, league_code))
        frame = frame[(~status_present) | completed]
    frame["home_goal_count"] = frame["home_goal_count"].astype(float)
    frame["away_goal_count"] = frame["away_goal_count"].astype(float)
    if "scoring_breakdown" in frame.columns:
        frame["goal_model_home_count"] = frame["scoring_breakdown"].map(
            lambda value: _regulation_goals_from_breakdown(value, "home")
        )
        frame["goal_model_away_count"] = frame["scoring_breakdown"].map(
            lambda value: _regulation_goals_from_breakdown(value, "visiting")
        )
    else:
        frame["goal_model_home_count"] = frame["home_goal_count"]
        frame["goal_model_away_count"] = frame["away_goal_count"]
    frame["goal_model_home_count"] = frame["goal_model_home_count"].fillna(frame["home_goal_count"])
    frame["goal_model_away_count"] = frame["goal_model_away_count"].fillna(frame["away_goal_count"])
    frame["home_team_id"] = frame["home_team_id"].astype(int)
    frame["away_team_id"] = frame["away_team_id"].astype(int)
    frame["home_win"] = np.where(
        frame["home_goal_count"] > frame["away_goal_count"], 1.0,
        np.where(frame["home_goal_count"] < frame["away_goal_count"], 0.0, np.nan),
    )
    return frame.sort_values(["game_date", "game_id"]).drop_duplicates("game_id").reset_index(drop=True)


def replay_v4_features(
    games: pd.DataFrame,
    config: V4ModelConfig | None = None,
    league_code: str | None = None,
) -> pd.DataFrame:
    """Create strictly pregame V4 features using a date-level embargo."""
    frame = prepare_v4_games(games, league_code=league_code)
    if frame.empty:
        return frame.assign(**{column: pd.Series(dtype=float) for column in V4_FEATURE_COLUMNS})

    rating = DynamicBayesianRating(config)
    goals = HierarchicalGoalDistribution(config)
    output: list[dict[str, Any]] = []

    for _, date_group in frame.groupby("game_date", sort=True):
        pending: list[tuple[int, int, int, dt.date, float, float, int | None]] = []
        for row in date_group.itertuples(index=False):
            game_date = _as_date(row.game_date)
            home_id = int(row.home_team_id)
            away_id = int(row.away_team_id)
            features = {
                **rating.pregame(home_id, away_id, game_date),
                **goals.pregame(home_id, away_id, game_date),
            }
            features["game_id"] = int(row.game_id)
            features["game_date"] = game_date
            features["home_win"] = None if pd.isna(row.home_win) else int(row.home_win)
            output.append(features)
            pending.append(
                (
                    home_id,
                    away_id,
                    int(row.game_id),
                    game_date,
                    float(row.goal_model_home_count),
                    float(row.goal_model_away_count),
                    None if pd.isna(row.home_win) else int(row.home_win),
                )
            )

        # Apply every result only after the date's forecasts were emitted.
        for home_id, away_id, _, game_date, home_goals, away_goals, home_win in pending:
            rating.update(home_id, away_id, home_win, game_date)
            goals.update(home_id, away_id, home_goals, away_goals, game_date)

    return pd.DataFrame(output).sort_values(["game_date", "game_id"]).reset_index(drop=True)


def compute_v4_context_features(
    historical_games: Iterable[Mapping[str, Any]],
    home_team_id: int,
    away_team_id: int,
    game_date: dt.date,
    config: V4ModelConfig | None = None,
    league_code: str | None = None,
) -> dict[str, float]:
    """Return a live-prediction feature snapshot from games strictly before a date."""
    rows = list(historical_games)
    if rows:
        frame = prepare_v4_games(pd.DataFrame(rows), league_code=league_code)
        frame = frame[frame["game_date"] < game_date].reset_index(drop=True)
    else:
        frame = pd.DataFrame(
            columns=[
                "game_id", "game_date", "home_team_id", "away_team_id",
                "home_goal_count", "away_goal_count", "home_win",
            ]
        )

    rating = DynamicBayesianRating(config)
    goals = HierarchicalGoalDistribution(config)
    if not frame.empty:
        for _, date_group in frame.groupby("game_date", sort=True):
            for row in date_group.itertuples(index=False):
                row_date = _as_date(row.game_date)
                rating.pregame(int(row.home_team_id), int(row.away_team_id), row_date)
                goals.pregame(int(row.home_team_id), int(row.away_team_id), row_date)
            for row in date_group.itertuples(index=False):
                row_date = _as_date(row.game_date)
                home_win = None if pd.isna(row.home_win) else int(row.home_win)
                rating.update(int(row.home_team_id), int(row.away_team_id), home_win, row_date)
                goals.update(
                    int(row.home_team_id),
                    int(row.away_team_id),
                    float(row.goal_model_home_count),
                    float(row.goal_model_away_count),
                    row_date,
                )

    return {
        **rating.pregame(int(home_team_id), int(away_team_id), game_date),
        **goals.pregame(int(home_team_id), int(away_team_id), game_date),
    }


def fit_v4_meta_model(
    feature_frame: pd.DataFrame,
    train_end: int | None = None,
    baseline_probabilities: Iterable[float] | None = None,
) -> tuple[Any, list[str]]:
    """Fit a small regularized probability combiner on pregame forecasts."""
    frame = feature_frame.reset_index(drop=True)
    end = train_end if train_end is not None else len(frame)
    columns = list(V4_PROBABILITY_COLUMNS)
    values = [frame[column].to_numpy(dtype=float) for column in columns]
    if baseline_probabilities is not None:
        baseline = np.asarray(list(baseline_probabilities), dtype=float)
        if len(baseline) != len(frame):
            raise ValueError("baseline_probabilities must align with feature_frame")
        frame = frame.copy()
        frame["v4_baseline_prob"] = baseline
        columns.append("v4_baseline_prob")
        values.append(baseline)
    X = np.column_stack(values)
    valid = np.isfinite(X).all(axis=1) & frame["home_win"].notna().to_numpy()
    train_valid = valid.copy()
    train_valid[end:] = False
    y = frame["home_win"].to_numpy(dtype=float)
    if train_valid.sum() < 8 or len(np.unique(y[train_valid])) < 2:
        # A neutral combiner is safer than a degenerate one-class logistic fit.
        return _ProbabilityCombiner(columns=columns, coefficients=None, intercept=0.0), columns

    estimator = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    estimator.fit(X[train_valid], y[train_valid].astype(int))
    return _ProbabilityCombiner(
        columns=columns,
        coefficients=estimator.coef_[0].astype(float),
        intercept=float(estimator.intercept_[0]),
    ), columns


@dataclass
class _ProbabilityCombiner:
    columns: list[str]
    coefficients: np.ndarray | None
    intercept: float

    def predict_proba(self, values: Mapping[str, float] | np.ndarray) -> float:
        if isinstance(values, Mapping):
            vector = np.asarray([float(values[column]) for column in self.columns], dtype=float)
        else:
            vector = np.asarray(values, dtype=float)
        if self.coefficients is None:
            return _clip_probability(float(np.mean(vector)))
        return _clip_probability(_sigmoid(self.intercept + float(np.dot(self.coefficients, vector))))


def v4_probability_from_context(
    context_features: Mapping[str, Any],
    combiner: _ProbabilityCombiner | None = None,
) -> dict[str, float]:
    """Produce V4 component and final probabilities for the API adapter."""
    rating = _clip_probability(float(context_features["v4_rating_home_prob"]))
    goals = _clip_probability(float(context_features["v4_goal_home_prob"]))
    values: dict[str, float] = {
        "v4_rating_home_prob": rating,
        "v4_goal_home_prob": goals,
    }
    if combiner is not None and "v4_baseline_prob" in combiner.columns:
        baseline = context_features.get("v4_baseline_prob", context_features.get("baseline_home_prob"))
        if baseline is None:
            raise KeyError("v4_baseline_prob")
        values["v4_baseline_prob"] = _clip_probability(float(baseline))
    if combiner is None:
        final = _clip_probability((rating + goals) / 2.0)
    else:
        final = combiner.predict_proba(values)
    values["home_prob"] = final
    values["away_prob"] = 1.0 - final
    return values
