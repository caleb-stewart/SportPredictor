from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from services.trainer import TrainerConfig, train_model_package
from services.training_dataset import DbConfig
from services.training_features import FEATURE_COLUMNS, K_VALUES


def _synthetic_dataset(num_games: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    start = dt.date(2021, 1, 1)

    for game_idx in range(num_games):
        game_id = 100000 + game_idx
        game_date = start + dt.timedelta(days=game_idx)
        season_id = "2024"
        home_team_id = 200 + (game_idx % 12)
        away_team_id = 300 + (game_idx % 12)

        for k in K_VALUES:
            base = rng.normal(loc=0.0, scale=1.0, size=len(FEATURE_COLUMNS))
            signal = 0.35 * (home_team_id - away_team_id) / 100.0
            feat_vals = {col: float(base[i] + signal) for i, col in enumerate(FEATURE_COLUMNS)}

            home_win = int((feat_vals["goals_for_avg_diff"] + feat_vals["shots_for_avg_diff"]) > 0.0)
            rows.append(
                {
                    "game_id": game_id,
                    "game_date": pd.Timestamp(game_date),
                    "season_id": season_id,
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "k_value": k,
                    "home_win": home_win,
                    **feat_vals,
                }
            )

    frame = pd.DataFrame(rows)
    return frame.sort_values(["game_date", "game_id", "k_value"]).reset_index(drop=True)


def _config(output_root: Path, version: str) -> TrainerConfig:
    return TrainerConfig(
        output_root=output_root,
        model_version=version,
        no_promote=True,
        export_dataset_path=None,
        min_accuracy=0.50,
        min_season_accuracy=0.45,
        min_season_games=10,
        base_oof_splits=3,
        meta_cv_splits=3,
        min_train_size=40,
        db=DbConfig(),
    )


def test_train_model_package_is_deterministic(monkeypatch, tmp_path: Path):
    dataset = _synthetic_dataset()
    monkeypatch.setattr("services.trainer.load_canonical_dataset", lambda _db: dataset.copy())

    details1, code1 = train_model_package(_config(tmp_path / "run1", "unit-v1"))
    details2, code2 = train_model_package(_config(tmp_path / "run2", "unit-v2"))

    assert code1 in (0, 2)
    assert code2 in (0, 2)
    assert details1["gates"] == details2["gates"]
    assert details1["chosen_model"]["name"] == details2["chosen_model"]["name"]
