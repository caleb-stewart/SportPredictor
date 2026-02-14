# SportPredictor

Rails API + Flask microservice for WHL game outcome prediction.

## Required DB Migration

Before using Predictor V2 persistence fields:

```bash
bin/rails db:migrate
```

## Predictor V2 (Hybrid Ensemble)

The v2 predictor uses pre-trained artifacts (not per-request training) and expects a matchup payload with precomputed rolling features for `k=5/10/15`.

### Train and Promote Model

```bash
PredictorFlask/pf-venv/bin/python PredictorFlask/train_whl_v2.py \
  --db-host localhost \
  --db-port 5432 \
  --db-name sportpredictor_development \
  --db-user postgres \
  --db-password qqqq
```

This writes versioned artifacts under `PredictorFlask/model_store/whl_v2/<version>/` and updates `active_model.json` only if gates pass.

### Export Canonical Training Dataset

```bash
PredictorFlask/pf-venv/bin/python PredictorFlask/export_whl_v2_dataset.py \
  --db-host localhost \
  --db-port 5432 \
  --db-name sportpredictor_development \
  --db-user postgres \
  --db-password qqqq
```

### Rails Rake Tasks

```bash
bin/rake predictor_v2:train
bin/rake predictor_v2:predict_upcoming
bin/rake predictor_v2:daily_pipeline
```

### Flask API Contract (v2)

`POST /whl/calc_winner`

Request:

```json
{
  "game_id": 1022064,
  "game_date": "2026-02-15",
  "home_team_id": 215,
  "away_team_id": 206,
  "features_by_k": {
    "5": { "home": { "goals_for_avg": 3.2 }, "away": { "goals_for_avg": 2.9 } },
    "10": { "home": { "goals_for_avg": 3.1 }, "away": { "goals_for_avg": 2.8 } },
    "15": { "home": { "goals_for_avg": 3.0 }, "away": { "goals_for_avg": 2.7 } }
  }
}
```

Response:

```json
{
  "home_team_prob": 0.62,
  "away_team_prob": 0.38,
  "predicted_winner_id": 215,
  "model_version": "20260214T181647Z",
  "model_family": "whl_v2_hybrid_logistic_stacker",
  "k_components": {
    "5": { "home_team_prob": 0.60, "away_team_prob": 0.40 },
    "10": { "home_team_prob": 0.63, "away_team_prob": 0.37 },
    "15": { "home_team_prob": 0.65, "away_team_prob": 0.35 }
  }
}
```
