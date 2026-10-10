# V4 dynamic/hierarchical prototype

V4 is an opt-in model family. It combines:

- a date-embargoed approximate Bayesian dynamic Bradley–Terry rating with
  posterior uncertainty;
- dynamic team attack and defense states with hierarchical shrinkage toward a
  league scoring level;
- a Poisson or overdispersed negative-binomial predictive goal distribution;
- a small regularized combiner for the rating and goal-distribution win
  probabilities.

The implementation is in `services/v4_models.py` and
`services/v4_training.py`. It does not change the active V2/V3 pointer unless
the CLI is explicitly given `--promote`.

## Run a prototype

```bash
./.venv/bin/python scripts/train_v4.py \
  --league whl \
  --output-root model_store/whl_v4 \
  --model-version 20261003T000000Z
```

The run writes `metrics.json`, `metadata.json`, a Joblib bundle, and
`walk_forward_predictions.csv`. A real game is accepted only when its status
is final (or the historical WHL status code is `4`). Same-calendar-date
results are applied after every forecast for that date, because historical
start times are not reliable enough to establish an intra-day ordering.

## Initial WHL result

The first run used 22,140 eligible games and a chronological 80/20 holdout of
4,428 games:

| model | accuracy | log loss | Brier |
| --- | ---: | ---: | ---: |
| V3 rebuilt with corrected point-in-time data | 0.6472 | 0.6286 | 0.2195 |
| V4 dynamic rating + goal distribution | 0.6369 | 0.6391 | 0.2242 |
| home-rate baseline | 0.5479 | 0.6893 | 0.2481 |

The V4 goal branch is currently weaker than the existing feature ensemble; it
is useful as an interpretable branch and as a candidate blending input, but it
is not a replacement yet. The active model was not changed.

The V3 comparison artifact was trained with:

```bash
./.venv/bin/python -m services.trainer \
  --league-code whl \
  --output-root model_store/whl_v3_baseline \
  --model-version 20261003T000001Z \
  --no-promote
```

For a promotion decision, compare models on identical eligible game IDs and
use a separate untouched terminal period after any hyperparameter search. The
stored first-run metrics are a diagnostic baseline, not a promotion gate.

## Research basis

The filtering design follows the state-space skill-rating framework in
[Duffield et al. (2024)](https://academic.oup.com/jrsssc/article/73/5/1262/7734616).
The dynamic attack/defense goal formulation is informed by [Ridall, Titman,
and Pettitt (2025)](https://academic.oup.com/jrsssc/article/74/3/717/7929974),
and adaptive team-specific evolution is motivated by [Macrì-Demartino, Egidi,
and Torelli (2026)](https://academic.oup.com/jrsssc/advance-article/doi/10.1093/jrsssc/qlag032/8704597).
The hockey-specific Bayesian count-model motivation is described in [A
Bayesian Approach to Predict the Number of Goals in
Hockey](https://doi.org/10.3390/stats2020017).
