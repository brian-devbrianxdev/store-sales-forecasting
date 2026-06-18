# Store Sales — Time Series Forecasting

A modular pipeline for the Kaggle *Store Sales — Time Series Forecasting*
competition. It trains four complementary forecasting legs and combines them
with a minimum-variance ensemble blend.

## Layout

```
config.yaml                  # ALL paths, dates, hyperparameters, leg sigmas, variants
pyproject.toml               # package metadata + `store-sales` console script
requirements.txt
src/store_sales/
  config.py                  # typed dataclasses + load_config()
  paths.py                   # repo directory resolver
  metrics.py                 # rmsle() — single definition
  io/                        # raw-data loading + submission read/write
  features/                  # calendar/holiday/oil + per-(store,family) lag features
  models/                    # the four training legs (+ catboost member)
  ensemble/                  # minimum-variance blend + final build/verify
  cli.py                     # `store-sales` orchestrator
store_sales_forecasting_local.ipynb   # Vietnamese course report (orchestrates the CLI)
data/                        # raw competition CSVs (train.csv via Git LFS)
submissions/                 # per-leg prediction CSVs + the final blend
```

## Install

```bash
pip install -e .            # core (numpy/pandas/pyyaml) + the store-sales CLI
pip install -e ".[train]"   # add the heavy training deps (lightgbm/catboost/darts/torch/...)
```

## The four legs

| Leg | Module | Notes |
|-----|--------|-------|
| LightGBM v8 | `models/lgbm_regularized.py` | regularized per-family direct-per-horizon GBT |
| darts GBT family | `models/darts_family.py` | 6 variants (`base/deeper/xgb/subsampled/weighted/cat_deep`) → the `family` sub-blend |
| Neural | `models/neural_ts.py` | global TSMixer/TiDE/NHiTS (GPU) |
| Chronos-2 (+covariates) | `models/chronos2_cov.py` | foundation model; **needs a separate env** |
| CatBoost member | `models/catboost_family.py` | diversity member reusing the v8 features |

### Chronos-2 environment

The Chronos legs require Python 3.11 + `chronos-forecasting>=2`, separate from
the main env (which ships chronos 1.5.3 without `Chronos2Pipeline`):

```bash
python3.11 -m venv .venv_chronos2
.venv_chronos2/bin/pip install chronos-forecasting torch pandas numpy pyyaml
PYTHONPATH=src .venv_chronos2/bin/python -m store_sales.cli train chronos2-cov --suffix _promo
```

## Usage

```bash
store-sales train lgbm-v8 --suffix reg
store-sales train darts-family --variant deeper
store-sales train tsmixer --epochs 30
store-sales blend build         # write the final 4-way + family CSVs
store-sales blend verify        # assert byte-exact reproduction of the committed blend
store-sales run-all             # every in-process leg, then the blend
```

All tuning lives in `config.yaml` — no hyperparameters are hardcoded in the
execution logic. The per-leg Kaggle leaderboard RMSLE values (`ensemble.*_sigma`)
are obtained by submitting each leg's CSV and drive the blend weights.

## Reproducibility

The ensemble blend reconstructs each leg's error covariance from its LB sigma
plus pairwise prediction differences — no ground truth needed. `blend verify`
rebuilds the blend from the committed leg CSVs and asserts a byte-exact match
against `submissions/submission_fam_cov_v8_tsmTuned_4way.csv`.
