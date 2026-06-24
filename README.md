# Store Sales — Time Series Forecasting

A modular pipeline for the Kaggle *Store Sales — Time Series Forecasting*
competition. It trains five complementary forecasting legs (family GBT,
Chronos-2, LightGBM-v8, TSMixer, TiDE) and combines them with a
**minimum-variance ensemble blend** (in `log1p` space). The metric is RMSLE;
the final score comes from submitting the blended CSV to Kaggle.

---

## 1. How to read the source

Everything lives under `src/store_sales/`. Start at the entry point and follow
the imports:

```
config.yaml                  # SINGLE source of truth: paths, dates, hyperparameters,
                             #   variant definitions, leg sigmas. Nothing is hardcoded
                             #   in the execution logic — tune here.
src/store_sales/
  cli.py                     # START HERE — the `store-sales` command dispatcher
                             #   (train <leg> | blend build|verify | run-all)
  config.py                  # typed dataclasses + load_config() that parse config.yaml
  paths.py                   # resolves repo root + data/ and submissions/ dirs
  metrics.py                 # rmsle() — the single definition
  io/
    data_loading.py          # load_raw_frames(): read the competition CSVs
    submissions.py           # load_log() / write_submission()
  features/
    calendar.py              # calendar, holiday table, oil, transactions
    lgbm_features.py         # per-(store,family) lag / rolling / promo features
    floor.py                 # regenerates data/floor_per_row.parquet for the
                             #   darts "weighted" variant (see §5)
  models/                    # the four training legs (one file each):
    lgbm_regularized.py      #   - LightGBM v8 (regularized, direct-per-horizon)
    darts_family.py          #   - darts GBT family (6 variants -> `family` sub-blend)
    catboost_family.py       #   - CatBoost diversity member
    neural_ts.py             #   - global TSMixer / TiDE / NHiTS
    chronos2.py              #   - Chronos-2 zero-shot
    chronos2_cov.py          #   - Chronos-2 with covariates (needs .venv_chronos2)
  ensemble/
    blend.py                 # min-variance math (cov reconstruction + weights)
    build.py                 # build the final 5-way + family CSVs; verify byte-exact
store_sales_forecasting_local.ipynb   # Vietnamese course report; orchestrates the CLI
data/                        # raw competition CSVs (train.csv via Git LFS)
submissions/                 # per-leg prediction CSVs + the final blend
```

The cleanest reading path: `cli.py` → the leg module you care about in
`models/` → its feature helpers in `features/` → `ensemble/build.py` for how the
legs are combined. All knobs are in `config.yaml`.

---

## 2. Install (main environment)

```bash
pip install -e .            # core: numpy/pandas/pyyaml + the `store-sales` CLI
pip install -e ".[train]"   # add heavy training deps: lightgbm/catboost/xgboost/
                            #   darts/torch/scikit-learn/pyarrow
```

The blend (`store-sales blend ...`) only needs the core install — it works from
the committed leg CSVs. Training the legs needs the `[train]` extras.

---

## 3. How to run

### Via the CLI

```bash
# train individual legs (CSVs land in submissions/)
store-sales train darts-family --variant base      # also: deeper xgb subsampled weighted cat_deep
store-sales train lgbm-v8 --suffix reg
store-sales train tsmixer --epochs 30
store-sales train neural --model tide --epochs 30   # extra neural leg (also: --model nhits)
store-sales train catboost

# build / check the ensemble
store-sales blend build        # write submission_family.csv + the final 5-way CSV
store-sales blend verify       # assert byte-exact reproduction of the committed blend

store-sales run-all            # every in-process leg (darts + v8 + tsmixer), then build
```

`run-all` does NOT run the Chronos legs — they need a separate interpreter (§4).

### Via the notebook

`store_sales_forecasting_local.ipynb` is a report that drives the same CLI via a
`run()` helper. Two flags at the top control it:

- `RUN_LEGS` — `True` = train the legs yourself; `False` = reassemble-only from
  existing CSVs.
- `FORCE_RETRAIN` — `True` = retrain even if a leg's CSV already exists.

A leg that targets a missing/incomplete interpreter (e.g. the Chronos venv) is
**skipped gracefully** and falls back to the committed CSV instead of crashing.

---

## 4. The Chronos-2 environment (`.venv_chronos2`)

The Chronos legs (`models/chronos2*.py`) need **Python 3.11 +
`chronos-forecasting==2.2.2`**, kept in a SEPARATE venv from the main env. Set it
up once:

```bash
# 1. create the venv (Python 3.11 specifically)
python3.11 -m venv .venv_chronos2

# 2. install deps (chronos-forecasting 2.2.2 is on PyPI; it pulls in torch + transformers)
.venv_chronos2/bin/python -m pip install --upgrade pip
.venv_chronos2/bin/python -m pip install numpy pandas pyyaml pyarrow "chronos-forecasting==2.2.2"

# 3. sanity check
.venv_chronos2/bin/python -c "from chronos import Chronos2Pipeline; print('chronos OK')"

# 4. run the covariate Chronos leg (PYTHONPATH=src so the package resolves without an install)
PYTHONPATH=src .venv_chronos2/bin/python -m store_sales.cli train chronos2-cov --suffix _promo
```

Notes:
- The model `amazon/chronos-2` is downloaded from Hugging Face on first run.
- It runs on CPU by default (`device_map="cpu"` in `chronos2_cov.py`); inference
  over all 1,782 series takes ~20s.
- `.venv_chronos2/` is git-ignored (it is ~1 GB) — recreate it with the steps
  above on a new machine.

---

## 5. Sigmas, submission, and the final score

The blend weights come from each leg's **standalone Kaggle leaderboard RMSLE
(`sigma`)**, stored in `config.yaml` under `ensemble.family_sigma` /
`ensemble.leg_sigma`. Each comment there names the CSV you submit to obtain that
number. Workflow:

1. Submit the **6 darts member CSVs** → fill `ensemble.family_sigma`.
2. `store-sales blend build` writes `submission_family.csv`; submit it →
   `ensemble.leg_sigma.family`.
3. Submit `submission_chronos2_cov_promo_oil_hol.csv`, `submission_v8_reg.csv`,
   `submission_tsmixer_tuned.csv`, `submission_tide.csv` → the other four
   `leg_sigma` values.
4. Re-run `blend build` with the final sigmas → writes
   `submission_fam_cov_v8_tsm_tide_5way.csv`.
5. **Submit that file** → its Kaggle LB score is the project's final RMSLE
   (this pipeline reaches **0.37379** on the public leaderboard).

The `math_LB` value printed by `blend build` is only an *estimate* — the official
RMSLE is the Kaggle score of the submitted blend.

---

## 6. Reproducibility

`blend verify` rebuilds the blend from the committed leg CSVs and asserts a
byte-exact match against `submissions/submission_fam_cov_v8_tsm_tide_5way.csv`.
The minimum-variance math reconstructs each leg's error covariance from its LB
sigma plus pairwise prediction differences — no ground-truth labels needed:

```
Cov_ij = (sigma_i^2 + sigma_j^2 - D_ij^2) / 2,   w = Σ⁻¹·1 / (1ᵀ·Σ⁻¹·1)
```

> Note: legs retrained locally (different library versions / hardware, or the
> regenerated `floor_per_row.parquet`) will not byte-match the committed CSVs;
> that is expected — re-measure their sigma on Kaggle.
