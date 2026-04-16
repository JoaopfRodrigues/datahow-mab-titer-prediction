# mAb Bioprocess Titer Prediction

Predictive model and REST inference server for monoclonal antibody (mAb) titer in simulated upstream bioprocesses.

## Architecture

```
src/
  features.py   — Feature engineering (44 features from time-series + DoE params)
  api.py        — FastAPI inference server (GET /health, POST /predict)
models/
  titer_model.pkl       — Serialized ElasticNet model + bootstrap ensemble
  test_predictions.csv  — Predictions for 20 test experiments
notebooks/
  01_eda.ipynb        — Exploratory data analysis
  02_modeling.ipynb    — Model training and selection
tests/
  test_features.py, test_api.py, test_model.py
```

## Quick Start

```bash
# With Poetry
poetry install
poetry run uvicorn src.api:app --port 8000

# With Docker
docker build -t mab-titer .
docker run -p 8000:8000 mab-titer

# Tests
poetry run pytest
```

## Using the API

Start the server with `poetry run uvicorn src.api:app --port 8000`, then:

**Health check:**
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

**Predict for a single experiment:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @scripts/sample_payload.json
# {"predicted_titer": 2564.12, "confidence_lower": 2103.45, "confidence_upper": 3021.78}
```

See `scripts/sample_payload.json` for the expected payload shape (one experiment = `timestamps` + `values` dict with Z:/W:/X: variables).

For evaluating interview test targets, see `notebooks/02_modeling.ipynb` cell 25 — drop the targets CSV into `interview_files/` and uncomment the second `test_targets_path` line.

## Design Decisions

### Feature Engineering

44 features extracted per experiment in three groups:
- **Z: scalar parameters** (13): DoE design settings, directly from the data
- **Full-duration features** (25): IVCD, VCD dynamics, metabolite endpoints, integrals, metabolic rates. Grounded in bioprocess literature (Clavaud 2013: iVCC; Hutter 2021: derivative-based features).
- **Common-window features** (6): Structural features from the shared days 0-7 window — VCD trajectory shape, growth fraction/acceleration, early-window IVCD and growth rate. These address the train/test duration mismatch.

### Feature Selection

Starting from 44 features, a two-stage selection process:
1. **Domain pruning (44 → 35)**: Removed 9 features identified in EDA as pure duration proxies (partial r ≈ 0), exact linear combinations, or redundant with existing features.
2. **RFE with ElasticNetCV (35 → 13)**: Recursive feature elimination using cross-validated ElasticNet. Performance improves steadily from 35 to ~13 features, then levels off. We select 13 for parsimony (p/n ≈ 0.13, consistent with Hastie et al. 2009).

### Cross-Validation

Standard K-Fold gives misleading estimates here because training is mostly short experiments (7-10d) while test is all 14d. Only 10/100 training experiments are 14d — standard CV overrepresents short experiments and underestimates test-time error (the same distribution mismatch problem Sugiyama et al. 2007 address with importance weighting). Our simpler fix: LOO-on-14d — hold out 1 of 10 fourteen-day experiments, train on the remaining 99, repeat 10x. This directly evaluates on the target distribution.

### Model: ElasticNet

Four models compared on 13 features with LOO-14d:

| Model | LOO-14d MAE |
|---|---|
| PLS (nc=13) | 483 |
| **ElasticNet (Optuna)** | **491** |
| GP (ARD Matern 5/2, top 10) | 542 |
| Random Forest (Optuna) | 639 |
| Baseline (predict mean) | 1219 |

PLS achieves 483 but only with nc=13 (all components) — effectively unregularized at that point. ElasticNet's Ridge-dominated shrinkage (l1_ratio=0.12) provides proper regularization for n=100. The 8-point difference is not statistically significant on 10 LOO folds. Nonlinear models (RF, GP) can't generalize with n=100.

### Uncertainty Quantification

The API returns bootstrap confidence bounds (5th/95th percentile across 50 bootstrap ElasticNet refits). These capture model uncertainty (how the fit varies across resamples), not total predictive uncertainty. For proper prediction intervals, conformal prediction (Vovk et al. 2005) would give distribution-free coverage guarantees.

### Challenges

1. **Train/test mismatch**: Only 10/100 training experiments match test duration (14d). Addressed with common-window features + LOO-14d validation.
2. **Small dataset**: n=100 is too small for nonlinear methods. Linear models (PLS, ElasticNet) extrapolate better.
3. **Feature selection**: 44 features for 100 samples risks overfitting. Two-stage selection (domain + RFE) narrows to 13.

## AI Tools

Developed with Claude Code (Claude Opus 4.6). See `AI_LOG.md` for the detailed process log — what was delegated, what was corrected, and key decisions at each step.

## Key References

- Clavaud et al. (2013). "Chemometrics and in-line near infrared spectroscopic monitoring of a biopharmaceutical CHO cell culture." Talanta.
- Hutter et al. (2021). "Knowledge Transfer Across Cell Lines Using Hybrid GP Models." Biotechnology & Bioengineering. DataHow team.
- Sugiyama et al. (2007). "Covariate Shift Adaptation by Importance Weighted Cross Validation." JMLR 8:985-1005.
- Zou & Hastie (2005). "Regularization and Variable Selection via the Elastic Net." JRSSB 67(2):301-320.
