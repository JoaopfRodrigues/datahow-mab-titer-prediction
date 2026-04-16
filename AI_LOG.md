# AI-Assisted Development Log

Development of this solution used **Claude Code** (Claude Opus 4.6) as an AI pair-programmer. This log captures the key decisions, delegated work, and course corrections.

## Session 1 — Setup, Research, Literature Review

**Delegated**: Data exploration, PubMed literature search, project structure setup, deep research report, retrieval and review of 5 key papers.

**Key findings from research**:
- PLS regression is standard in DataHow's domain (Clavaud 2013, Narayanan 2023)
- Integrated VCD (iVCC) is a standard bioprocess feature — confirmed by our EDA (r=0.66)
- Feature engineering → tabular models is the right approach for 100 samples
- Research saved to `notebooks/00_research_notes.md`

**Key paper insights**:
- Hutter et al. (2021, DataHow team): GP models on similar data structure, derivative-based features
- Bayer & von Stosch (2020): hybrid models beat ANN/RSM, experiment-wise splitting essential
- Clavaud (2013): iVCC predicts titer at R²=0.95 on fixed-duration batches

**Mistake caught**: Wrong PDF retrieved for Von Stosch paper (DOI mismatch) — directed re-retrieval.

**My direction**: Chose PubMed-only for literature (no bioRxiv/ChEMBL), Jupyter for EDA, practical time budget.

## Session 2 — EDA Notebook (Iterative Approach)

**First attempt failed**: Claude generated the full notebook at once. Reviewed it and found nonsensical charts and low-quality analysis. **Scrapped and restarted cell-by-cell.**

This became a project rule: build notebooks one cell at a time, review each output, discuss whether it matches expectations before proceeding.

**Cell-by-cell findings**:
- W: variables are 100% deterministic from Z: (zero reconstruction error) — but still useful for derived features
- ExpDuration = data length = target day (all consistent; earlier buggy analysis had falsely suggested otherwise)

**Corrections caught during review**:
- I projected literature's "glucose depletion" onto data that actually shows glucose accumulation in 68/100 experiments
- Glutamine bimodal split more dramatic than I initially described
- Lactate follows a hump (rise then fall), not "mostly increasing"

**Biggest gap found**: No partial correlations controlling for duration. Most "top features" were just duration proxies. Adding partial correlation analysis completely reordered the feature ranking:
- int_Amm drops from r=0.55 to partial r=-0.03 (entirely a proxy)
- VCD_end (+0.51) and VCD_max (+0.50) emerge as best independent predictors
- int_Gln (-0.43) was being suppressed — real negative signal

## Session 3 — Feature Engineering + CV Strategy

**Delegated**: Implementation of `src/features.py` — 44 features in 6 groups with three interfaces (batch DataFrame, single-experiment API, shared core function).

**Key insight (user-driven)**: A 7-day experiment IS the first 7 days of a 14-day experiment. All 100 experiments share days 0-7. We should extract features from this common window. I challenged Claude's "covariate shift" framing — short experiments aren't a different distribution, they're partial observations of the same biological process.

Added 6 structural common-window features (days 0-7) to features.py, total 44 features.

**CV strategy discussion**: Standard K-Fold is misleading here — 70% of training data is 7-9d, test is 100% 14d. Reviewed 6 strategies and literature (Sugiyama 2007 IWCV, sklearn LODO). Decided on LODO + LOO-14d.

## Session 4 — Modeling + Final Model Selection

**Delegated**: Model training with feature selection upfront, then comparison on selected features.

**Feature selection** (two-stage):
1. Domain pruning (44→35): removed 9 features identified in EDA as pure duration proxies, exact linear combinations, or redundant
2. RFE with ElasticNetCV (35→13): recursive elimination using cross-validated ElasticNet. Performance improves steadily to ~13 features, then levels off.

**Model comparison on 13 features with LOO-14d**:
- Duration mismatch confirmed empirically: 14d MAE is 3-5x worse than overall for every model
- First Optuna run (overall CV objective) was counterproductive — optimized for short experiments
- Second run (LOO-14d objective) found more conservative, robust hyperparameters
- PLS achieves MAE=483 but only with nc=13 (all components — no dimensionality reduction)
- ElasticNet achieves MAE=491 with Ridge-dominated shrinkage (l1_ratio=0.12)
- GP and RF significantly worse (542, 639) — n=100 too small for nonlinear models

**PLS → ElasticNet reversal**: Initially leaned toward PLS (lower MAE), but realized PLS at nc=p performs no compression — it's effectively unregularized. ElasticNet's shrinkage is more appropriate. The 8-point difference is not statistically significant on 10 LOO folds.

**My direction**: Insisted on empirical validation (not assumptions), directed LOO-14d objective after overall-CV failed, added GP from Hutter 2021, required justified HP ranges, caught that feature selection should happen before models (not after).

## Session 5 — API, Tests, Docker

**Delegated**: REST API (`src/api.py` — FastAPI + Pydantic DTOs), test suite, Dockerfile.

**API design**:
- NaN/Inf validation on input (Python's json.loads accepts NaN tokens despite RFC 8259)
- Timestamp monotonicity validation (reversed timestamps produce silently wrong features)
- Post-prediction output validation (guard against model overflow producing non-finite values)
- Custom `RequestValidationError` handler maps Pydantic 422s to 400 per spec, with sanitized error messages

**Test suite**: 35 tests covering features, API, and model serialization — including edge cases (negative VCD, NaN injection, model unavailable, non-finite output).

**Dockerfile**: Multi-stage build, non-root user, minimal attack surface.

## Session 6 — Bootstrap UQ + Final Documentation

**Delegated**: Bootstrap model training (50 resamples of ElasticNet), API integration for confidence bounds.

**Key decision**: Labeled the bounds as "confidence bounds" (model uncertainty), not "prediction intervals." Bootstrap percentile intervals only capture epistemic uncertainty (how the fit varies across resamples), not aleatoric uncertainty (residual noise). True prediction intervals require adding a residual component (Stine 1985, Efron & Tibshirani 1993). Honest labeling avoids overstating what the intervals measure.

**Limitation acknowledged**: For proper UQ, a GP posterior (as in Hutter et al. 2021) or conformal prediction (Vovk et al. 2005) would give distribution-free coverage guarantees. The bootstrap confidence bounds are a partial solution — they show the model's sensitivity to the training sample but understate total predictive uncertainty.

**Final cross-check**: Re-verified all notebook annotations against actual cell outputs. Updated README and AI_LOG with final model details.

## Summary: Human vs. AI Contributions

**My key decisions**:
1. Cell-by-cell notebook workflow
2. Partial correlation analysis (caught that most "top features" were just duration proxies)
3. Common-window feature insight (challenged the "covariate shift" framing)
4. CV strategy: pushed from KFold to LODO/LOO-14d
5. Demanded justified hyperparameter ranges (not defaults)
6. Restructured notebook to do feature selection before models (not after)
7. Insisted imports match discovery order (no forward-looking imports)
8. Reversed PLS selection: PLS only wins at nc=p (no compression) — chose ElasticNet for proper regularization

**What Claude did well**: Literature search, parallel research agents, clean feature engineering, API + tests matching the OpenAPI spec, systematic annotation verification against actual outputs.

**What needed correction**: Initial notebook generation (scrapped), "covariate shift" framing (too narrow), plain KFold despite having researched LODO (inconsistency), narrow hyperparameter search, annotations written from memory without verifying outputs, forward-looking imports that broke the exploration narrative.
