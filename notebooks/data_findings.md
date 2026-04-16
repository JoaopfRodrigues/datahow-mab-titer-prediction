# Data Findings

Working notes from EDA — key observations for modeling decisions.

## 1. Data Structure

100 train / 20 test experiments. 26 columns: RowID, Exp, Time[day], 13 Z: scalars, 4 W: inputs, 6 X: observations. Z: values only at t=0 (NaN at t>0 is structural). No real missing data.

## 2. ExpDuration = Data Length = Target Day

All three always match. `n_timesteps == ExpDuration + 1` (days 0 through N). All experiments start at day 0.

## 3. Test Set — All 14d

All 20 test experiments are 14 days. Z: distributions identical to train except duration. Only 10/100 train experiments are 14d — thin training slice for test conditions.

## 4. W: Is Fully Deterministic From Z:

All 4 W: variables perfectly reconstructable from Z: scalars (zero error across 100 experiments). Step functions: W:temp/pH shift at Z:tempShift/phShift, W:FeedGlc/Gln active when Z:FeedStart <= t < Z:FeedEnd (strict `<` for FeedEnd). W: still useful for computing derived features (cumulative feed via integration).

## 5. Titer-Duration Relationship

Range 283-4823, right-skewed (median 1148, mean 1315). Strong duration dependence: mean titer 7d=811, 8d=1128, 9d=1403, 10d=1634, 14d=2380. Duration explains R²=0.38 (r=0.618). Variance increases with duration (fan-shaped scatter).

## 6. Z: Parameter Correlations

Z:ExpDuration r=+0.618 (dominant). Z:tempStart r=-0.289 (Spearman=-0.373, nonlinear component). Z:FeedStart r=-0.186. All others |r| < 0.18. DoE is well-balanced (max inter-Z correlation |r|=0.289).

## 7. IVCD — Corrected for Duration

Raw r=0.661, but IVCD correlates r=0.754 with duration. **Partial r=0.378 controlling for duration** — moderate independent contribution, not dominant. Clavaud's R²=0.95 was on fixed-duration batches — not comparable.

## 8. Gln Bimodal Split = Low VCD

Two groups: 58 depleters (Gln->0 by day 3-6) and 42 accumulators (Gln rises to 15-40). All 37 experiments with VCD_max < 20 are accumulators. Gln accumulation is a *consequence* of low cell growth — cells don't consume fast enough relative to feed. Once VCD_max is in the model, the accumulator label adds nothing.

## 9. Partial Correlations (controlling for duration)

The key analysis — reordered our entire feature ranking.

| Feature | Raw r | Partial r | Assessment |
|---------|-------|-----------|------------|
| VCD_end | +0.55 | **+0.51** | Best independent predictor |
| VCD_max | +0.59 | **+0.50** | Strong independent signal |
| int_Gln | -0.20 | **-0.43** | Suppressed — real negative signal |
| Lysed_end | +0.70 | +0.41 | Mostly duration proxy |
| int_Lac | +0.64 | +0.40 | Mixed |
| cum_FeedGlc | +0.57 | +0.40 | Mixed |
| IVCD | +0.66 | +0.38 | Mostly duration proxy |
| int_Amm | +0.55 | -0.03 | **Entirely a duration proxy** |
| cum_FeedGln | +0.54 | +0.11 | Mostly duration proxy |
| Amm_end | +0.21 | -0.20 | Sign reversal — proxy |

**Surprise**: int_Amm and cum_FeedGln nearly zero after duration control. Amm_end reverses sign.

## 10. Temperature Shift

56 experiments: shift never occurs (tempShift > duration). 15: shift on last day only. 29: meaningful post-shift data. Of those 29, 76% show slower growth post-shift. `days_post_shift = max(0, duration - tempShift)` captures this as a continuous feature.
