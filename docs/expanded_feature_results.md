# Expanded Feature Results

This document summarizes the expanded-feature logistic-regression experiment.

## Purpose

The initial logistic-regression baseline used a small, intentionally simple feature set.

This experiment tested whether adding a larger set of leakage-safe features would improve model performance while keeping the model family, prediction target, chronological split structure, and default `0.50` action threshold fixed.

The goal was to isolate the effect of feature expansion.

## Method

The expanded dataset adds 32 new features to the original 9-feature set, producing 41 total model features.

The added features cover several categories:

* market-relative returns;
* additional trend features;
* volatility and range features;
* volume and liquidity features;
* cross-sectional rank features.

The expanded features require 60-day rolling windows, which drops 1,200 early training rows. Validation and test row counts remain unchanged.

## Validation Results

| Model / Rule | Equal-Weight Utility | Liquidity-Weight Utility | Action Rate | Mean Response Taken |
|---|---:|---:|---:|---:|
| Original logistic, threshold 0.50 | 0.000000 | 2.870843 | 0.281836 | -0.000298 |
| Expanded-feature logistic, threshold 0.50 | 0.175228 | 2.670977 | 0.291550 | 0.000127 |
| Tuned-threshold logistic, threshold 0.48 | 1.289191 | 6.283775 | 0.876114 | 0.000145 |

## Expanded-Feature Model Details

On the validation split, the expanded-feature logistic model had:

| Metric | Value |
|---|---:|
| Accuracy | 0.503194 |
| Precision | 0.500913 |
| Recall | 0.293645 |
| Action rate | 0.291550 |
| Equal-weight utility | 0.175228 |
| Liquidity-weight utility | 2.670977 |
| Equal-weight total profit | 0.556750 |
| Liquidity-weight total profit | 3.293026 |
| Mean response taken | 0.000127 |

## Interpretation

The expanded-feature model produced a modest improvement in some diagnostics, but not in the main validation score.

Compared with the original logistic-regression baseline at the same `0.50` threshold, the expanded-feature model improved validation accuracy, precision, action rate, mean response taken, and equal-weight utility. Most importantly, equal-weight total profit changed from negative to positive.

However, the expanded-feature model did not improve validation liquidity-weighted utility. Its liquidity-weighted utility was slightly lower than the original logistic-regression baseline.

This suggests that the added features improved broad average behavior, but did not improve the model's ability to select the most useful liquidity-weighted opportunities.

The result also shows that adding features alone is not guaranteed to improve the project’s main score. Feature expansion may become more useful in a later combined experiment, but by itself it did not improve the main validation score.

## Coefficient Notes

The largest standardized coefficients included:

| Feature | Coefficient |
|---|---:|
| rolling_mean_return_20d | -0.121823 |
| range_5d_mean | 0.070422 |
| return_5d_rank | 0.061725 |
| spy_return_5d | 0.056156 |
| range_20d_mean | -0.054389 |
| log_dollar_volume | -0.050286 |
| relative_volume_20d | 0.049658 |
| return_3d | -0.048064 |

The coefficients suggest that the model is using a mix of trend, range, market movement, liquidity, and cross-sectional rank information. The signs should not be overinterpreted as causal effects, but the coefficient table is useful for checking that the expanded model is responding to plausible feature families.

## Main Lesson

Feature expansion was useful, but not sufficient.

The experiment improved some validation diagnostics and made the default-threshold logistic model less weak under equal-weight evaluation. However, it did not improve the main validation liquidity-weighted utility score.

This result suggests that adding more leakage-safe features helped broad average behavior, but did not by itself improve the project’s main score.

The next controlled experiment should test model-family change separately, using the original baseline feature set. That keeps the comparison clean: logistic regression versus a tree-based model, without also changing the feature set.

A later combined experiment could use both expanded features and a nonlinear model, but that should be treated as a separate extension rather than folded into the model-family test.