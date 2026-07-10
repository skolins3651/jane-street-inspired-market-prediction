# Initial Baseline Results

This document records the first baseline results for the Jane Street–inspired market prediction project.

The goal of these baselines is not to produce a profitable trading system. The goal is to establish a simple, reproducible evaluation framework for the one-day SPY-relative outperformance problem.

## Modeling Dataset

The baseline models use the split-labeled feature dataset:

```text
data/processed/modeling_dataset_with_features_and_splits_1d.parquet
```

The dataset contains one row per stock-date observation. SPY is used as the benchmark and is not included as a tradable prediction row.

## Splits

The initial chronological splits are:

| Split | Date Range | Rows |
|---|---:|---:|
| Train | 2016-02-02 to 2021-12-31 | 44,730 |
| Validation | 2022-01-03 to 2023-12-29 | 15,030 |
| Test | 2024-01-02 to 2025-12-30 | 15,030 |

No random shuffling is used.

## Baselines Compared

The initial baseline comparison includes:

| Baseline | Description |
|---|---|
| Always-pass | Never takes the long stock opportunity. |
| Always-take | Takes every long stock opportunity. |
| 20-day momentum rule | Takes the opportunity when `return_20d > 0`. |
| Logistic regression | Uses the engineered feature set to predict `target_1d`, then takes the opportunity when predicted probability is at least `0.5`. |

## Summary Results

The table below reports the most digestible baseline comparison using equal-weight evaluation on the validation and test splits.

| Split | Baseline | Accuracy | Precision | Recall | Action Rate | Mean Response Taken |
|---|---|---:|---:|---:|---:|---:|
| validation | Always-pass | 0.502661 | undefined | 0.000000 | 0.000000 | undefined |
| validation | Always-take | 0.497339 | 0.497339 | 1.000000 | 1.000000 | 0.000103 |
| validation | 20-day momentum rule | 0.500998 | 0.498394 | 0.518930 | 0.517831 | 0.000133 |
| validation | Logistic regression | 0.498802 | 0.493154 | 0.279465 | 0.281836 | -0.000298 |
| test | Always-pass | 0.508250 | undefined | 0.000000 | 0.000000 | undefined |
| test | Always-take | 0.491750 | 0.491750 | 1.000000 | 1.000000 | 0.000009 |
| test | 20-day momentum rule | 0.497405 | 0.490860 | 0.592207 | 0.593280 | -0.000025 |
| test | Logistic regression | 0.499534 | 0.478630 | 0.198485 | 0.203925 | -0.000496 |

These results should be interpreted as baseline diagnostics rather than a trading claim. The validation and test results are close to random classification performance, which suggests that the first feature set does not yet provide strong evidence of reliable next-day predictive power.

## Main Takeaways

The baseline results suggest that the initial one-day classification problem is difficult.

The always-take baseline produces accuracy close to the underlying target rate, which is near 50% across the train, validation, and test splits. This is expected because the target is whether a stock outperforms SPY over the next trading day.

The 20-day momentum rule is more selective than always-take, but it does not clearly establish strong predictive power in the validation and test periods.

The logistic regression baseline trains successfully and produces valid probability estimates and action decisions. However, its validation and test results are close to the non-ML baselines rather than clearly superior. This is a useful result: the project now has a functioning ML baseline, but the first feature set does not yet provide strong evidence of reliable next-day predictive power.

## Logistic Regression Coefficients

The logistic regression coefficient ranking provides a first diagnostic view of which standardized features the model uses most heavily. These coefficients should not be overinterpreted as stable financial relationships. They are best treated as a debugging and interpretability check for the initial baseline model.

## Interpretation

The initial baseline results create a working benchmark for future modeling work.

The most important conclusions are:

* the data pipeline now supports end-to-end modeling;
* the target, weights, splits, features, actions, and metrics are all connected;
* simple baselines produce plausible but not impressive results;
* future improvements should be judged against these baselines rather than against intuition alone.

## Deferred Improvements

Future extensions may explore:

* additional features;
* alternative prediction horizons;
* probability threshold tuning;
* stronger regularization or model selection;
* tree-based models;
* transaction-cost-aware evaluation;
* a Jane Street-style utility score.