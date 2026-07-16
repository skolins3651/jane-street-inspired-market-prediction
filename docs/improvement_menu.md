# Improvement Menu

This document defines a small set of controlled improvement experiments for the Jane Street–inspired market prediction project.

The goal is not to search endlessly for the best possible trading algorithm. The goal is to improve the baseline framework in a disciplined, interpretable way while preserving a clear record of what changed and why.

## Purpose

The project now has a complete baseline modeling and scoring framework:

* a one-day SPY-relative prediction target;
* leakage-safe engineered features;
* chronological train, validation, and test splits;
* non-ML baseline strategies;
* a logistic-regression baseline;
* an adapted Jane Street-style utility score;
* equal-weight and liquidity-weight evaluation.

The next step is to test a limited number of improvement experiments. Each experiment should change one major component of the baseline framework so that any change in results can be interpreted clearly.

## Current Baseline Lessons

The initial baseline results suggest that the one-day prediction problem is difficult.

The logistic-regression baseline produces valid predictions and has the best validation liquidity-weighted utility among the initial baselines. However, its equal-weight behavior is weak, and it does not clearly dominate simpler strategies on the test split.

The equal-weight and liquidity-weight comparisons show that a strategy can look better under one weighting scheme than the other. Future experiments should therefore report both weighting schemes, even though validation liquidity-weighted utility is the main comparison score.

## Evaluation Rules

Improvement experiments should follow these rules:

* use the training split to fit models;
* use the validation split to compare experiments and choose settings;
* reserve the test split for final reporting;
* change one major component at a time;
* compare against always-pass, always-take, the 20-day momentum rule, and logistic regression;
* report liquidity-weighted utility, equal-weight utility, total profit, daily-profit volatility, action rate, and mean response taken;
* avoid interpreting a higher utility score as meaningful if it comes from pathological behavior.

Examples of pathological behavior include:

* extremely low action rate without a clear reason;
* utility driven by one or two outlier dates;
* positive liquidity-weighted utility with strongly negative equal-weight behavior;
* improvements caused by accidental leakage;
* repeated tuning on the test split.

## Menu Item A — Probability Threshold Tuning

The logistic-regression baseline currently takes an opportunity when its predicted probability is at least `0.5`.

This experiment tests whether the same model performs better when the action threshold is changed.

The hypothesis is that a more selective threshold may improve utility by taking fewer, higher-confidence opportunities.

This experiment changes only the action rule. It does not change the feature set or model family.

## Menu Item B — Feature Expansion

The initial feature set is intentionally simple. This experiment adds a small number of new leakage-safe features built from information available by the prediction date.

Possible additions include:

* market-relative recent return;
* rolling mean return;
* rolling volatility ratio;
* short-term reversal feature;
* cross-sectional return rank.

The hypothesis is that the initial model may be limited by weak or overly raw features.

This experiment changes the feature set while keeping the modeling and evaluation framework otherwise comparable.

## Menu Item C — Tree-Based Model

The logistic-regression baseline is linear after feature scaling. This experiment tests a simple tree-based model that can capture nonlinear relationships and feature interactions.

Candidate models include:

* random forest;
* histogram gradient boosting;
* another simple gradient-boosted tree classifier.

The hypothesis is that nonlinear interactions among returns, volatility, range, and volume features may contain predictive information that logistic regression cannot capture.

This experiment changes the model family while keeping the prediction target, split structure, and utility scoring framework fixed.

## Menu Item D — Utility-Aware Action Rule

Classification accuracy and utility are not the same objective. A model can have mediocre accuracy but still produce useful actions if it selects opportunities with favorable weighted responses.

This experiment chooses an action threshold or action rule using validation liquidity-weighted utility rather than classification accuracy alone.

The hypothesis is that optimizing the decision rule around the project’s actual utility score may produce more relevant trading-style behavior.

This experiment should be interpreted carefully. Equal-weight utility and companion diagnostics must be checked to make sure any improvement is not purely an artifact of liquidity weighting.

## Stop Rule

After the planned improvement experiments are complete, the project should select the best version based on validation liquidity-weighted utility, supported by equal-weight utility and companion diagnostics.

The project should then move on to final interpretation and presentation rather than continue open-ended optimization.

The purpose of this menu is to support a clear modeling story: build a baseline, identify weaknesses, test controlled improvements, compare results, and document what was learned.