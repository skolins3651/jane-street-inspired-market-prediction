# Improvement Hypotheses

This document summarizes the initial baseline results and turns them into concrete hypotheses for controlled improvement experiments.

The goal is not to search endlessly for a better score. The goal is to identify a small number of plausible improvements, test them one at a time, and document what each experiment teaches about the prediction problem.

## Baseline Lessons

The initial modeling framework is complete: the project has a prediction target, feature set, chronological splits, baseline strategies, a simple machine-learning model, and an adapted utility score.

The baseline results suggest that next-day SPY-relative stock outperformance is difficult to predict with the initial feature set.

Several important lessons emerged:

* always-pass is useful as a zero-action reference point;
* always-take is a strong simple benchmark because the target rate is close to 50%;
* the 20-day momentum rule is more selective than always-take but does not clearly dominate it;
* logistic regression produces valid probabilities and actions, but its performance is mixed;
* liquidity-weighted utility and equal-weight utility can lead to different interpretations;
* test results should be treated as final-reporting context, not as a tuning guide.

## Main Weaknesses to Investigate

The initial results point to several possible weaknesses.

### Default action threshold may be too crude

The logistic-regression baseline uses a probability threshold of `0.5`.

That threshold is natural for binary classification, but it may not be natural for a trading-style execute-or-pass problem. A model may need to be more selective in order to produce better utility.

### Feature set may be too limited

The first feature set is intentionally simple. It contains recent returns, volatility, range, gap, volume, dollar volume, and relative volume features.

These features may be too raw to capture useful relationships. Additional leakage-safe features may better represent market-relative movement, short-term reversal, cross-sectional rank, or recent trend behavior.

### Linear model may be too restrictive

Logistic regression is useful as a transparent baseline, but it can only model linear relationships after feature scaling.

The relationship between recent returns, volatility, range, volume, and future outperformance may be nonlinear or interaction-driven. A simple tree-based model may capture some of those relationships more effectively.

### Classification metrics and utility are not the same objective

Accuracy, precision, and recall are useful diagnostics, but the project’s main scoring framework is based on weighted selected response and daily consistency.

A model that is not impressive by classification accuracy may still be useful if its selected actions produce better utility. Conversely, a model with decent classification metrics may still be unattractive if its selected responses are economically weak.

## Improvement Hypotheses

The next improvement experiments will test the following hypotheses.

### Hypothesis A — Threshold tuning may improve action quality

Changing the logistic-regression action threshold may improve validation utility by making the model more or less selective.

This tests whether the baseline model’s probabilities contain useful ranking information even if the default `0.5` threshold is not optimal.

### Hypothesis B — Additional features may improve signal quality

Adding a small number of leakage-safe features may improve the model’s ability to distinguish next-day outperformers from underperformers.

This tests whether the first feature set was too limited or too raw.

### Hypothesis C — A tree-based model may capture nonlinear structure

A simple tree-based classifier may improve results by capturing interactions among returns, volatility, range, and volume features.

This tests whether the logistic-regression baseline is limited by its linear structure.

### Hypothesis D — Utility-aware action rules may better match the project objective

Choosing actions using validation utility rather than classification intuition alone may produce more relevant trading-style behavior.

This tests whether the action rule should be aligned more directly with the adapted utility score.

## Evaluation Discipline

Each experiment should change one major component at a time.

Experiments should be compared primarily using validation liquidity-weighted utility. Equal-weight utility and companion diagnostics should be used to check whether an apparent improvement is robust or pathological.

The test split should not be used to choose features, thresholds, model families, or action rules.

## Contingency Plan

If none of the planned experiments materially improves validation liquidity-weighted utility, the project will not continue open-ended optimization.

Instead, it will choose either:

* the best controlled result from the planned experiments; or
* one additional scoped extension experiment based on the observed failure mode.

Examples of scoped extension experiments include a longer prediction horizon, a more selective action rule, or a liquidity-aware threshold.

The purpose of the contingency plan is to preserve experimental discipline while still leaving room for one thoughtful adjustment if the planned menu does not produce a meaningful improvement.