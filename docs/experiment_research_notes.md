# Experiment Research Notes

This document explains the rationale behind the planned improvement experiments for the Jane Street–inspired market prediction project.

The purpose of these notes is to connect modeling choices to specific weaknesses observed in the baseline results. The project is not trying every possible model or feature. It is testing a small set of controlled changes that can be explained clearly.

## Guiding Principle

Each improvement experiment should answer one focused question.

A useful experiment changes one major part of the framework while keeping the rest of the pipeline comparable. This makes it easier to tell whether a result changed because of the experiment itself or because too many things changed at once.

The core evaluation standard is validation liquidity-weighted utility, supported by equal-weight utility and companion diagnostics.

## Probability Threshold Tuning

The logistic-regression baseline produces predicted probabilities for `target_1d`.

The default action rule is:

```text
action = 1 if predicted_probability >= 0.5 else 0
```

This threshold is natural for binary classification, but it may not be ideal for an execute-or-pass trading-style problem.

A model may rank opportunities reasonably well even if the default threshold is too aggressive or too conservative. Threshold tuning tests whether changing the action cutoff improves utility without changing the model or feature set.

Possible threshold values include:

```text
0.40
0.45
0.50
0.55
0.60
```

A lower threshold takes more opportunities. A higher threshold is more selective.

The main risk is overfitting the threshold to the validation split. To reduce that risk, threshold tuning should use a small, predefined grid rather than searching hundreds of possible values.

## Feature Expansion

The initial feature set is intentionally simple. It includes recent returns, volatility, intraday range, overnight gap, volume, dollar volume, and relative volume.

Feature expansion tests whether the baseline model was limited by the information it received.

New features should be leakage-safe, meaning they use only information available by the close of the prediction date.

Reasonable feature families include:

* market-relative features;
* rolling average return features;
* volatility comparison features;
* short-term reversal features;
* cross-sectional rank features.

Examples include:

```text
return_1d_minus_spy
return_5d_minus_spy
rolling_mean_return_5d
rolling_mean_return_20d
volatility_ratio_5d_20d
return_1d_reversal
cross_sectional_return_20d_rank
```

The purpose is not to add as many features as possible. The purpose is to add a small number of interpretable features that represent plausible market behavior.

## Tree-Based Modeling

Logistic regression is a useful first machine-learning baseline because it is simple and interpretable.

However, logistic regression is limited because it models a linear relationship between standardized features and the log-odds of the target.

A tree-based model can capture nonlinear relationships and interactions. For example, a return signal may matter differently in high-volatility and low-volatility environments, or volume features may matter only when paired with recent price movement.

A simple tree-based classifier can test whether the initial baseline was limited by model structure rather than by the feature set alone.

Candidate model families include:

* random forest;
* histogram gradient boosting;
* another simple gradient-boosted tree classifier.

The model should remain intentionally constrained. The goal is not exhaustive hyperparameter optimization. The goal is to test whether a nonlinear model provides a meaningful improvement over logistic regression.

## Utility-Aware Action Rules

Classification metrics and trading utility measure different things.

A model can have mediocre accuracy but still be useful if the actions it takes have positive weighted responses. Conversely, a model can have decent classification accuracy but poor utility if it takes too many low-quality opportunities.

A utility-aware action rule chooses actions using validation liquidity-weighted utility rather than classification accuracy alone.

This may involve selecting a probability threshold that maximizes validation utility, or applying an additional rule that restricts actions to more favorable conditions.

This experiment should be interpreted carefully because utility can be sensitive to outliers and weighting schemes. Any apparent improvement should be checked against:

* equal-weight utility;
* total profit;
* daily-profit volatility;
* action rate;
* mean response taken;
* concentration in a small number of dates.

## Why These Experiments Come First

The planned experiments are ordered from simplest to more complex.

Threshold tuning comes first because it changes only the action rule. It asks whether the existing logistic-regression probabilities are being used effectively.

Feature expansion comes next because it tests whether the model needs better inputs.

Tree-based modeling follows because it tests whether nonlinear structure matters.

Utility-aware action rules come after the basic pieces are understood because they optimize more directly around the project’s final score.

## What Would Count as a Meaningful Improvement

A strong improvement should satisfy most of the following:

* higher validation liquidity-weighted utility than the current logistic-regression baseline;
* reasonable equal-weight utility;
* positive or improved total profit;
* non-pathological action rate;
* daily profit not dominated by one or two dates;
* clear explanation of what changed;
* reproducible implementation.

A result is weaker if it improves only one score while making the surrounding diagnostics worse.

## What This Project Is Not Doing

The project is not performing an unrestricted search over all possible models, thresholds, features, or scoring rules.

The following are intentionally avoided for now:

* large hyperparameter sweeps;
* repeated tuning on the test split;
* adding features without a rationale;
* changing target definitions mid-experiment;
* comparing models without the same chronological split structure;
* optimizing utility without checking diagnostics.

These restrictions help preserve the credibility of the experiment results.