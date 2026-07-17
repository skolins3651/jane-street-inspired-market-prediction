# Threshold Tuning Results

This document summarizes the probability-threshold tuning experiment for the logistic-regression baseline.

## Purpose

The original logistic-regression baseline used the default action rule:

```text
action = 1 if predicted_probability >= 0.50 else 0
```

That threshold is natural for binary classification, but it may not be ideal for an execute-or-pass trading-style problem.

This experiment tested whether changing the action threshold could improve validation utility without changing the model, features, prediction target, or split structure.

## Method

The experiment evaluated a predefined grid of probability thresholds:

```text
0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55
```

Each threshold was evaluated on the validation split using both equal-weight and liquidity-weight adapted utility.

The selected threshold was chosen by validation liquidity-weighted utility.

## Selected Threshold

The best threshold by validation liquidity-weighted utility was `0.48`.

| Threshold | Weighting Scheme | Utility | Total Profit | c_stat | Action Rate | Mean Response Taken |
|---:|---|---:|---:|---:|---:|---:|
| 0.48 | liquidity-weighted | 6.283775 | 6.120572 | 1.026665 | 0.876114 | 0.000145 |
| 0.48 | equal-weight | 1.289191 | 1.909195 | 0.675253 | 0.876114 | 0.000145 |

## Comparison to Original Threshold

The original `0.50` threshold had positive liquidity-weighted utility but negative equal-weight total profit.

| Threshold | Weighting Scheme | Utility | Total Profit | c_stat | Action Rate | Mean Response Taken |
|---:|---|---:|---:|---:|---:|---:|
| 0.50 | liquidity-weighted | 2.870843 | 3.435401 | 0.835665 | 0.281836 | -0.000298 |
| 0.50 | equal-weight | 0.000000 | -1.263731 | -0.740568 | 0.281836 | -0.000298 |

Lowering the threshold from `0.50` to `0.48` improved validation liquidity-weighted utility and also produced positive equal-weight utility.

## Interpretation

The result suggests that the logistic-regression probabilities contain some useful ranking information, but the default `0.50` threshold was too selective for this dataset and scoring framework.

The selected threshold of `0.48` takes a much larger fraction of opportunities than the original threshold. Its validation action rate is approximately `87.6%`, compared with approximately `28.2%` for the original `0.50` threshold.

This means the tuned model is not acting as a highly selective trading rule. It is closer to a softened always-take strategy that excludes the lowest-probability opportunities.

That is still a useful result. The experiment shows that action-rule choice matters, and that a small threshold change can materially affect utility. However, the high action rate should be considered when comparing this strategy against always-take and future experiments.

## Main Lesson

Threshold tuning improved the logistic-regression baseline without changing the model or feature set.

The improvement is encouraging because it appears under both liquidity-weighted and equal-weight utility. The result is also a reminder that classification defaults do not necessarily match trading-style objectives.

Future experiments should continue to evaluate both utility and companion diagnostics, especially action rate, total profit, and equal-weight behavior.

## Presentation Note

This experiment may benefit from a later visualization of threshold versus utility and threshold versus action rate.

Those plots are not necessary for the core implementation, but they may help communicate the result in the final project presentation.