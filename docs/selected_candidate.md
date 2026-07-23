# Selected Candidate

This document selects the strongest candidate from the planned improvement experiments.

The selected candidate is not treated as proof of a profitable trading strategy. It is the best-performing version under the project’s current validation framework and will be used as the main comparison point for any scoped extension experiment.

## Selection Rule

The primary selection criterion is validation liquidity-weighted utility.

Equal-weight utility, action rate, mean response taken, and other companion diagnostics are used as sanity checks.

The test split is not used to select the candidate.

## Candidate Comparison

| Model / Rule | Equal-Weight Utility | Liquidity-Weight Utility | Action Rate | Mean Response Taken |
|---|---:|---:|---:|---:|
| Original logistic, threshold 0.50 | 0.000000 | 2.870843 | 0.281836 | -0.000298 |
| Tuned-threshold logistic, threshold 0.48 | 1.289191 | 6.283775 | 0.876114 | 0.000145 |
| Expanded-feature logistic, threshold 0.50 | 0.175228 | 2.670977 | 0.291550 | 0.000127 |
| Gradient boosting, threshold 0.50 | 0.000000 | 0.468004 | 0.167265 | -0.000208 |
| Utility-aware top-20/day rule | 0.132720 | 4.248294 | 0.666667 | 0.000057 |

## Selected Candidate

The selected candidate is:

```text
Logistic regression with probability threshold 0.48
```

This candidate produced the highest validation liquidity-weighted utility among the planned improvement experiments.

It also improved equal-weight utility relative to the original logistic-regression baseline, which makes the result more credible than an improvement that appears only under liquidity weighting.

## Interpretation

The selected candidate is best understood as an improved action rule for a weak but somewhat useful baseline model.

The original logistic-regression model used the default classification threshold of `0.50`. Lowering the threshold to `0.48` caused the model to take a much broader set of opportunities and improved validation utility.

This suggests that the model’s probability estimates contain some useful information, but the default classification threshold was poorly matched to the project’s execute-or-pass utility framework.

## Caveats

The selected candidate is not highly selective.

Its validation action rate is approximately `87.6%`, meaning it takes most available opportunities. This makes it closer to a softened always-take strategy than a narrow high-conviction trading rule.

That caveat matters. A stronger model would ideally improve utility while taking a smaller and more clearly differentiated set of opportunities.

The selected candidate should therefore be treated as the best current candidate, not as a fully satisfying endpoint.

## Scoped Extension Context

Because the planned improvement experiments produced only mixed results, the project will run one scoped extension experiment before finalizing the modeling story.

The selected candidate will serve as the benchmark for that extension.

The extension should be chosen based on the observed failure mode: the current models appear to contain broad weak signal, but they do not rank the best opportunities sharply enough to support a highly selective action rule.

## Later Update

After the planned improvement experiments, the project tested one scoped extension: a cross-sectional rank-regression model.

That model superseded the threshold-tuned logistic candidate. The threshold-tuned logistic model remains the strongest candidate from the planned improvement menu, but the final selected modeling candidate is the cross-sectional rank-regression model with top-8-per-day selection.

See [`modeling_experiment_summary.md`](modeling_experiment_summary.md) for the final consolidated modeling comparison.
