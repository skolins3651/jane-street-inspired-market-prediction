# Project Scope

## Original Competition

The [Jane Street Market Prediction competition](https://www.kaggle.com/competitions/jane-street-market-prediction/overview) asked participants to decide whether to accept or pass on anonymized trading opportunities. Each observation included a weight, anonymized numerical features, and several return-related response variables. Models were evaluated using a utility score designed to reward profitable and reasonably consistent decisions over time.

The original competition dataset is no longer available through Kaggle. This project therefore does not attempt to reproduce Jane Street’s proprietary observations or claim that public equity data is equivalent to the original data.

## Public-Data Adaptation

This project reconstructs the competition’s general decision framework using publicly available daily market data downloaded through `yfinance`. The initial dataset contains adjusted daily open, high, low, close, and volume observations for 30 liquid U.S. equities and the SPY market benchmark from 2016 through 2025.

Each future modeling observation will represent one stock on one trading date. Information available on or before that date will be used to estimate a future return-related outcome and ultimately decide whether to take or pass on the opportunity.

## What the Project Preserves

The adaptation preserves several important ideas from the original challenge:

* financial prediction using time-ordered data;
* multiple market observations across each trading day;
* an execute-or-pass decision;
* observation weights that can represent differing economic importance;
* chronological model evaluation rather than random data splitting;
* a utility-oriented evaluation measure rather than accuracy alone.

## Important Differences and Limitations

The replacement dataset uses named public equities rather than anonymized proprietary trading opportunities. It has daily rather than intraday observations, interpretable market features rather than anonymous features, and targets that must be constructed from future public-market returns.

The fixed stock universe was selected using present-day knowledge and is not a point-in-time reconstruction of a historical index. Results may therefore be affected by survivorship and selection bias. The data source is suitable for an educational portfolio project but should not be treated as institutional-grade market data.

These differences will be documented throughout the project so that the final results are presented as a Jane Street–inspired public-data study rather than a reproduction of the original competition.
