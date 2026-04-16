# Portfolio Optimization & Risk Dashboard

A Python-based tool that utilizes Scipy’s SLSQP algorithm to find the Maximum Sharpe Ratio portfolio for any given set of assets.

### Key Features

* Mean-Variance Optimization: Automated weight allocation to maximize risk-adjusted returns.
* Risk Visualization: 4-pane dashboard featuring Cumulative Growth, Correlation Heatmaps, and Drawdown charts.
* Automated Data Pipeline: Integration with yfinance for real-time market data and ticker validation.

### How to use

* Input the start date of interest
* Input the end date of interest
* Input the tickers of interest
* Optimal weights, sharpe ratios of individual stocks and the optimized portfolio will be printed, a dashboard will be shown.