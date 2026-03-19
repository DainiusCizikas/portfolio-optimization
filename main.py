import numpy as np
import yfinance as yf
import scipy.optimize as sco
#import pandas as pd
#import matplotlib.pyplot as plt
#from tabulate import tabulate

def get_inputs():
    target_tickers = [t.strip() for t in input("Input the comma delimited ticker symbols:\n").split(',')]
    start_date = '2024-01-01' #input("Input the start date (yyyy-mm-dd):\n")
    end_date = '2024-12-31' #input("Input the end date (yyyy-mm-dd):\n")

    #tickers = []
    #for i, ticker in enumerate(target_tickers):
    #    ticker = yf.Ticker(target_tickers[i])
    #    tickers.append(ticker)

    #Create ticker validation and remove the invalid ones - additional assignment
    #Check the validity of the dates - additional assignment
    #Make the target_tickers with .strip() - additional assignment

    return target_tickers, start_date, end_date

def calculate_returns(tickers, start, end):
    ticker_list = tickers
    start_date = start
    end_date = end

    df_data = yf.download(ticker_list, start_date, end_date, auto_adjust=False)
    adj_close_prices = df_data['Adj Close']
    log_returns = np.log(adj_close_prices / adj_close_prices.shift(1)).dropna()

    return log_returns

def create_covariance_matrix(df_returns):

    cov_matrix = df_returns.cov()
    annualized_cov_matrix = cov_matrix * 252
    #covariance_matrix.index.name = None
    #covariance_matrix.columns.name = None

    return annualized_cov_matrix

def portfolio_performance(weights, average_returns, cov_matrix, risk_free_rate):

    returns = weights @ average_returns
    volatility = np.sqrt(weights @ cov_matrix @ weights)
    sharpe = (returns - risk_free_rate) / volatility

    return returns, volatility, sharpe

def objective_function(weights, mean_returns, cov_matrix, risk_free_rate):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def mean_variance_optimization(covariance, returns):

    annual_average_return = returns.mean() * 252
    number_assets = len(annual_average_return)
    initial_guess = np.ones(number_assets) / number_assets

    bounds = tuple((0, 1) for _ in range(number_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    extra_data = (annual_average_return, covariance, 0.04)

    result = sco.minimize(fun=objective_function, x0=initial_guess, args = extra_data, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

def in_sample_back_test():
    print("In sample Back Test")

def create_visualization():
    print("Create Visualization")


if __name__ == '__main__':
    input_tickers, input_start_date, input_end_date = get_inputs() #Add type hints?

    adj_log_returns = calculate_returns(input_tickers, input_start_date, input_end_date) #Add type hints?

    covariance_matrix = create_covariance_matrix(adj_log_returns)

    optimized_weights = mean_variance_optimization(covariance_matrix, adj_log_returns)

    for ticker, weight in zip(adj_log_returns.columns, optimized_weights):
        print(f"{ticker}: {weight:.2%}")
    in_sample_back_test()
    create_visualization()