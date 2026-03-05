import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

def get_inputs():
    target_tickers = input("Input the comma delimited ticker symbols:\n").split(',')
    start_date = input("Input the start date (yyyy-mm-dd):\n")
    end_date = input("Input the end date (yyyy-mm-dd):\n")

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

    print(type(ticker_list))
    print(type(start_date))
    print(type(end_date))
    df_data = yf.download(ticker_list, start_date, end_date, auto_adjust=False)

    adj_close_prices = df_data['Adj Close']

    log_returns = np.log(adj_close_prices / adj_close_prices.shift(1))

    return log_returns

def create_covariance_matrix():
    print("Create Covariance Matrix")

def mean_variance_optimization():
    print("Mean Variance Optimization")

def in_sample_back_test():
    print("In sample Back Test")

def create_visualization():
    print("Create Visualization")


if __name__ == '__main__':
    input_tickers, input_start_date, input_end_date = get_inputs() #Add type hints?

    adj_log_returns = calculate_returns(input_tickers, input_start_date, input_end_date) #Add type hints?

    create_covariance_matrix()
    mean_variance_optimization()
    in_sample_back_test()
    create_visualization()