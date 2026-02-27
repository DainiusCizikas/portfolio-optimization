import yfinance as yf
import pandas as pd

def get_inputs():
    inputs = input("Input the comma delimited ticker symbols:\n").split()
    start_date = input("Input the start date (yyyy-mm-dd):\n")
    end_date = input("Input the end date (yyyy-mm-dd):\n")
    tickers = []

    for i, ticker in enumerate(inputs):
        ticker = yf.Ticker(inputs[i])
        tickers.append(ticker)

    #Create ticker validation and remove the invalid ones - additional assignment

    return tickers, start_date, end_date


def calculate_returns(tickers, start_date, end_date):
    print("Calculate Returns")

def create_covariance_matrix():
    print("Create Covariance Matrix")

def mean_variance_optimization():
    print("Mean Variance Optimization")

def in_sample_back_test():
    print("In sample Back Test")

def create_visualization():
    print("Create Visualization")



if __name__ == '__main__':
    get_inputs()
    calculate_returns()
    create_covariance_matrix()
    mean_variance_optimization()
    in_sample_back_test()
    create_visualization()