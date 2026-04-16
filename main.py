import numpy as np
import yfinance as yf
import scipy.optimize as sco
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def is_valid(ticker_unvalidated): #Checking if a ticker exists
    try:
        ticker_checked = yf.Ticker(ticker_unvalidated)
        history = ticker_checked.history(period='1d')
        return not history.empty #If there is history True is returned
    except:
        return False #If the history does not exist, an error will be thrown

def get_inputs():
    start_date = input("Input the start date (yyyy-mm-dd):\n")
    end_date = input("Input the end date (yyyy-mm-dd):\n")
    target_tickers = [t.strip() for t in input("Input the comma delimited ticker symbols:\n").split(',')] #Removing extra spaces for each ticker
    invalid_tickers = []

    for ticker_unvalidated in target_tickers: #Starts the validation process and removes the bad tickers
        is_ticker_valid = is_valid(ticker_unvalidated)
        if not is_ticker_valid:
            print(f"{ticker_unvalidated} is not a valid ticker symbol. Removing {ticker_unvalidated}")
            invalid_tickers.append(ticker_unvalidated)

    target_tickers = list(set(target_tickers) - set(invalid_tickers))
    #Check the validity of the dates - additional assignment

    return target_tickers, start_date, end_date

def calculate_returns(tickers, start, end): #Calculates the returns of all assets
    ticker_list = tickers
    start_date = start
    end_date = end

    df_data = yf.download(ticker_list, start_date, end_date, auto_adjust=False)
    adj_close_prices = df_data['Adj Close'] #These prices are adjusted for stock splits and dividends
    log_returns = np.log(adj_close_prices / adj_close_prices.shift(1)).dropna() #Using the logarithmic scale makes it easier to work with numbers

    return log_returns

def create_covariance_matrix(df_returns): #Creates the covariance matrix

    cov_matrix = df_returns.cov()
    annualized_cov_matrix = cov_matrix * 252 #Usually there are 252 trading days in a year

    return annualized_cov_matrix

def portfolio_performance(weights, average_returns, cov_matrix, risk_free_rate): #Performance measures of a portfolio are calculated

    returns = weights @ average_returns
    volatility = np.sqrt(weights @ cov_matrix @ weights)
    sharpe = (returns - risk_free_rate) / volatility

    return returns, volatility, sharpe

def objective_function(weights, mean_returns, cov_matrix, risk_free_rate): #Gets the negative sharpe ratio of a portfolio for the optimizer to minimize
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2] #

def mean_variance_optimization(covariance, returns): #Performs the optimization

    annual_average_return = returns.mean() * 252
    number_assets = len(annual_average_return)
    initial_guess = np.ones(number_assets) / number_assets

    bounds = tuple((0, 1) for _ in range(number_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}) #Only allows positive weights from 0 to 1. All weights must add up to 1

    extra_data = (annual_average_return, covariance, risk_free_rate)

    result = sco.minimize(fun=objective_function, x0=initial_guess, args = extra_data, method='SLSQP', bounds=bounds, constraints=constraints) #Starts the main optimization process

    return result.x

def in_sample_back_test(returns, weights): #Calculates the optimized individual and portfolio returns
    returns = returns.cumsum()
    returns = np.exp(returns) - 1
    portfolio_return =  returns @ weights

    return returns, portfolio_return

def create_visualization(returns, portfolio_returns, asset_weights): #Creates a dashboard of visualizations
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    for ticker in returns.columns:
        ax[0, 0].plot(returns.index, returns[ticker], label=f"{ticker} Only")

    ax[0, 0].plot(returns.index, portfolio_returns, label="Optimized Portfolio", linewidth=2, color='black') #Optimized returns will be black and wider
    ax[0, 0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax[0, 0].set_title("Cumulative Returns")
    ax[0, 0].set_xlabel("Date")
    ax[0, 0].set_ylabel("Total Return %")
    ax[0, 0].legend(loc="upper left")
    ax[0, 0].grid(True, alpha=0.3)
    #Plots the returns of individual and portfolio returns

    correlation = returns.corr()
    sns.heatmap(correlation, annot=True, ax=ax[0, 1], cmap="coolwarm")
    ax[0, 1].set_title("Correlation matrix")
    #Creates a correlation matrix

    sns.barplot(x=returns.columns, y=asset_weights, ax=ax[1, 0]) #match the colors to previous subplots
    ax[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax[1, 0].set_title("Optimal asset weights")
    ax[1, 0].set_xlabel("Asset")
    ax[1, 0].set_ylabel("Weight %")
    #Shows a bar chart of the optimal weights

    wealth = portfolio_returns + 1
    peak = wealth.cummax()
    drawdown = (wealth/peak) - 1
    drawdown = drawdown.clip(upper=0)
    #Calculates the inputs for the max drawdown chart

    ax[1, 1].plot(portfolio_returns.index, drawdown, label="Drawdown", color='red')
    ax[1, 1].fill_between(portfolio_returns.index, drawdown, 0, color='red', alpha=0.3)
    ax[1, 1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax[1, 1].set_xlabel("Date")
    ax[1, 1].set_ylabel("Max Drawdown %")
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].set_title("Max Drawdown")
    #Creates a max drawdown chart

    plt.tight_layout() #Shows all visualizations so it fills the whole screen

    plt.show()

if __name__ == '__main__':
    risk_free_rate = 0.04 #Rate of 1 year US Treasury bills

    input_tickers, input_start_date, input_end_date = get_inputs()

    adj_log_returns = calculate_returns(input_tickers, input_start_date, input_end_date)
    adj_log_returns_mean = adj_log_returns.mean() * 252

    covariance_matrix = create_covariance_matrix(adj_log_returns)

    optimized_weights = mean_variance_optimization(covariance_matrix, adj_log_returns)

    for ticker, weight in zip(adj_log_returns.columns, optimized_weights): #Prints the optimal weight for each ticker
        print(f"{ticker}: {weight:.2%}")

    cum_returns, cum_portfolio_returns = in_sample_back_test(adj_log_returns, optimized_weights)

    for index, ticker in enumerate(adj_log_returns.columns):
        individual_weights = np.zeros(len(adj_log_returns.columns))
        individual_weights[index] = 1

        individual_sharpe = portfolio_performance(individual_weights, adj_log_returns_mean, covariance_matrix, risk_free_rate)[2]
        print(f"{ticker} Sharpe Ratio: {individual_sharpe:.3}") #Prints the sharpe ratios of each ticker


    portfolio_sharpe = portfolio_performance(optimized_weights, adj_log_returns_mean, covariance_matrix, risk_free_rate)[2]
    print(f"Portfolio Sharpe Ratio: {portfolio_sharpe:.3}") #Prints the sharpe ratio of the optimized portfolio

    create_visualization(cum_returns, cum_portfolio_returns, optimized_weights)