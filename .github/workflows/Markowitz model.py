import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization

# on average there are 252 trading days in a year
NUM_TRADING_DAYS = 252

# we will generate random w (different portfolios)
NUM_PORTFOLIOS = 10000

# stocks we are going to handle
stocks = ['AAPL', 'WMT', 'GE', 'AMZN', 'DB']

# historical data - define START and END dates
start_date = '2010-01-01'
end_date = '2017-01-01'

def download_data():
    # Dictionary to store data
    stock_data = {}

    for stock in stocks:
        try:
            # We are considering closing prices
            ticker = yf.Ticker(stock)
            history = ticker.history(start=start_date, end=end_date)
            if 'Close' in history.columns:
                stock_data[stock] = history['Close']
            else:
                print(f"Warning: 'Close' prices not found for {stock}. Skipping.")
        except Exception as e:
            print(f"Error downloading data for {stock}: {e}")

    # Convert to DataFrame
    return pd.DataFrame(stock_data)

def show_data(data):
    if data is None or data.empty:
        print("No data to show.")
    else:
        print("Downloaded data preview:")
        print(data.head())  # Show a preview of the data
        data.plot(figsize=(10, 5), title="Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend(data.columns)
        plt.show()

def calculate_return(data):
    """
    Calculate log daily returns.
    Log normalization is used to make variables comparable.
    """
    log_return = np.log(data / data.shift(1))  # Shift by 1: s(t+1)/s(t)
    return log_return[1:]  # Remove the first row (undefined return)

def show_statistics(returns):
    """
    Show mean and covariance of annualized returns.
    """
    print("Annualized Mean Returns:")
    print(returns.mean() * NUM_TRADING_DAYS)
    print("\nAnnualized Covariance Matrix:")
    print(returns.cov() * NUM_TRADING_DAYS)

def show_mean_variance(returns, weights):
    # we are after the annual return
    portfolio_return = np.sum(returns.mean()*weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))

    print("Expected portfolio mean (return): ", portfolio_return)
    print("Expected portfolio volatility (standard deviation): ", portfolio_volatility)

def show_portfolios(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns/volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

def generate_portfolios(returns, portfolio_weights=None):

    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.rand(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)

def statistics (weights, returns):
    portfolio_return = np.sum(returns.mean()*weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))

    return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])

# scipy optimize module can find the minimum of a given function
# the maximum of f(x) is the min of -f(x)
def min_function_sharpe(weights, returns):
    # we grabbed the item w index 2
    return -statistics(weights, returns)[2]

# what are the constraints? the sum of weights = 1 !!!
# sum w - 1 = 0  f(x)=0 this is the function to minimize

def optimize_portfolio(weights, returns):
    # Initial guess: equally distributed weights
    initial_weights = np.ones(len(stocks)) / len(stocks)

    # Constraint: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # Bounds: weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(len(stocks)))

    # Optimize for maximum Sharpe ratio
    result = optimization.minimize(fun=min_function_sharpe,
                                   x0=initial_weights,
                                   args=returns,
                                   method='SLSQP',
                                   bounds=bounds,
                                   constraints=constraints)

    return result

def print_optimal_portfolio(optimum, returns):
    print("Optimal portfolio:", optimum.x.round(3))
    print("Expected return, volatility and Sharpe ratio: ", statistics(optimum['x'].round(3),
                                                                       returns))

def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(opt.x, rets)[1], statistics(opt.x, rets)[0], 'g*', markersize=20.0)
    plt.show()


if __name__ == '__main__':
    # Download the data and store it in the 'dataset' variable
    dataset = download_data()
    dataset = dataset.replace(0, np.nan).dropna()

    # Check for missing data
    print("\nChecking for missing data:")
    print(dataset.isnull().sum())
    print("\nDataset is clean and ready:")
    print(dataset.info())
    print(dataset.head())

    # Display the stock data
    #show_data(dataset)

    # Calculate and display log daily returns
    log_daily_returns = calculate_return(dataset)

    # Debugging log returns
    if log_daily_returns.empty:
        print("Log daily returns are empty. Check the dataset or calculation.")
    else:
        print("\nLog Daily Returns Preview:")
        print(log_daily_returns.head())  # Preview first 5 rows
        print("\nShape of log_daily_returns:", log_daily_returns.shape)

        # Show statistics of the log returns
        print("\nStatistics of Log Returns:")
        show_statistics(log_daily_returns)

    # Initial weights are used only inside the optimization process.
    # If you want to print initial weights, define them explicitly here:
    #initial_weights = np.ones(len(stocks)) / len(stocks)
    #print("Initial Weights:", initial_weights)
    print("Returns:", log_daily_returns.mean())
    #print("Optimization Result:", optimum)

    pweights, means, risks = generate_portfolios(log_daily_returns)
    optimum = optimize_portfolio(pweights, log_daily_returns)
    show_optimal_portfolio(optimum, log_daily_returns, means, risks)
    print_optimal_portfolio(optimum, log_daily_returns)
