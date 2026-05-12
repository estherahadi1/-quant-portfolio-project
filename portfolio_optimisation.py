import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
data = yf.download(stocks, start='2020-01-01')['Adj Close']

returns = data.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()

num_portfolios = 5000
results = []

for _ in range(num_portfolios):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)

    portfolio_return = np.sum(mean_returns * weights)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    results.append([portfolio_return, portfolio_std])

results = np.array(results)

plt.scatter(results[:,1], results[:,0])
plt.xlabel('Risk (Std Dev)')
plt.ylabel('Return')
plt.title('Portfolio Optimisation')
plt.show()