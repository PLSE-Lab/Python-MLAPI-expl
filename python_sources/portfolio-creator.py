#!/usr/bin/env python
# coding: utf-8

# ## Configs

# In[ ]:


import pandas as pd
import numpy as np

from datetime import datetime
from os import listdir
from os.path import isfile, join, splitext


# In[ ]:


START_TIME = datetime(2017, 3, 4)
TRADING_DAYS = 755# 253 # TODO: determine this automatically
NUMBER_OF_PORTFOLIO_SECURITIES = 10
INITAL_TICKER = 'FB'
MAX_VOLATILITY_OF_SINGLE_SECURITY = 0.02
RISK_FREE_RATE = 0.02
RISK_FREE_RATE_DAILY = (1 + RISK_FREE_RATE) ** (1/365) - 1

BAN_LIST = ["KRMD", "FRPT", "EC", "MFINL", "CABO", "CSGP", "NEE", "TRNS"]


# ## Read Data

# In[ ]:


get_ipython().run_cell_magic('time', '', "\npath = '/kaggle/input/stock-market-dataset'\n\ndef read_data(path, start_time):\n    data = {}\n    files = [f for f in listdir(path) if isfile(join(path, f))]\n    for f in files:\n        ticker = pd.read_csv(join(path, f))\n        ticker['Date'] = pd.to_datetime(ticker['Date'])\n\n        symbol = splitext(f)[0]\n        data_for_period = ticker[ticker['Date'] > start_time][['Date', 'Adj Close']]\n        if data_for_period.shape[0] != TRADING_DAYS:\n            continue # entered the market after the start of analysis\n        \n        data[symbol] = data_for_period\n        data[symbol]['Return'] = data[symbol]['Adj Close'].pct_change()  # daily return\n        data[symbol] = data[symbol].dropna()\n\n    return data\n\netfs = read_data(join(path, 'etfs'), START_TIME)\nstocks = read_data(join(path, 'stocks'), START_TIME)\n\ntickers = {**etfs, **stocks}")


# ## Portfolio Optimisation

# In[ ]:


portfolio = {}
portfolio[INITAL_TICKER] = tickers[INITAL_TICKER]['Return'].to_numpy()
get_return = lambda portfolio: np.mean(np.array(list(portfolio.values())), axis=0)


# In[ ]:


def portfolio_stat(portfolio):
    ret = portfolio.mean()
    std = portfolio.std()
    sharpe_ratio = (ret - RISK_FREE_RATE_DAILY) / std
    
    return ret, std, sharpe_ratio

def portfolio_stat_annual(portfolio):
    ret = (portfolio.mean() + 1) ** 365 - 1
    std = portfolio.std() * (250 ** 0.5)
    sharpe_ratio = (ret - RISK_FREE_RATE) / std
    
    return ret, std, sharpe_ratio


# In[ ]:


ret, std, sharpe_ratio = portfolio_stat(tickers[INITAL_TICKER]['Return'])
print('1) Ticker = {}, Avg Return = {}, Std Deviation = {}, Sharpe Ratio = {}'.format(INITAL_TICKER, ret, std, sharpe_ratio))

while len(portfolio) < NUMBER_OF_PORTFOLIO_SECURITIES:
    possible_matches = []
    for t in tickers:
        if t in portfolio:
            continue
            
        new_portfolio = {**portfolio, t: tickers[t]['Return']}
        _, _, sharpe_ratio = portfolio_stat(get_return(new_portfolio))
        possible_matches.append((t, sharpe_ratio))
        
    
    for t, ratio in reversed(sorted(possible_matches, key=lambda item: item[1])):
        if tickers[t]['Return'].std() > MAX_VOLATILITY_OF_SINGLE_SECURITY:
            continue
            
        if t in BAN_LIST:
            continue
            
        if t not in portfolio:
            break
    
    
    portfolio[t] = tickers[t]['Return']
    ret, std, sharpe_ratio = portfolio_stat(get_return(portfolio))
    print('{}) Ticker = {}, Avg Return = {}, Std Deviation = {}, Sharpe Ratio = {}'.format(len(portfolio), t, ret, std, sharpe_ratio))


# In[ ]:


ret, std, sharpe_ratio = portfolio_stat_annual(get_return(portfolio))
print('Portfolio Stats: Avg Return = {}, Std Deviation = {}, Sharpe Ratio = {}'.format(ret, std, sharpe_ratio))


# In[ ]:


# ! pip install yfinance > /dev/null 2>&1


# In[ ]:


# import yfinance as yf

# get_or_default = lambda info, key: info[key] if key in info else ""
    

# for t in portfolio:
#     info = yf.Ticker(t).info
#     print('ticker: {}, sector: {}, market cap: {}'.format(t, get_or_default(info, 'sector'), get_or_default(info, 'marketCap')))

