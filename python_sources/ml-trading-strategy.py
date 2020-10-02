# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import datetime as dt
from pylab import mpl, plt
import warnings
import sys
from sklearn.linear_model import LinearRegression  
from itertools import product

 

def read_data(crypto_file):
    raw = pd.read_csv(crypto_file,
                  index_col=0, parse_dates=True)
    
    column = 'Close'
    
    data = (pd.DataFrame(raw[column]).dropna())
    
    return data

def moving_averages_crossover_strategy(data, SMA1, SMA2):
    '''
    Trading based on simple moving averages (SMAs)
    Calculates the values for the shorter SMA
    Calculates the values for the longer SMA        
    '''
    data['SMA1'] = data['Close'].rolling(SMA1).mean()  
    data['SMA2'] = data['Close'].rolling(SMA2).mean()
    
    return data
    
    
def position(data):
    ''' np.where(cond, a, b) 
    evaluates the condition cond element-wise and places a when True and b otherwise.
    Given that the strategy leads to two periods only during which crypto should be shorted
    '''
    data.dropna(inplace=True)
    data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
    
    return data
    
    
def plotfig(data):
    data.plot(figsize=(10, 6));
 
    return True
    
def plot_position(data):
    ax = data.plot(secondary_y='Position', figsize=(10, 6))
    ax.get_legend().set_bbox_to_anchor((0.25, 0.85));
    return True
    
def plot_return(data):
    data['Returns'].hist(bins=35, figsize=(10, 6));
    return True
    
def plot_return_strategy(data):
    ax = data[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    data['Position'].plot(ax=ax, secondary_y='Position', style='--')
    ax.get_legend().set_bbox_to_anchor((0.25, 0.85));
    return True
    
def vectorized_backtesting(data):
    '''
    Calculates the log returns of crypto
    Multiplies the position values, shifted by one day, by the log returns
    Sums up the log returns for the strategy and the benchmark investment 
    and calculates the exponential value to arrive at the absolute performance.
    Calculates the annualized volatility for the strategy and the benchmark investment.
    '''
    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1)) 
    data['Strategy'] = data['Position'].shift(1) * data['Returns']  
 
    data.dropna(inplace=True)
    np.exp(data[['Returns', 'Strategy']].sum())  
    
    print (data[['Returns', 'Strategy']].std() * 252 ** 0.5  )
    
    return True
    
    
def optimization(SMA1, SMA2, sma1, sma2, data):
    '''
    Specifies the parameter values for SMA1.
    Specifies the parameter values for SMA2.
    Combines all values for SMA1 with those for SMA2.
    Records the vectorized backtesting results in a DataFrame object.
    '''
    
    results = pd.DataFrame()
    symbol = 'Close'
    for SMA1, SMA2 in product(sma1, sma2):  
        data = pd.DataFrame(data[symbol])
        data.dropna(inplace=True)
        data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))
        data['SMA1'] = data[symbol].rolling(SMA1).mean()
        data['SMA2'] = data[symbol].rolling(SMA2).mean()
        data.dropna(inplace=True)
        data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
        data['Strategy'] = data['Position'].shift(1) * data['Returns']
        data.dropna(inplace=True)
        perf = np.exp(data[['Returns', 'Strategy']].sum())
        results = results.append(pd.DataFrame(
                    {'SMA1': SMA1, 'SMA2': SMA2,
                 'MARKET': perf['Returns'],
                 'STRATEGY': perf['Strategy'],
                 'OUT': perf['Strategy'] - perf['Returns']},
                 index=[0]), ignore_index=True) 
    return results
    
    
def random_walk_hypothesis(data):
    '''
    Defines a column name for the current lag value.
    Creates the lagged version of the market prices for the current lag value.
    Collects the column names for later reference.
    '''
    lags = 5
    cols = []
    for lag in range(1, lags + 1):
        col = 'lag_{}'.format(lag)  
        data[col] = data['Close'].shift(lag)  
        cols.append(col) 

    data.dropna(inplace=True)
    
    reg = np.linalg.lstsq(data[cols], data['Close'], rcond=-1)[0]
    reg.round(3)
      
    plt.figure(figsize=(10, 6))
    plt.bar(cols, reg);
 
    
    data['Prediction'] = np.dot(data[cols], reg)
    
    data[['Close', 'Prediction']].iloc[-75:].plot(figsize=(10, 6));
    return True
 
 
    
def linear_ols_regression(data):
    '''
    The linear OLS regression implementation from scikit-learn is used.
    The regression is implemented on the log returns directly
    and on the direction data which is of primary interest.
    The real-valued predictions are transformed to directional values (+1, -1).
    The two approaches yield different directional predictions in general.
    However, both lead to a relatively large number of trades over time. 
    '''
    
    data['returns'] = np.log(data / data.shift(1))
    data.dropna(inplace=True)
    data['direction'] = np.sign(data['returns']).astype(int)
    data.head()
    lags = 2
    cols = []
    for lag in range(1, lags + 1):
        col = 'lag_{}'.format(lag)
        data[col] = data['returns'].shift(lag)
        cols.append(col)
    data.dropna(inplace=True)
    
    model = LinearRegression()  
    
    data['pos_ols_1'] = model.fit(data[cols], data['returns']).predict(data[cols])   
    data['pos_ols_2'] = model.fit(data[cols], data['direction']).predict(data[cols])  
    
    data[['pos_ols_1', 'pos_ols_2']].head()  
    data[['pos_ols_1', 'pos_ols_2']] = np.where(
            data[['pos_ols_1', 'pos_ols_2']] > 0, 1, -1)  
    
    data['pos_ols_1'].value_counts()     
    data['pos_ols_2'].value_counts()  
    
    (data['pos_ols_1'].diff() != 0).sum()    
    (data['pos_ols_2'].diff() != 0).sum() 
    
    data['strat_ols_1'] = data['pos_ols_1'] * data['returns']   
    data['strat_ols_2'] = data['pos_ols_2'] * data['returns']
    
    data[['returns', 'strat_ols_1', 'strat_ols_2']].sum().apply(np.exp)
    (data['direction'] == data['pos_ols_1']).value_counts()  
    (data['direction'] == data['pos_ols_2']).value_counts() 
    data[['returns', 'strat_ols_1', 'strat_ols_2']].cumsum(
        ).apply(np.exp).plot(figsize=(10, 6));
            
    return True