#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
from pandas_datareader import data as net
import datetime
import json
import csv


# In[32]:


start = datetime.datetime(2014,1,1)
end = datetime.datetime(2019,4,1)
start_less = datetime.datetime(2018,1,1)
end_less = datetime.datetime(2019,4,1)

# Let's get Apple stock data; Apple's ticker symbol is AAPL
# First argument is the series we want, second is the source ("yahoo" for Yahoo! Finance), third is the start date, fourth is the end date
bit_more = net.DataReader("BTC-USD", 'yahoo', start, end)
bit_less = net.DataReader("BTC-USD", 'yahoo', start_less, end_less)


# In[33]:


js = pd.read_json('../input/chart_data_full_86400.json')
js.to_csv('csv.csv')
js300 = pd.read_json('../input/chart_data_full_300.json')
js300.to_csv('csv300.csv')
bit = pd.read_csv('csv.csv')
data = pd.read_csv('csv300.csv', index_col = 'date')
# x = json.loads(js)
#
# f = csv.writer(open("test.csv", "wb+"))


# In[34]:


data = data[['close', 'high', 'low', 'open', 'volume', 'weightedAverage']]
data = data.rename(columns = {'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume', 'weightedAverage': 'Adj Close', 'date': 'Date'})
bit = bit[['close', 'high', 'low', 'open', 'volume', 'weightedAverage']]
bit = bit.rename(columns = {'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume', 'weightedAverage': 'Adj Close', 'date': 'Date'})
# data.head(3)


# In[35]:


full_set = pd.concat([bit_more, data], axis = 0, sort = True)
bit_full = pd.concat([bit_less, bit], axis = 0, sort = True)


# In[36]:


import matplotlib
import matplotlib.pyplot as plt  # Import matplotlib
from matplotlib import pylab
# This line is necessary for the plot to appear in a Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')
# Control the default size of figures in this Jupyter notebook
get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (15, 9)  # Change the size of plots


# In[37]:


get_ipython().system('pip install mpl_finance')


# In[38]:


# full_set["Adj Close"].plot(grid=True)  # Plot the adjusted closing price of AAPL

from matplotlib.dates import DateFormatter, WeekdayLocator,     DayLocator, MONDAY 
from mpl_finance import candlestick_ohlc   #  pip install mpl_finance

def pandas_candlestick_ohlc(dat, stick="day", otherseries=None):
    mondays = WeekdayLocator(MONDAY)  # major ticks on the mondays
    alldays = DayLocator()  # minor ticks on the days
    dayFormatter = DateFormatter('%d')  # e.g., 12

    # Create a new DataFrame which includes OHLC data for each period specified by stick input
    transdat = dat.loc[:, ["Open", "High", "Low", "Close"]]
    if (type(stick) == str):
        if stick == "day":
            plotdat = transdat
            stick = 1  # Used for plotting
        elif stick in ["week", "month", "year"]:
            if stick == "week":
                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1])  # Identify weeks
            elif stick == "month":
                transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month)  # Identify months
            transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0])  # Identify years
            grouped = transdat.groupby(list(set(["year", stick])))  # Group by year and other appropriate variable
            plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [],
                                    "Close": []})  # Create empty data frame containing what will be plotted
            for name, group in grouped:
                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0, 0],
                                                       "High": max(group.High),
                                                       "Low": min(group.Low),
                                                       "Close": group.iloc[-1, 3]},
                                                      index=[group.index[0]]))
            if stick == "week":
                stick = 5
            elif stick == "month":
                stick = 30
            elif stick == "year":
                stick = 365

    elif (type(stick) == int and stick >= 1):
        transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
        grouped = transdat.groupby("stick")
        plotdat = pd.DataFrame(
            {"Open": [], "High": [], "Low": [], "Close": []})  # Create empty data frame containing what will be plotted
        for name, group in grouped:
            plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0, 0],
                                                   "High": max(group.High),
                                                   "Low": min(group.Low),
                                                   "Close": group.iloc[-1, 3]},
                                                  index=[group.index[0]]))

    else:
        raise ValueError(
            'Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')

    # Set plot parameters, including the axis object ax used for plotting
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
#     if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
    weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
#     else:
#         weekFormatter = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_formatter(weekFormatter)

    ax.grid(True)

    # Create the candelstick chart
    candlestick_ohlc(ax, list(
        zip(list(plotdat.index.tolist()), plotdat["Open"].tolist(), plotdat["High"].tolist(),
            plotdat["Low"].tolist(), plotdat["Close"].tolist())),
                     colorup="green", colordown="red", width=stick * .4)

    # Plot other series (such as moving averages) as lines
    if otherseries != None:
        if type(otherseries) != list:
            otherseries = [otherseries]
        dat.loc[:, otherseries].plot(ax=ax, lw=1.3, grid=True)

    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.show() 

pandas_candlestick_ohlc(bit_full.reset_index())
pandas_candlestick_ohlc(bit.reset_index())


# In[39]:


y = full_set['Adj Close'] 


# In[40]:


X = full_set.drop(['Adj Close'], axis = 1)


# In[41]:


import keras
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
import tensorflow as tf

model = Sequential()
model.add(Dense(256, input_dim=5))
model.add(BatchNormalization())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1, kernel_initializer='normal'))
model.add(Activation('relu'))


# In[42]:


NoNaN = TerminateOnNaN()
earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 200, verbose = 0, mode = 'auto')
mcp_save = ModelCheckpoint('model.best.hdf5', save_best_only = True, monitor = 'val_loss', mode = 'auto')
lr_loss = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 7, verbose = 0, mode = 'auto')
rate_reduction = ReduceLROnPlateau(monitor = 'val_loss', patience = 3, verbose = 0, factor = 0.5, min_lr = 0.00001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=0)
model.compile(loss='mean_squared_error', optimizer='adam')


# In[43]:


from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(max_train_size=None, n_splits=9)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = y.values[train_index], y.values[test_index]


# In[44]:


model.fit(X_train, y_train, 
          epochs = 6000, 
          batch_size = 512, 
          verbose=0, 
          validation_data=(X_test, y_test),
          callbacks=[reduce_lr, NoNaN, earlyStopping, mcp_save, lr_loss, rate_reduction])

model.load_weights('model.best.hdf5')


# In[46]:


submit = pd.DataFrame({'volume': model.predict(X_test).tolist()}) 
submit.head(12)


# In[ ]:




