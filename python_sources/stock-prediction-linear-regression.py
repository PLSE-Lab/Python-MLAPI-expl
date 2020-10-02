#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from datetime import datetime as dt

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/prices.csv') 


# In[ ]:


df = df.loc[df['symbol']=='CSCO']
df.drop('symbol', axis=1, inplace=True)
print(np.shape(df))
df.head()


# In[ ]:


# changing the index to date

df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

df.head()


# In[ ]:


# finding NaNs

df.dropna(axis=0 , inplace=True)
df.isna().sum()


# ### Visualization
# ### Plotting CandelStick : Depicts all data using candlestick charts

# In[ ]:


Min_date = df.index.min()
Max_date = df.index.max()
print ("First date is",Min_date)
print ("Last date is",Max_date)
print (Max_date - Min_date)


# In[ ]:


# Plotting Candlestick 
from plotly import tools
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, iplot_mpl
init_notebook_mode()
import plotly.plotly as py
import plotly.graph_objs as go

trace = go.Ohlc(x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'])
data = [trace]
iplot(data, filename='simple_ohlc')


# ### Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[ ]:


# Creating new column 
num = 20 # forcasting 20 days ahead
df['label'] = df['close'].shift(-num) # forcasting close column
                                     
print(df.shape)


# In[ ]:


Data = df.drop(['label'],axis=1)
X = Data.values
X = preprocessing.scale(X)
X = X[:-num]

df.dropna(inplace=True)
Target = df.label
y = Target.values

print(np.shape(X), np.shape(y))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test , y_test) #Returns the coefficient of determination R^2 of the prediction


# In[ ]:


# Prediction 

X_Predict = X[-num:]
Forecast = lr.predict(X_Predict)
print(Forecast)


# ### Plotting the results

# In[ ]:


Date = np.array(df.index) 
last_Date = Date[len(Date)-1]
print(last_Date)


# In[ ]:


# creating timeserie from the last date

trange = pd.date_range('2016-12-2', periods=num, freq='d')
trange


# In[ ]:


# Adding the predicted values to dateframe
Predict_df = pd.DataFrame(Forecast, index=trange)
Predict_df.columns = ['forecast']
Predict_df


# In[ ]:


df_concat = pd.concat([df, Predict_df], axis=1)
df_concat.tail(num)


# In[ ]:


# zooming in the forecast part

df_concat['forecast'].plot(color='orange', linewidth=3)
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

