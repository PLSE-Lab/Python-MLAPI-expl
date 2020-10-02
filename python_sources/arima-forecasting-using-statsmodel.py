#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # ARIMA Forecasting using Statsmodel
# 
# Forecasting is a magic nowadays. Finance institution seeking for good forecasting model to ensure the "uncertain future". It also be a magic if you could done well forecasting for your financial activity :)

# ## 1. Load the Data
# 
# We will drop first 5 row to make the data fully numerical. 

# In[ ]:


df = pd.read_csv("/kaggle/input/exchange-rates/exchange_rates.csv")
df = df.drop(df.index[0:5]).dropna()
df.head()


# In[ ]:


df.columns


# ## 2. Drop ND and NA values
# 
# We will use the numerical values only, so we will drop all the NaN and ND valued rows. 

# In[ ]:


df.head()


# In[ ]:


df.dtypes


# We see that the format was not fixed, so we will fix it. 

# In[ ]:


df[df.columns[0]]


# In[ ]:


df = df[df != 'ND']
df.dropna()


# In[ ]:


from datetime import datetime

df[df.columns[0]] = pd.to_datetime(df[df.columns[0]]) 
df[df.columns[1:len(df.columns)]] = df[df.columns[1:len(df.columns)]].astype(float)


# ## 3. See the trend
# 
# 

# In[ ]:


import matplotlib.pyplot as plt

print(len(df))
print(df.dtypes)


# In[ ]:


import seaborn as sns

for i in range(1,len(df.columns)):
    plt.figure(figsize=(15,4))
    sns.lineplot(x = df[df.columns[0]], y = df[df.columns[i]])


# We dont see the function as stationary, so we need to make it stationary. 

# In[ ]:


for i in range(1,len(df.columns)):
    plt.figure(figsize=(15,4))
    sns.lineplot(x = df[df.columns[0]], y = np.log(df[df.columns[i]]))


# ## 4. Make Stationary Data
# 
# We will use ADF (Augmented Dickey Fuller) for statistical test 
# 
# (Thanks to https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/ for the lesson!)

# In[ ]:


from statsmodels.tsa.stattools import adfuller

def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[ ]:


for i in range(1,len(df.columns)):
    df[df.columns[i]] = df[df.columns[i]].fillna(method='ffill')
    print('\n',df.columns[i])
    adf_test(df[df.columns[i]])


# We will try to differentiate the data 

# In[ ]:


for i in range(1,len(df.columns)):
    plt.figure(figsize=(15,4))
    df[df.columns[i]] = df[df.columns[1]] - df[df.columns[i]].shift(1)
    df[df.columns[i]].dropna().plot()
    


# I think the first was stationary, but the rest is works for later, so it needs different treatment. We will proceed to the first.

# In[ ]:


plt.figure(figsize=(15,4))
df[df.columns[1]].dropna().plot()
adf_test(df[df.columns[1]].dropna())


# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
# using 1,1,1 ARIMA Model
model = ARIMA(df[df.columns[1]].dropna(), order=(1,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# yea the p-value was less than 0.05 so it should be significant. We can plot the residuals to ensure the mean is near-zero. 

# In[ ]:


plt.figure(figsize=(15,4))
residuals = pd.DataFrame(model_fit.resid)
residuals.plot(title="Residuals")
residuals.plot(kind='kde', title='Density')
plt.show()


# ## 5. Predicting Values
# 
# We will split the data into train and test set, with ratio of 7:3. 
# 
# (Thanks to https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/) 

# In[ ]:


data = df[df.columns[1]].dropna().values

size = int(len(data) * 0.7)
train, test = data[0:size], data[size:len(data)]
history = [x for x in train]
predictions = list()

for t in range(len(test)):
    model = ARIMA(history, order=(1,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))


# Yea, too long to wait, so I decide to immediately plot it. Not even halfway but I am impatient for the graph. The graph took differential form. 

# In[ ]:


from sklearn.metrics import mean_squared_error

#error = mean_squared_error(test, predictions)
#print('Test MSE: %.3f' % error)
plt.figure(figsize=(15,4))
plt.plot(test, label = 'actual')
plt.plot(predictions, color='red', label = 'predicted')
plt.legend()
plt.show()


# ## 6. Optimization
# 
# Optimization could be done from the stationary test of datas, or from the ARIMA models. The p, d, and q values could be varied. This work would be done later. 
