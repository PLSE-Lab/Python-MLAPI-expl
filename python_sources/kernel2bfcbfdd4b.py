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


# In[ ]:


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# In[ ]:


df = pd.read_excel("/kaggle/input/commons/common_dum.xlsx")
df


# In[ ]:


y=df


# In[ ]:


from datetime import datetime
def datemaker(string):
    date_str = '01-04-'+string[-4:]
    date_object = datetime.strptime(date_str, '%m-%d-%Y').date()
    return date_object
datemaker("2017-2018")
y


# In[ ]:


df=df.iloc[::-1] # reversing the dataset
df['Year']=df['Year'].apply(datemaker)
df = df.set_index('Year')
y.plot(figsize=(15, 6))
plt.show()
df


# In[ ]:


# (df[['GDP (INR)']].values)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from sklearn import linear_model
# y=df
# reg = linear_model.LinearRegression()
gdp_true=df[['GDP (INR)']].values
train_data=(df[['Prices (USD/bbl)','Forex (USD/INR)']].values)
# reg.fit(train_data, gdp_true )


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
# reg = linear_model.Ridge(alpha=0)
lasso= linear_model.Lasso()
from sklearn.linear_model import LinearRegression

np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)

# transforming the data to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]
# print(x,y)
model = linear_model.Ridge(alpha=0,tol=1)
x=train_data
y=gdp_true
model.fit(x, y)
y_pred = model.predict(x)
x=np.array(x)
model = linear_model.Lasso(alpha=1,tol=1)
x=train_data
y=gdp_true
model.fit(x, y)
y_pred2 = model.predict(x)
x=np.array(x)
# plt.scatter(x[:,0], y, s=10)
# plt.plot(x[:,0], y_pred, color='r')
# plt.show()
from matplotlib import pyplot as plt
plt.plot(df.index, gdp_true,
             df.index, y_pred,
                df.index, y_pred2)
# plt.plot(y[-1::-1])
model.get_params


# In[ ]:


# from sklearn import tree
from sklearn.linear_model import SGDRegressor
clf = SGDRegressor()
# model = linear_model.Ridge(alpha=1,tol=1)
x=train_data
y=gdp_true
clf.fit(x, y)
y_pred = clf.predict(x)
x=np.array(x)
from sklearn.ensemble import AdaBoostRegressor

model = AdaBoostRegressor(random_state=0, n_estimators=100)
x=train_data
y=gdp_true
model.fit(x, y)
y_pred2 = model.predict(x)
x=np.array(x)
# plt.scatter(x[:,0], y, s=10)
# plt.plot(x[:,0], y_pred, color='r')
# plt.show()
from matplotlib import pyplot as plt
plt.plot(df.index, gdp_true,
             df.index, y_pred
         ,
                df.index, y_pred2)
# plt.plot(y[-1::-1])
model.get_params
# y_pred


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score

gdp_pred = model.predict(train_data)

# The coefficients
# print('Coefficients: \n', model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(gdp_true, gdp_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(gdp_true, gdp_pred))
# z=np.array([gdp_true, gdp_pred])
# z.plot(figsize=(15, 6))
# dir(reg)


# In[ ]:





# In[ ]:


# from matplotlib import pyplot as plt
# plt.plot(df.index, gdp_true[-1::-1],
#              df.index, gdp_pred[-1::-1])


# In[ ]:


# reg.coef_
# df


# In[ ]:


#Time Series
#df=df[["GDP (INR)"]]
df.index = pd.to_datetime(df.index)
print(df)
y = df[["GDP (INR)"]].resample('Y').mean()
y.plot(figsize=(15, 6))
plt.show()


# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(df,freq=2)
fig = decomposition.plot()
plt.show()


# In[ ]:


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
for param in pdq:
    for param_seasonal in seasonal_pdq:
#         print (param,param_seasonal)
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[ ]:


print (df.shape)
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 1),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.show()
# plt.show()
kappa=df.index[10]
print (kappa)


# In[ ]:


pred = results.get_prediction(start = 5 , end=40, dynamic=False)
pred_ci = pred.conf_int()
pred.predicted_mean
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('GDP Values')
plt.legend()
# plt.show()
a=list(df.index)

from dateutil.relativedelta import relativedelta
for i in range (0,10):
    a.append(a[29] + relativedelta(years=i+1))
plt.plot(a[4:],pred.predicted_mean)
ax = df["GDP (INR)"].plot(label='observed')


# In[ ]:




