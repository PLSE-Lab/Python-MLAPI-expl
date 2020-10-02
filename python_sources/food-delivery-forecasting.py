#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/train.csv')
train.head()


# In[ ]:


data = train[train['center_id']==55]
data = data[data['meal_id'] == 1885]


# In[ ]:


data


# In[ ]:


data.info()


# In[ ]:


corrmat = data.corr()


# In[ ]:



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
num_correlation = data.select_dtypes(exclude='object').corr()
plt.figure(figsize=(10,8))
plt.title('High Correlation')
sns.heatmap(num_correlation > 0.4, annot=True, square=True)


# In[ ]:


#saleprice correlation matrix
k = 9 #number of variables for heatmap
cols = corrmat.nlargest(k, 'num_orders')['num_orders'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 9}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# we will take  homepage_featured,  diff, num order

# In[ ]:


sns.distplot(data['num_orders'])


# In[ ]:


data['num_orders'].plot()


# In[ ]:


data['num_orders'].skew()


# In[ ]:





# In[ ]:


sns.boxplot(['num_orders'], data=data)


# In[ ]:


import numpy as np
t = np.log1p(data['num_orders'])
t.plot()


# In[ ]:


sns.boxplot(t)


# In[ ]:


data['diff'] = data['base_price']- data['checkout_price']


# In[ ]:


data


# In[ ]:





# In[ ]:


# we will take  homepage_featured,  diff, num order
# data.drop(columns=['id','week'])

x = data[['homepage_featured','diff','num_orders']]
x = x.reset_index()


# In[ ]:


a = x['num_orders'].quantile(0.98)
a


# In[ ]:


# x = x[x['num_orders']<a]


# In[ ]:


sns.boxplot(x['num_orders'])


# In[ ]:


np.log1p(x['num_orders']).plot()


# # normalization

# In[ ]:


# x['num_orders'] = np.log1p(x['num_orders'])
# x


# In[ ]:


x.drop(columns='index',inplace=True)


# In[ ]:


# lets partition data

x_train = x.drop(columns='num_orders')
y_train = x['num_orders']


# In[ ]:


X_train = x_train.iloc[:138,:]
X_test = x_train.iloc[138:,:]
Y_train =  y_train.iloc[:138]
Y_test = y_train.iloc[138:]


# In[ ]:


print(len(X_test))
print(len(Y_test))


# In[ ]:


Y_test


# In[ ]:


from xgboost import XGBRegressor
model_2 = XGBRegressor(
 learning_rate =0.3,
 eval_metric='rmse',
    n_estimators=5000,
  
  
 )
#model.fit(X_train, y_train)
model_2.fit(X_train, Y_train, eval_metric='rmse', 
          eval_set=[(X_test, Y_test)], early_stopping_rounds=1000, verbose=100)


# In[ ]:


model =XGBRegressor(
 learning_rate =0.001,
    n_estimators=2)
  


# In[ ]:


model.fit(X_train,Y_train)


# In[ ]:


preds = model.predict(x_train)


# In[ ]:


preds


# In[ ]:


# preds = np.exp(preds)


# In[ ]:


len(preds)


# In[ ]:


# true_value = np.exp(x['num_orders'])
# true_value


# In[ ]:


plt.plot(x['num_orders'])
plt.plot(preds, color='r')


# # Prohpet model

# In[ ]:


x


# In[ ]:


x['Date'] = pd.date_range('2015-01-01', periods=145, freq='W')
x


# In[ ]:


from fbprophet import Prophet


# In[ ]:


prophet_model = x[['Date','num_orders']]
prophet_model =prophet_model.rename(columns={'Date':'ds',
                             'num_orders':'y'})
prophet_model


# In[ ]:


m = Prophet(changepoint_prior_scale=0.001)
m.fit(prophet_model)


# In[ ]:


future = m.make_future_dataframe(periods=10)


# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.iplot(fig)


# In[ ]:


plt.figure(figsize=(25,5))
plt.plot(x['num_orders'])
plt.plot(forecast['yhat'], color='r')


# In[ ]:


x


# # Arima

# In[ ]:


df = x[['Date','num_orders']]
df = df.set_index(['Date'])
df


# In[ ]:


rolmean = df.rolling(window=6).mean()
rolstd = df.rolling(window=6).std()


# In[ ]:


plt.figure(figsize=(25,5))
plt.plot(df, color='blue', label='original cases')
plt.plot(rolmean, color='red', label='rolling mean')
plt.plot(rolstd, color='black', label='rolling standard deviation')
plt.legend(loc='best')
plt.show()


# In[ ]:


from statsmodels.tsa.stattools import adfuller
def test(data):
    rolmean = data.rolling(window=2).mean()
    rolstd = data.rolling(window=2).std()
    plt.figure(figsize=(25,5))
    plt.plot(data, color='blue', label='original cases')
    plt.plot(rolmean, color='red', label='rolling mean')
    plt.plot(rolstd, color='black', label='rolling standard deviation')
    plt.legend(loc='best')
    plt.show()
    
    dftest = adfuller(data['num_orders'], autolag = 't-stat')
    dfoutput = pd.Series(dftest[0:4], index=['test statitics','p_value','lags used','number of observations'])
    for key,value in dftest[4].items():
        dfoutput['critcal value (%s)'%key] = value
        
    print(dfoutput)


# In[ ]:


test(df)


# In[ ]:


df_log = np.log(df)
test(df_log)


# In[ ]:


movingaverage = df_log.rolling(window=4).mean()

df_log_minus = df_log - movingaverage
df_log_minus.dropna(inplace=True)
df_log_minus.head(12)


# In[ ]:


test(df_log_minus)


# In[ ]:


from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(df_log_minus, nlags=50)
lag_pacf = pacf(df_log_minus, nlags=20, method='ols')

plt.figure(figsize=(10,8))
#plot acf
plt.subplot(211)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_log_minus)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_log_minus)), linestyle='--', color='gray')
plt.title('ACF')
plt.legend(loc='best')

#plot pacf
plt.subplot(212)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_log_minus)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_log_minus)), linestyle='--', color='gray')
plt.title('PACF')
plt.legend(loc='best')


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df_log, order=(2,1,1))
result = model.fit(disp=-1)
plt.figure(figsize=(10,8))
plt.plot(df_log_minus)
plt.plot(result.fittedvalues, color='r')
plt.title('RSS %-4F'% sum((result.fittedvalues- df_log_minus['num_orders'])**2))


# In[ ]:


result.fittedvalues


# In[ ]:


pred_arima_diff = pd.Series(result.fittedvalues, copy=True)
pred_arima_diff


# In[ ]:


pred_arima_diff_cumsum = pred_arima_diff.cumsum()
pred_arima_diff_cumsum.tail()


# In[ ]:


prediction = pd.Series(df_log['num_orders'].iloc[0], index=df_log.index)
prediction = prediction.add(pred_arima_diff_cumsum, fill_value=0)
prediction.head()


# In[ ]:


prediction = np.exp(prediction)
prediction = prediction.reset_index()
prediction.drop(columns='Date', inplace=True)

# prediction = pd.DataFrame(prediction)


# In[ ]:


prediction.plot()
x['num_orders'].plot()


# In[ ]:


forecast['yhat']


# In[ ]:


# a = pd.DataFrame()
b = (forecast['yhat']) 


# In[ ]:


b[0]


# In[ ]:


plt.figure(figsize=(25,5))
plt.plot(x['num_orders'])
plt.plot(b, color='r')


# In[ ]:


result.plot_predict(1,155)
plt.figure(figsize=(10,8))

