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


# Import all of them 
sales=pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv", parse_dates=True, squeeze=True)

# settings
import warnings
warnings.filterwarnings("ignore")

item_cat=pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
item=pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
sub=pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")
shops=pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")
test=pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")


# In[ ]:


from __future__ import print_function
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smtsa


# In[ ]:


sales.head()


# In[ ]:


ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,8))
plt.title('Total Sales of the company')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts);


# In[ ]:


plt.figure(figsize=(5.5, 5.5))
ts.plot(color='b')
plt.title('Monthly item sold')
plt.xlabel('Monthly')
plt.ylabel('Sales')
plt.xticks(rotation=30)


# In[ ]:


from sklearn.linear_model import LinearRegression
trend_model = LinearRegression(normalize=True, fit_intercept=True)
trend_model.fit(np.array(ts.index).reshape((-1,1)), ts.values)
print('Trend model coefficient={} and intercept={}'.format(trend_model.coef_[0], trend_model.intercept_) 
)


# In[ ]:


residuals = np.array(ts.values) - trend_model.predict(np.array(ts.index).reshape((-1,1)))
plt.figure(figsize=(5.5, 5.5))
pd.Series(data=residuals, index=ts.index).plot(color='b')
plt.title('Residuals of trend model for sales')
plt.xlabel('monthly')
plt.ylabel('sales')
plt.xticks(rotation=30)


# In[ ]:


ts= pd.DataFrame(ts)


# In[ ]:


ts.index.size


# In[ ]:


len(residuals)


# In[ ]:


ts['Residuals'] = residuals


# In[ ]:


sa= sales.groupby('date_block_num')['date'].apply(np.copy)


# In[ ]:


sa= sa.map(lambda x: x[0])


# In[ ]:


ts['year']= (pd.to_datetime(sa)).dt.year


# In[ ]:


ts.shape


# In[ ]:


sales.shape


# In[ ]:


ts.tail()


# In[ ]:


seasonal_sub_series_data = ts.groupby(by=['year'])['Residuals'].aggregate([np.mean, np.std]) 


# In[ ]:


seasonal_sub_series_data.columns = ['yearly Mean', 'yearly Standard Deviation']


# In[ ]:


seasonal_sub_series_data


# In[ ]:


plt.figure(figsize=(5.5, 5.5))
seasonal_sub_series_data['yearly Mean'].plot(color='b')
plt.title('yearly Mean of Residuals')
plt.xlabel('Time')
plt.ylabel('sales')
plt.xticks(rotation=30)


# In[ ]:


plt.figure(figsize=(5.5, 5.5))
seasonal_sub_series_data['yearly Standard Deviation'].plot(color='b')
plt.title('yearly Standard Deviation of Residuals')
plt.xlabel('Time')
plt.ylabel('sales')
plt.xticks(rotation=30)


# In[ ]:


import seaborn as sns


# In[ ]:


plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(data=ts, y='Residuals', x='year')
g.set_title('yearly Mean of Residuals')
g.set_xlabel('Time')
g.set_ylabel('sales')


# In[ ]:


ts.index.size


# In[ ]:


lag = range(0,34)
acf = []
for l in lag:
    acf.append(ts['item_cnt_day'].autocorr(l))


# In[ ]:


plt.figure(figsize=(5.5, 5.5))
plt.plot(acf, marker='.', color='b')
plt.title('Autocorrelation function for sales')
plt.xlabel('Lag in terms of number of months')
plt.ylabel('Autocorrelation function')


# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[ ]:


#Plot autocorrelation and confidence intervals using the plot_acf function
plt.figure(figsize=(5.5, 5.5))
plot_acf(ts['item_cnt_day'], lags=30)


# In[ ]:


#Plot autocorrelation and confidence intervals using the plot_acf function
plt.figure(figsize=(5.5, 5.5))
plot_pacf(ts['item_cnt_day'], lags=30)


# In[ ]:


from statsmodels.tsa import stattools


# In[ ]:


adf_result = stattools.adfuller(ts['item_cnt_day'], autolag='AIC')


# In[ ]:


print('p-val of the ADF test in sales:', adf_result[1])


# In[ ]:


ts['5-month Moving Avg'] = ts['item_cnt_day'].rolling(5).mean()


# In[ ]:


fig = plt.figure(figsize=(5.5, 5.5))
ax = fig.add_subplot(2,1,1)
ts['item_cnt_day'].plot(ax=ax, color='b')
ax.set_title('sales during Oct2013-Oct2015')
ax = fig.add_subplot(2,1,2)
ts['5-month Moving Avg'].plot(ax=ax, color='r')
ax.set_title('5-month Moving Average')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)


# In[ ]:


MA2 = ts['item_cnt_day'].rolling(window=2).mean()
TwoXMA2 = MA2.rolling(window=2).mean()

MA4 = ts['item_cnt_day'].rolling(window=4).mean()
TwoXMA4 = MA4.rolling(window=2).mean()

MA3 = ts['item_cnt_day'].rolling(window=3).mean()
ThreeXMA3 = MA3.rolling(window=3).mean()


# In[ ]:


MA2 = MA2.loc[~pd.isnull(MA2)]
TwoXMA2 = TwoXMA2.loc[~pd.isnull(TwoXMA2)]

MA4 = MA4.loc[~pd.isnull(MA4)]
TwoXMA4 = TwoXMA4.loc[~pd.isnull(TwoXMA4)]

MA3 = MA3.loc[~pd.isnull(MA3)]
ThreeXMA3 = TwoXMA4.loc[~pd.isnull(ThreeXMA3)]


# In[ ]:


f, axarr = plt.subplots(3, sharex=True)
f.set_size_inches(5.5, 5.5)

ts['item_cnt_day'].plot(color='b', linestyle = '-', ax=axarr[0])
MA2.plot(color='r', linestyle = '-', ax=axarr[0])
TwoXMA2.plot(color='r', linestyle = '--', ax=axarr[0])
axarr[0].set_title('2 month MA & 2X2 month MA')

ts['item_cnt_day'].plot(color='b', linestyle = '-', ax=axarr[1])
MA4.plot(color='g', linestyle = '-', ax=axarr[1])
TwoXMA4.plot(color='g', linestyle = '--', ax=axarr[1])
axarr[1].set_title('4 month MA & 2X4 month MA')

ts['item_cnt_day'].plot(color='b', linestyle = '-', ax=axarr[2])
MA3.plot(color='k', linestyle = '-', ax=axarr[2])
ThreeXMA3.plot(color='k', linestyle = '--', ax=axarr[2])
plt.xticks(rotation=45)
axarr[2].set_title('3 month MA & 3X 3month MA')


# In[ ]:


sales['date']= pd.to_datetime(sales['date'])


# In[ ]:


sales= sales.set_index('date')


# In[ ]:


quaterly = sales['item_cnt_day'].resample('Q')
quaterly_mean = quaterly.mean()


# In[ ]:


type(quaterly_mean)


# In[ ]:


quaterly_mean.head()


# In[ ]:


semi = sales['item_cnt_day'].resample('SM')
semi_mean = semi.mean()


# In[ ]:


fig = plt.figure(figsize=(5.5, 5.5))
ax = fig.add_subplot(1,1,1)

semi_mean.plot(ax=ax, color='b')
quaterly_mean.plot(ax=ax, color='r')

ax.set_title('semi-monthly sales (blue) & quaterly Mean (red)')
ax.set_xlabel('monthly')
ax.set_ylabel('sales')


# In[ ]:


quater_mean = ts['item_cnt_day'].rolling(5).mean()
quater_mean.dropna(inplace= True)


# In[ ]:


quater_std = ts['item_cnt_day'].rolling(5).std()


# In[ ]:


quater_std.dropna(inplace= True)


# In[ ]:


fig = plt.figure(figsize=(5.5, 5.5))
ax = fig.add_subplot(1,1,1)

quater_mean.plot(ax=ax, color='b')
quater_std.plot(ax=ax, color='r')

ax.set_title('quater statistics: Mean (blue) & Std. Dev. (red)')


# In[ ]:


adf_result = stattools.adfuller(quater_mean, autolag='AIC')
print('p-val of the ADF test in sales:', adf_result[1])


# In[ ]:


semi_mean.dropna(inplace= True)
quaterly_mean.dropna(inplace= True)


# In[ ]:


adf_result = stattools.adfuller(semi_mean, autolag='AIC')
print('p-val of the ADF test in sales:', adf_result[1])


# In[ ]:


adf_result = stattools.adfuller(quaterly_mean, autolag='AIC')
print('p-val of the ADF test in sales:', adf_result[1])


# In[ ]:


adf_result = stattools.adfuller(ts['item_cnt_day'], autolag='AIC')
print('p-val of the ADF test in sales:', adf_result[1])


# In[ ]:


plt.figure(figsize=(5.5, 5.5))
plot_acf(semi_mean, lags=45)


# In[ ]:


plt.figure(figsize=(5.5, 5.5))
plot_pacf(semi_mean, lags=45)


# Seasonal adjustment

# In[ ]:


MA4 = ts['item_cnt_day'].rolling(window=4).mean()
TwoXMA4 = MA4.rolling(window=2).mean()
TwoXMA4 = TwoXMA4.loc[~pd.isnull(TwoXMA4)]


# In[ ]:


fig = plt.figure(figsize=(5.5, 5.5))
ax = fig.add_subplot(1,1,1)
ts['item_cnt_day'].plot(ax=ax, color='b', linestyle='-')
TwoXMA4.plot(ax=ax, color='r', linestyle='-')
plt.xticks(rotation=60)
ax.set_title('monthly sales  and 2X4 quarter sales')


# In[ ]:


residuals = ts['item_cnt_day']-TwoXMA4
residuals = residuals.loc[~pd.isnull(residuals)]


# In[ ]:


fig = plt.figure(figsize=(5.5, 5.5))
ax = fig.add_subplot(1,1,1)
residuals.plot(ax=ax, color='b', linestyle='-')
plt.xticks(rotation=60)
ax.set_title('Residuals in Quaterly sales')


# In[ ]:


from pandas.plotting import autocorrelation_plot


# In[ ]:


fig = plt.figure(figsize=(5.5, 5.5))
ax = fig.add_subplot(2,2,2)
autocorrelation_plot(residuals, ax=ax)
ax.set_title('ACF of Residuals in Quaterly sales time series')


# In[ ]:


residuals_qtr_diff = residuals.diff(4)
residuals_qtr_diff = residuals_qtr_diff.loc[~pd.isnull(residuals_qtr_diff)]


# In[ ]:


fig = plt.figure(figsize=(5.5, 5.5))
ax = fig.add_subplot(1,1,1)
autocorrelation_plot(residuals_qtr_diff, ax=ax)
ax.set_title('ACF of Quaterly Differenced Residuals')


# In[ ]:


first_order_diff = ts['item_cnt_day'].diff(1)


# In[ ]:


fig, ax = plt.subplots(2, sharex=True)
fig.set_size_inches(5.5, 5.5)
ts['item_cnt_day'].plot(ax=ax[0], color='b')
ax[0].set_title('sales values during oct 2013-oct 2015')
first_order_diff.plot(ax=ax[1], color='r')
ax[1].set_title('First-order differences of sales values during oct 2013-oct 2015')


# In[ ]:


fig, ax = plt.subplots(2, sharex=True)
fig.set_size_inches(5.5, 5.5)
autocorrelation_plot(ts['item_cnt_day'], color='b', ax=ax[0])
ax[0].set_title('ACF of monthly sales values')
autocorrelation_plot(first_order_diff.iloc[1:], color='r', ax=ax[1])
ax[1].set_title('ACF of first differences of monthly sales values')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)


# 
# Let us obtain the confidence intervals, Ljung-Box Q-statistics and p-values for the original monthly sales values
# 

# In[ ]:


acf_sales, confint_sales, qstat_sales, pvalues_sales = stattools.acf(ts['item_cnt_day'],
                                                                 unbiased=True,
                                                                 nlags=20,
                                                                 qstat=True,
                                                                 alpha=0.05)


# 
# Let us check if at confidence level 95% (alpha=0.05) if the null hypothesis is rejected at any of the lags
# 

# In[ ]:


alpha = 0.05
for l, p_val in enumerate(pvalues_sales):
    if p_val > alpha:
        print('Null hypothesis is accepted at lag = {} for p-val = {}'.format(l, p_val))
    else:
        print('Null hypothesis is rejected at lag = {} for p-val = {}'.format(l, p_val))


# 
# The above results show statistically significant ACF in the original monthly sales values
# 

# 
# Let us obtain the confidence intervals, Ljung-Box Q-statistics and p-values for the differenced sales  values
# 

# In[ ]:


acf_first_diff, confint_first_diff, qstat_first_diff, pvalues_first_diff = stattools.acf(first_order_diff.iloc[1:],
                                                                                         unbiased=True,
                                                                                         nlags=20,
                                                                                         qstat=True,
                                                                                         alpha=0.05)


# 
# Let us check if at confidence level of 95% (alpha = 0.05) ,if the null hypothesis is rejected at any of the lags
# 

# In[ ]:


alpha = 0.05
for l, p_val in enumerate(pvalues_first_diff):
    if p_val > alpha:
        print('Null hypothesis is accepted at lag = {} for p-val = {}'.format(l, p_val))
    else:
        print('Null hypothesis is rejected at lag = {} for p-val = {}'.format(l, p_val))


# The above results show that ACF is essentially random in the differenced sales values for lags 0 to 7

# In[ ]:


from statsmodels.tsa import seasonal


# In[ ]:


decompose_model = seasonal.seasonal_decompose(ts.item_cnt_day.tolist(), freq=12, model='additive')


# In[ ]:


fig, axarr = plt.subplots(4, sharex=True)
fig.set_size_inches(5.5, 5.5)

ts['item_cnt_day'].plot(ax=axarr[0], color='b', linestyle='-')
axarr[0].set_title('Monthly sales')

pd.Series(data=decompose_model.trend, index=ts.index).plot(color='r', linestyle='-', ax=axarr[1])
axarr[1].set_title('Trend component in monthly sales')

pd.Series(data=decompose_model.seasonal, index=ts.index).plot(color='g', linestyle='-', ax=axarr[2]) 
axarr[2].set_title('Seasonal component in monthly sales')

pd.Series(data=decompose_model.resid, index=ts.index).plot(color='k', linestyle='-', ax=axarr[3])
axarr[3].set_title('Irregular variations in monthly sales')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
plt.xticks(rotation=10)


# In[ ]:


adf_result = stattools.adfuller(decompose_model.resid[np.where(np.isfinite(decompose_model.resid))[0]], autolag='AIC')


# In[ ]:


print('p-val of the ADF test on irregular variations in employment data:', adf_result[1])


# In[ ]:


decompose_model = seasonal.seasonal_decompose(ts.item_cnt_day.tolist(), freq=12, model='multiplicative') 


# In[ ]:


fig, axarr = plt.subplots(4, sharex=True)
fig.set_size_inches(5.5, 5.5)

ts['item_cnt_day'].plot(ax=axarr[0], color='b', linestyle='-')
axarr[0].set_title('Monthly sales')

axarr[1].plot(decompose_model.trend, color='r', linestyle='-')
axarr[1].set_title('Trend component in monthly sales')

axarr[2].plot(decompose_model.seasonal, color='g', linestyle='-')
axarr[2].set_title('Seasonal component in monthly sales')

axarr[3].plot(decompose_model.resid, color='k', linestyle='-')
axarr[3].set_title('Irregular variations in monthly sales')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
plt.xticks(rotation=10)


# In[ ]:


adf_result = stattools.adfuller(decompose_model.resid[np.where(np.isfinite(decompose_model.resid))[0]], autolag='AIC')


# In[ ]:


print('p-val of the ADF test on irregular variations in sales data:', adf_result[1])


# In[ ]:


def double_exp_smoothing(x, alpha, beta):
    yhat = [x[0]]
    for t in range(1, len(x)):
        if t==1:
            F, T= x[0], x[1] - x[0]
        F_n_1, F = F, alpha*x[t] + (1-alpha)*(F+T)
        T=beta*(F-F_n_1)+(1-beta)*T
        yhat.append(F+T)
    return yhat


# In[ ]:


ts['DEF00'] = double_exp_smoothing(ts['item_cnt_day'],0, 0)
ts['DEF01'] = double_exp_smoothing(ts['item_cnt_day'],0, 1)
ts['DEF10'] = double_exp_smoothing(ts['item_cnt_day'],1, 0)
ts['DEF11'] = double_exp_smoothing(ts['item_cnt_day'],1, 1)


# In[ ]:


fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(hspace=.5, wspace=.5)

ax = fig.add_subplot(2,2,1)
ts['item_cnt_day'].plot(color='b', linestyle = '-', ax=ax)
ts['DEF00'].plot(color='r', linestyle = '--', ax=ax)
ax.set_title('Alpha 0 and Beta 0')

ax = fig.add_subplot(2,2,2)
ts['item_cnt_day'].plot(color='b', linestyle = '-', ax=ax)
ts['DEF01'].plot(color='r', linestyle = '--', ax=ax)
ax.set_title('Alpha 0 and Beta 1')

ax = fig.add_subplot(2,2,3)
ts['item_cnt_day'].plot(color='b', linestyle = '-', ax=ax)
ts['DEF10'].plot(color='r', linestyle = '--', ax=ax)
ax.set_title('TES: alpha=1, beta=0')

ax = fig.add_subplot(2,2,4)
ts['item_cnt_day'].plot(color='b', linestyle = '-', ax=ax)
ts['DEF11'].plot(color='r', linestyle = '--', ax=ax)
ax.set_title('TES: alpha=1, beta=1')


# In[ ]:


def single_exp_smoothing(x, alpha):
    F = [x[0]]
    for t in range(1, len(x)):
        F.append(alpha * x[t] + (1 - alpha) * F[t-1])
    return F
ts['Single_Exponential_Forecast'] = single_exp_smoothing(ts['item_cnt_day'], 1)


# In[ ]:


fig = plt.figure(figsize=(5.5, 5.5))
ax = fig.add_subplot(2,1,1)
fig.subplots_adjust(hspace=.5)
ts['Single_Exponential_Forecast'].plot(ax=ax)
ax.set_title('Single Exponential Smoothing')
ax = fig.add_subplot(2,1,2)
ts['DEF10'].plot(ax=ax, color='r')
ax.set_title('Double Smoothing Forecast')


# In[ ]:


f, axarr = plt.subplots(2, sharex=True)
f.set_size_inches(5.5, 5.5)
ts['item_cnt_day'].plot(color='b', linestyle = '-', ax=axarr[0])
ts['DEF10'].plot(color='r', linestyle = '--', ax=axarr[0])
axarr[0].set_title('Actual Vs Double Smoothing Forecasting')

ts['item_cnt_day'].plot(color='b', linestyle = '-', ax=axarr[1])
ts['Single_Exponential_Forecast'].plot(color='r', linestyle = '--', ax=axarr[1])
axarr[1].set_title('Actual Vs Single Smoothing Forecasting')


# In[ ]:


import statsmodels.tsa.api as smtsa  


# In[ ]:


ar1model = smtsa.ARMA(ts['item_cnt_day'].tolist(), order=(1, 0))
ar1=ar1model.fit(maxlag=30, method='mle', trend='nc')
ar1.summary()


# In[ ]:


arma_obj = smtsa.ARMA(ts['item_cnt_day'].tolist(), order=(1, 1)).fit(maxlag=20, method='mle', trend='nc') 


# In[ ]:


arima_obj = ARIMA(ts['item_cnt_day'].tolist(), order=(0,2,1))
arima_obj_fit = arima_obj.fit(disp=0)
arima_obj_fit.summary()


# In[ ]:


pred=np.append([0,0],arima_obj_fit.fittedvalues.tolist())
ts['ARIMA']=pred
diffval=np.append([0,0], arima_obj_fit.resid+arima_obj_fit.fittedvalues)
ts['diffval']=diffval


# In[ ]:


x = sm.qqplot(arima_obj_fit.resid, line='s')


# In[ ]:


f, axarr = plt.subplots(1, sharex=True)
f.set_size_inches(5.5, 5.5)
ts['diffval'].iloc[2:].plot(color='b', linestyle = '-', ax=axarr)
ts['ARIMA'].iloc[2:].plot(color='r', linestyle = '--', ax=axarr)
axarr.set_title('ARIMA(0,2,1)')
plt.xlabel('Index')
plt.ylabel('Closing')


# In[ ]:


mod = sm.tsa.statespace.SARIMAX(ts['item_cnt_day'], trend='n')
sarimax= mod.fit()
sarimax.summary()


# In[ ]:


f, err, ci=arima_obj_fit.forecast(30)
plt.plot(f)
plt.plot(ci)
plt.xlabel('Forecasting Index')
plt.ylabel('Forecasted value')


# In[ ]:


aicVal=[]
for ari in range(1, 3):
    for maj in range(0,3):
        arma_obj = smtsa.ARMA(ts.item_cnt_day.tolist(), order=(ari, maj)).fit(maxlag=30, method='mle', trend='nc') 
        aicVal.append([ari, maj, arma_obj.aic])


# In[ ]:


aicVal


# In[ ]:


arma_obj_fin = smtsa.ARMA(ts.item_cnt_day.tolist(), order=(1, 1)).fit(maxlag=30, method='mle', trend='nc') 
ts['ARMA']=arma_obj_fin.predict()
arma_obj_fin.summary()


# In[ ]:


f, axarr = plt.subplots(1, sharex=True)
f.set_size_inches(5.5, 5.5)
ts['item_cnt_day'].iloc[1:].plot(color='b', linestyle = '-', ax=axarr)
ts['ARMA'].iloc[1:].plot(color='r', linestyle = '--', ax=axarr)
axarr.set_title('ARMA(1,1)')
plt.xlabel('Index')
plt.ylabel('sales')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




