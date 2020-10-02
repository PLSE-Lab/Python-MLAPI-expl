#!/usr/bin/env python
# coding: utf-8

# # State Space Modelling for M5 Forecasting Accuracy

# This notebook was inspired by this discussion post: [Any "time series" model < 0.7?](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/147961). In this notebook I attempt to break 0.7 using only [state space models](http://www.scholarpedia.org/article/State_space_model), rather than with neural networks.

# There are various notebooks that have used LightGBM with categorical variables to model seasonality and trend in sales, e.g.:
# * [for_Japanese_beginner(with WRMSSE in LGBM))](https://www.kaggle.com/girmdshinsei/for-japanese-beginner-with-wrmsse-in-lgbm)
# * [Very fst Model](https://www.kaggle.com/ragnar123/very-fst-model)
# * [m5-baseline](https://www.kaggle.com/harupy/m5-baseline)
# * [M5 - Simple FE](https://www.kaggle.com/kyakovlev/m5-simple-fe)
# 
# These use sklearn LabelEncoder variously for year, quarter-in-year, month-in-year, week-in-year, week-in-month, day-in-month, and day-of-week. (Interestingly, all these notebooks treat the resultant labels as numeric rather than categorical in LightGBM, but that is a matter for another day.)

# My own investigation suggests there are at least three seasonalities within the sales data: month-in-year, day-in-month, and day-of-week. However, adding week-in-year improves LightGBM performance; week-in-year and month-in-year could perhaps be better modelled together as day-in-year.
# 
# There are several ways to deal with multiple seasonality in time series forecasting (whether we are using state space models or neural networks) each with their drawbacks:
# * Exponential smoothing with multiple seasonalities incorporated directly (i.e., Holt-Winters method). This approach was taken by the winner of the M4 competition as a precursor step to modelling and forecasting with a neural network, see [here](https://www.sciencedirect.com/science/article/pii/S0169207019301153). That work was done in R; Python's Holt-Winters only allows a single seasonality.
# * [TBATS](https://medium.com/intive-developers/forecasting-time-series-with-multiple-seasonalities-using-tbats-in-python-398a00ac0e8a): this is too computationally expensive for automatically modelling and forecasting 30490 series (even offline on a faster machine).
# * SARIMA (seasonal auto-regressive integrated moving average) allows modelling a single seasonality only, but one can encode the other seasonalities using Fourier terms as exogenous variables and employ SARIMAX, see the TBATS link above for a comparison of the two methods. I have found that trying to do all the computation (including the other exogenous variables that we have) in one SARIMAX takes far too long (~72 hours for all 30490 series on dual-threaded quad core @ 2.80 GHz).

# In this notebook I instead take a step-wise approach to trend and seasonality modelling and forecasting, using [differencing](https://machinelearningmastery.com/remove-trends-seasonality-difference-transform-python/), seasonal decomposition, ARMA, SARMA, ARMAX, and [Holt-Winters](https://machinelearningmastery.com/exponential-smoothing-for-time-series-forecasting-in-python/) for variety. Note that [ARMA](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model), SARMA, and ARMAX are all special cases of SARIMAX and the same function is used to estimate all of them. ARMA assumes the series is weakly stationary, i.e., the moments (mean, variance, skewness, kurtosis, etc.) are constant over time and the error process can be modelled as an **A**uto-**R**egressive (AR) **M**oving **A**verage (MA) process. SARMA adds a single **S**easonality term, ARMAX adds e**X**ogenous variables (in our case dummies for `event_name_1` and the binary SNAP features).

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


from tqdm import tqdm_notebook as tqdm
import gc
import pickle
from datetime import datetime, timedelta
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 10)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


input_path = '/kaggle/input/m5-forecasting-accuracy/'


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.
                      format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


# read data files
calendar = pd.read_csv(input_path + '/calendar.csv')
calendar = reduce_mem_usage(calendar)
sell_prices = pd.read_csv(input_path + '/sell_prices.csv')
sell_prices = reduce_mem_usage(sell_prices)
sales_train_val = pd.read_csv(input_path + '/sales_train_validation.csv')
sales_train_val = reduce_mem_usage(sales_train_val)
sample_submission = pd.read_csv(input_path + '/sample_submission.csv')


# In[ ]:


NUM_ITEMS = sales_train_val.shape[0]  # 30490
DAYS_PRED = sample_submission.shape[1] - 1  # 28


# In[ ]:


# keep a multiple of DAYS_PRED days (columns) for convenience
ncols = sales_train_val.shape[1] - sales_train_val.shape[1] % DAYS_PRED
sales_train_val = sales_train_val.iloc[:, -ncols:]


# In[ ]:


# Take the transpose so that we have one day for each row, and 30490 items' sales as columns
sales_train_val = sales_train_val.T


# In[ ]:


# plot total sales
plt.style.use('classic')
salesTotal = np.sum(sales_train_val, axis=1)
plt.plot(salesTotal)
plt.xticks(salesTotal.index[np.arange(1, len(salesTotal), step=280)])
plt.show()


# In[ ]:


# difference the series to remove trend
lag_sales_train_val = sales_train_val.iloc[:-DAYS_PRED, :]
lag_sales_train_val_test = sales_train_val.iloc[-DAYS_PRED:, :]
sales_train_val = sales_train_val.iloc[DAYS_PRED:, :]
sales_train_val = sales_train_val - lag_sales_train_val.values


# In[ ]:


# plot total differenced sales
plt.style.use('classic')
salesTotalI = np.sum(sales_train_val, axis=1)
plt.plot(salesTotalI)
plt.xticks(salesTotalI.index[np.arange(1, len(salesTotalI), step=280)])
plt.show()


# # Seasonal Decomposition and ARMA Forecasting
# statsmodels help pages:
# * [seasonal_decompose](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html)
# * [SARIMAX](https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
# 
# The idea is to use `seasonal_decompose` to get the mean (which may be fluctuating) and seasonality (which is periodic) of the training data. The mean is then forecast by an ARMA(3,2) model using `SARIMAX` and the seasonality is forecast by simple repetition. This is quite slow and gives poor dynamic forecasts if we enforce a day-in-year seasonality (`period=365`) so we down-sample to 28-day blocks and make a one-step-ahead forecast to cover the whole test period. The downside of this is that we forecast a constant mean for the test period, whereas it may be trending. Hopefully the subsequent steps will compensate for this.

# In[ ]:


# remove yearly seasonality by down-sampling, decomposing, and forecasting with ARMA
def yearly_season(ts):
    ts = pd.Series(ts)
    ts.index = pd.to_datetime(ts.index)
    ts.index.freq = 'N'
    ts28 = ts.resample('28N').sum()
    decomp = seasonal_decompose(ts28.values, 'additive', period=13, extrapolate_trend='freq')
    try:
        # the order here was chosen offline using autoarima for the total sales, 
        # we could use autoarima for each series but that would take a lot longer
        mod = SARIMAX(decomp.trend, order=(3, 0, 2))
        # increase maxiter and use 'powell' here as it is gradient-free and the default method often doesn't converge for these data
        res = mod.fit(disp=False, cov_type='robust', maxiter=200, method='powell')
        predict = res.get_prediction(start=0, end=len(ts28))
        predicted_mean = np.repeat(predict.predicted_mean, 28) / 28
    except:
        predicted_mean = np.repeat(np.mean(ts28), len(ts28)+1)
        predicted_mean = np.repeat(predicted_mean, 28) / 28
    # extend seasonal component by one period
    seasonal = decomp.seasonal
    seasonal = np.append(seasonal, seasonal[-13])
    seasonal = np.repeat(seasonal, 28) / 28
    predicted_mean += seasonal
    return predicted_mean


# In[ ]:


# Run the parallel job for yearly seasonal - uncomment this if you edit yearly_seasonal()
#fit_forecast1 = Parallel(n_jobs=-1)(delayed(yearly_season)(
#    sales_train_val.iloc[:, i].values
#) for i in tqdm(range(NUM_ITEMS)))
#file = open('yearly_seasonal2.pkl', 'wb')
#pickle.dump(fit_forecast1, file)
#file.close()


# In[ ]:


# load pre-calculate yearly seasonal fit & forecast - comment out if you are editing yearly_seasonal()
file = open('/kaggle/input/seasonalities2/seasonality2/yearly_seasonal2.pkl', 'rb')
fit_forecast1 = pickle.load(file)
file.close()


# In[ ]:


# get yearly seasonal in same format as sales_train_val
fit_forecast1 = np.concatenate(fit_forecast1)
fit_forecast1 = fit_forecast1.reshape(-1, NUM_ITEMS, order='F')
fit1 = fit_forecast1[:-DAYS_PRED, :]
forecast1 = fit_forecast1[-DAYS_PRED:, :]
del fit_forecast1 ; gc.collect()


# In[ ]:


# plot total sales and total fit & forecast
plt.plot(salesTotalI, color='blue')
fitTotal = pd.Series(np.sum(fit1, axis=1))
fitTotal.index = salesTotalI.index
plt.plot(fitTotal, color='orange')
forecastTotal = pd.Series(np.sum(forecast1, axis=1))
forecastTotal.index = ['d_' + str(i) for i in range(1914,1914+28)]
plt.plot(forecastTotal, color='red')
plt.xticks(salesTotalI.index[np.arange(1, len(salesTotalI), step=280)])
plt.show()


# In[ ]:


# remove trend and first seasonal component from sales
sales_train_val = sales_train_val - fit1


# In[ ]:


# remove first 28-day period as we have no fit
sales_train_val = sales_train_val.iloc[28:, :]


# In[ ]:


# plot transformed series
sales1Total = np.sum(sales_train_val, axis=1)
plt.plot(sales1Total)
plt.xticks(sales1Total.index[np.arange(1, len(sales1Total), step=280)])
plt.show()


# # SARMA for Day-of-Week Seasonality

# In[ ]:


# SARMA with day-of-week seasonality
def dow_season(ts):
    try:
        mod = SARIMAX(ts, order=(1, 0, 1), seasonal_order=(1, 1, 1, 7))
        res = mod.fit(disp=False, cov_type='robust', maxiter=200, method='powell')
        predict = res.get_prediction(start=0, end=len(ts) + 27)
        predicted_mean = predict.predicted_mean
    except:
        predicted_mean = np.repeat(np.mean(ts), len(ts) + 28)
    return predicted_mean


# In[ ]:


# run the parallel job - uncomment this if you edit dow_season()
#fit_forecast2 = Parallel(n_jobs=-1)(delayed(dow_season)(
#    sales_train_val.iloc[:, i].values
#) for i in tqdm(range(NUM_ITEMS)))
#file = open('dow_seasonal2.pkl', 'wb')
#pickle.dump(fit_forecast2, file)
#file.close()


# In[ ]:


# load pre-calculated dow seasonality - comment this out if you are editing dow_season()
file = open('/kaggle/input/seasonalities2/seasonality2/dow_seasonal2.pkl', 'rb')
fit_forecast2 = pickle.load(file)
file.close()


# In[ ]:


# get day-of-week seasonal in same format as sales_train_val 
fit_forecast2 = np.concatenate(fit_forecast2)
fit_forecast2 = fit_forecast2.reshape(-1, NUM_ITEMS, order='F')
fit2 = fit_forecast2[:-DAYS_PRED, :]
forecast2 = fit_forecast2[-DAYS_PRED:, :]
del fit_forecast2 ; gc.collect()


# In[ ]:


# plot transformed and fit
sales1Total = np.sum(sales_train_val, axis=1)
plt.plot(sales1Total, color='blue')
fit2Total = pd.Series(np.sum(fit2, axis=1))
fit2Total.index = sales1Total.index
plt.plot(fit2Total, color='orange')
forecast2Total = pd.Series(np.sum(forecast2, axis=1))
forecast2Total.index = ['d_' + str(i) for i in range(1914,1914+28)]
plt.plot(forecast2Total, color='red')
plt.xticks(sales1Total.index[np.arange(1, len(sales1Total), step=280)])
plt.show()


# In[ ]:


# remove second seasonal component from sales
sales_train_val = sales_train_val - fit2


# In[ ]:


# plot transformed series
sales2Total = np.sum(sales_train_val, axis=1)
plt.plot(sales2Total)
plt.xticks(sales2Total.index[np.arange(1, len(sales2Total), step=280)])
plt.show()


# In[ ]:


# it is not obvious from the plot, but because of construction, we have no fit for the first day
sales_train_val = sales_train_val.iloc[1:, :]


# # ARMAX for Events and SNAP

# In[ ]:


# there are obvious spikes for Thanksgiving and 25th December, and presumably other events, so we model these with dummies and ARMAX
# calendar: keep needed rows & columns
calendar.drop(['wm_yr_wk', 'weekday', 'd', 'event_type_1', 'event_type_2'], axis=1, inplace=True)
calendar = calendar.iloc[-(sales_train_val.shape[0] + DAYS_PRED * 2):, :]
calendar['date'] = pd.to_datetime(calendar['date'])

# one-hot-encode event_name_1
event_name_1_ohe = pd.get_dummies(calendar['event_name_1'], dummy_na=True, dtype=np.int8)
calendar.drop(['event_name_1'], axis=1, inplace=True)
calendar = pd.concat([calendar, event_name_1_ohe], axis=1)

# one-hot-encode event_name_2
#event_name_2_ohe = pd.get_dummies(calendar['event_name_2'], dummy_na=True, dtype=np.int8)
#calendar.drop(['event_name_2'], axis=1, inplace=True)
#calendar = pd.concat([calendar, event_name_2_ohe], axis=1)


# In[ ]:


# we can also add Fourier terms (first and second order) to model the day-in-month seasonality
# but this takes too long to compute the ARMAX
#fourier = pd.DataFrame({'date': calendar['date'].unique()})
#fourier = fourier.set_index(pd.PeriodIndex(fourier['date'], freq='D'))
#fourier['sin30'] = np.sin(2 * np.pi * fourier.index.day / 30.4375)
#fourier['cos30'] = np.cos(2 * np.pi * fourier.index.day / 30.4375)
#fourier['sin30_2'] = np.sin(4 * np.pi * fourier.index.day / 30.4375)
#fourier['cos30_2'] = np.cos(4 * np.pi * fourier.index.day / 30.4375)
#fourier.drop(columns=['date'], inplace=True)
calendar.drop(columns=['date'], inplace=True)

# merge by concat the calendar features with fourier
#fourier.index = calendar.index
#calendar = pd.concat([calendar, fourier], axis=1)
#del fourier


# In[ ]:


# set calendar index to align with sales_train_val
calendar.index = ['d_' + str(i) for i in range(67, 1914 + DAYS_PRED * 2)]


# In[ ]:


# ARMAX(1,1) - exogenous variables are one-hot-encoded event_name_1 and binary SNAP features
def events_snap(ts):
    try:
        mod = SARIMAX(ts, order=(1, 0, 1), 
                      exog=calendar.drop(['wday', 'month', 'year', 'event_name_2'], axis=1).iloc[:-(DAYS_PRED * 2), :].values)
        res = mod.fit(disp=False, cov_type='robust', maxiter=200, method='powell')
        predict = res.get_prediction(start=0, end=len(ts) + 27, 
                                     exog=calendar.drop(['wday', 'month', 'year', 'event_name_2'], axis=1).\
                                                   iloc[-(DAYS_PRED * 2):-DAYS_PRED, :].values)
        predicted_mean = predict.predicted_mean
    except:
        predicted_mean = np.repeat(np.mean(ts), len(ts) + 28)
    return predicted_mean


# In[ ]:


# run the parallel job - this has ETA of ~70 hours on Kaggle so I have precomputed it
#fit_forecast3 = Parallel(n_jobs=-1)(delayed(events_snap)(
#    sales_train_val.iloc[:, i].values
#) for i in tqdm(range(NUM_ITEMS)))
#file = open('events_snap2.pkl', 'wb')
#pickle.dump(fit_forecast3, file)
#file.close()


# In[ ]:


# load pre-calculated events, snap fit & forecast
file = open('/kaggle/input/seasonalities2/seasonality2/events_snap2.pkl', 'rb')
fit_forecast3 = pickle.load(file)
file.close()


# In[ ]:


# get events, snap fit & forecast in same format as sales_train_val 
fit_forecast3 = np.concatenate(fit_forecast3)
fit_forecast3 = fit_forecast3.reshape(-1, NUM_ITEMS, order='F')
fit3 = fit_forecast3[:-DAYS_PRED, :]
forecast3 = fit_forecast3[-DAYS_PRED:, :]
del fit_forecast3 ; gc.collect()


# In[ ]:


# plot transformed
plt.plot(sales2Total, color='blue')
fit3Total = pd.Series(np.sum(fit3, axis=1))
fit3Total.index = sales2Total.index[1:]
plt.plot(fit3Total, color='orange')
forecast3Total = pd.Series(np.sum(forecast3, axis=1))
forecast3Total.index = ['d_' + str(i) for i in range(1914,1914+28)]
plt.plot(forecast3Total, color='red')
plt.xticks(sales2Total.index[np.arange(1, len(sales2Total), step=280)])
plt.show()


# In[ ]:


# remove trend and first seasonal component from sales
sales_train_val = sales_train_val - fit3


# In[ ]:


# plot transformed series
sales3Total = np.sum(sales_train_val, axis=1)
plt.plot(sales3Total)
plt.xticks(sales3Total.index[np.arange(1, len(sales3Total), step=280)])
plt.show()


# # Holt-Winters for Day-in-Month Seasonality

# In[ ]:


# model day-in-month seasonality with Holt-Winters
def dayinmonth(ts):
    mod = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=30)
    res = mod.fit()
    fit = res.fittedvalues
    forecast = res.forecast(DAYS_PRED)
    fit_forecast = np.append(fit, forecast)
    return fit_forecast


# In[ ]:


# run the parallel job 
#fit_forecast4 = Parallel(n_jobs=-1)(delayed(dayinmonth)(
#    sales_train_val.iloc[:, i].values
#) for i in tqdm(range(NUM_ITEMS)))
#file = open('dayinmonth.pkl', 'wb')
#pickle.dump(fit_forecast4, file)
#file.close()


# In[ ]:


# load pre-calculated day-in-month fit & forecast
file = open('/kaggle/input/seasonalities2/seasonality2/dayinmonth2.pkl', 'rb')
fit_forecast4 = pickle.load(file)
file.close()


# In[ ]:


# get day-in-month in same format as sales_train_val 
fit_forecast4 = np.concatenate(fit_forecast4)
fit_forecast4 = fit_forecast4.reshape(-1, NUM_ITEMS, order='F')
fit4 = fit_forecast4[:-DAYS_PRED, :]
forecast4 = fit_forecast4[-DAYS_PRED:, :]
del fit_forecast4 ; gc.collect()


# In[ ]:


# plot transformed
plt.plot(sales3Total, color='blue')
fit4Total = pd.Series(np.sum(fit4, axis=1))
fit4Total.index = sales3Total.index
plt.plot(fit4Total, color='orange')
forecast4Total = pd.Series(np.sum(forecast4, axis=1))
forecast4Total.index = ['d_' + str(i) for i in range(1914,1914+28)]
plt.plot(forecast4Total, color='red')
plt.xticks(sales2Total.index[np.arange(1, len(sales2Total), step=280)])
plt.show()


# # Final Forecast & Submission

# In[ ]:


# combine forecasts
preds = forecast1 + forecast2 + forecast3 + forecast4 + lag_sales_train_val_test
preds = pd.DataFrame(data=preds)
preds.index =  ['d_' + str(i) for i in range(1914,1914+28)]


# In[ ]:


# sanity check
predsTotal = preds.sum(axis=1)
predsTotal.plot()


# In[ ]:


# submission
submission = preds.T
submission = pd.concat((submission, submission), ignore_index=True)

idColumn = sample_submission[["id"]]
submission[["id"]] = idColumn  

cols = list(submission.columns)
cols = cols[-1:] + cols[:-1]
submission = submission[cols]
colsdeneme = ["id"] + [f"F{i}" for i in range (1,29)]
submission.columns = colsdeneme

submission.to_csv('submission.csv', index=False)


# # Notes
# * SARIMAX may display a warning: `UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.` However, the default starting parameters are zeros anyway, so I don't know what effect, if any, this has. I believe convergence is more important.
# * SARIMAX may also display a warning: `UserWarning: Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.` Ditto.
