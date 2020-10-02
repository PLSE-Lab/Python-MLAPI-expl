#!/usr/bin/env python
# coding: utf-8

# # Brief comparison between Time Series methods to predict daily confirmed cases of Coronavirus

# My personal goal in this notebook was to learn about Time Series and use it to help forecasting the possible evolution of the Coronavirus outbreak.
# This is not intended to be a comprehensive study, but more a beginner's introduction to apply time series.
# 
# The methods compared are classical time series like Autoregressive Integrated Moving Average (ARIMA)  and Holt-Winters Exponential Smoothing, machine learning  methods like several types of Long Short Term Memory (LSTM) networks, and Facebook Prophet as well.
# 
# ARIMA requires data sets to be stationary, and most data related to the Coronavirus outbreak has shown an exponential growth, though it has started to reduce the slope in the trend in some places.
# This exercise demonstrates a comparison between different models using root mean squared error to measure the model performance after using Walk-forward validation testing.
# 
# The dataset used is corona-virus-report\covid_19_clean_complete.csv from https://www.kaggle.com.
# 
# Before getting into the forecast analysis, there are some visual comparisons between countries in features like "Confirmed" (accumulated number of confirmed cases), "New_Confirmed" (daily number of confirmed cases), "Deaths" (accumulated number of deaths), "New_Deaths" (daily number of deaths), "Recovered" (accumulated number of recovered cases), "New_ Recovered" (daily number of recovered cases), "Active" (difference between the number of confirmed cases and the number of deaths and recovered cases).
# The last feature is CFR (Case fatality rate) which is the ratio between deaths to confirmed cases. 
# 
# Important references:
# 
# - Italy space time & spreading of Covid19
# https://www.kaggle.com/lumierebatalong/italy-space-time-spreading-of-covid19
# 
# - How to Grid Search Triple Exponential Smoothing for Time Series Forecasting in Python
# https://machinelearningmastery.com/how-to-grid-search-triple-exponential-smoothing-for-time-series-forecasting-in-python/
# 
# - How to Develop LSTM Models for Time Series Forecasting
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
# 
# - How To Backtest Machine Learning Models for Time Series Forecasting
# https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/
# 
# - Time Series Forecast Case Study with Python: Monthly Armed Robberies in Boston
# https://machinelearningmastery.com/time-series-forecast-case-study-python-monthly-armed-robberies-boston/
# 

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


# #### Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import plotly.offline as py
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from math import exp,log
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import acf, pacf


# In[ ]:


py.init_notebook_mode()
pd.plotting.register_matplotlib_converters
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")
#delta_positive forces values to be positive in case of 
#log transformations 
delta_positive=2


# #### Load and prepare data

# In[ ]:


data_file = "/kaggle/input/corona-virus-report/covid_19_clean_complete.csv"
df_data = pd.read_csv(data_file,na_filter=False)


# In[ ]:


# Calculate features
# New_Confirmed,New_Deaths, and New_Recovered are calculated 
# by substracting the values from the previous day per country

df_data['Active'] = df_data['Confirmed'] - df_data['Deaths'] - df_data['Recovered']
df_data['New_Confirmed'] = 0
df_data['New_Deaths'] = 0
df_data['New_Recovered'] = 0

arr_data = df_data[["Confirmed","Deaths","Recovered"]].values
arr_data =  np.concatenate(                           (df_data.index.values.reshape(-1,1),                             arr_data), axis=1)

arr_new_data = df_data[["New_Confirmed","New_Deaths","New_Recovered"]].values
arr_new_data =  np.concatenate(                           (df_data.index.values.reshape(-1,1),                             arr_new_data), axis=1)

num_days = len(np.unique(df_data["Date"]))
num_countries_provinces = int(arr_data.shape[0]/num_days)

for ix in range(len(arr_new_data)):
    if ix < num_countries_provinces:
        for col in range(1,4):
            arr_new_data[ix,col]=arr_data[ix,col]
    else:
        for col in range(1,4):
            arr_new_data[ix,col]=            arr_data[ix,col]-arr_data[ix-num_countries_provinces,col]
            
df_data["New_Confirmed"] = arr_new_data[:,1]
df_data["New_Deaths"] = arr_new_data[:,2]
df_data["New_Recovered"] = arr_new_data[:,3]


# In[ ]:


# case_fatality_rate = deaths/confirmed
arr_data_Confirmed = df_data[["Confirmed"]].values
arr_data_Deaths = df_data[["Deaths"]].values
case_fatality_rate = arr_data_Deaths/arr_data_Confirmed*100
where_are_NaNs = np.isnan(case_fatality_rate)
case_fatality_rate[where_are_NaNs] = 0.
df_data["CFR"] = case_fatality_rate


# In[ ]:


df_data['Date'] = pd.to_datetime(df_data['Date'])
df_data = df_data.set_index('Date')


# In[ ]:


df_data


# #### Create a specific dataframe for a particular country

# In[ ]:


country = "Mexico"
df_country = df_data[df_data["Country/Region"]==country]
df_country


# #### Visualization

# In[ ]:


def plot_data(title, data, labels):
    fig = go.Figure(layout_title_text=title)
    for (datum, label) in zip(data,labels):
        fig.add_trace(go.Scatter(x=datum.index,
                                 y=datum.values,
                      mode='lines+markers',
                      name=label))
    fig.show()
    return(fig)

def plot_data_list(title, data, labels):
    fig = go.Figure(layout_title_text=title)
    for (datum, label) in zip(data,labels):
        fig.add_trace(go.Scatter(x=np.array(range(len(datum))),
                                 y=datum,
                      mode='lines+markers',
                      name=label))
    fig.show()

fig = plot_data(country,
          [df_country["Confirmed"],df_country["Deaths"],\
           df_country["Recovered"],df_country["Active"]], 
          ["Confirmed", "Deaths", "Recovered", "Active"])


# In[ ]:


fig = plot_data(country,
          [df_country["Deaths"],df_country["New_Deaths"]], 
          ["Deaths","New_Deaths"])
fig = plot_data(country,
          [df_country["Confirmed"],df_country["New_Confirmed"]], 
          ["Confirmed","New_Confirmed"])
fig = plot_data(country + " Active",
          [df_country["Active"]], 
          ["Active"])
fig = plot_data(country + " Case Fatality Rate",
          [df_country["CFR"]], 
          ["CFR"])


# #### Create helper functions

# In[ ]:


def get_initial_date(country, type_case):
    try:
        initial_date = df_data[(df_data[type_case]>0) &                                (df_data["Country/Region"]==country)].index[0]
    except:
        initial_date = None
    return(initial_date)

def list_of_positive_cases (country, type_case):
    initial_date = get_initial_date(country, type_case)
    if initial_date is not None:
        list_positive_cases = df_data[(df_data.index>=initial_date) &                 (df_data["Country/Region"]==country)].        groupby(["Date"])[type_case].        agg('sum').values
    else:
        list_positive_cases = []
    return(list_positive_cases)

def get_rmse(var1, var2):
    return np.sqrt(((var1-var2) ** 2).mean())

def plot_countries_comparision(countries_to_compare,type_case):
    positive_cases_daily = [list_of_positive_cases(x,type_case)                              for x in countries_to_compare]
    plot_data_list("Number of " + type_case + " since first case in each country",
                   positive_cases_daily,
                   countries_to_compare)

    for i in range(len(countries_to_compare)-1):
        for j in range(i+1,len(countries_to_compare)):
            min_days = np.min([len(positive_cases_daily[i]),                              len(positive_cases_daily[j])])
            rmse_2 = get_rmse(positive_cases_daily[i][:min_days],
                positive_cases_daily[j][:min_days])
            print("Mean daily difference (RMSE)" +                   countries_to_compare[i] + "-" + countries_to_compare[j],          '{:2.2f}'.format(rmse_2))
            
def get_similar_countries(base_country,type_case, n_countries=3):
    base_list_of_positive_cases = list_of_positive_cases(base_country,type_case)
    num_days_base_country = len(base_list_of_positive_cases)

    countries_to_compare = list(np.unique(df_data["Country/Region"].values))
    countries_to_compare.remove(base_country)

    positive_cases_daily = [list_of_positive_cases(x,type_case)                             for x in countries_to_compare]

    list_rmse = []
    for ix,country in enumerate(countries_to_compare):
        num_days_second_country = len(positive_cases_daily[ix])
        if num_days_base_country <= num_days_second_country:
            min_days = np.min([num_days_base_country,num_days_second_country])
            rmse_2 = get_rmse(base_list_of_positive_cases[:min_days],
                    positive_cases_daily[ix][:min_days])
        else:
            rmse_2 = np.inf
        list_rmse.append(rmse_2)

    top_n = [countries_to_compare[x] for x in np.argsort(list_rmse)[:n_countries]]
    return(top_n)


# #### Compare a country's metric to others with similar behavior 

# In[ ]:


base_country="Mexico"
type_case="New_Confirmed"
n_countries=3
countries_to_compare = get_similar_countries(base_country,type_case, n_countries)
countries_to_compare.append(base_country)
plot_countries_comparision(countries_to_compare,type_case)


# #### Open comparision between countries according to one feature

# In[ ]:


countries_to_compare = ["Belgium","Spain","Italy","Mexico", "New Zealand"]
type_case="New_Deaths"
plot_countries_comparision(countries_to_compare,type_case)


# ## Forecast Analysis

# ### FB Prophet

# In[ ]:


def create_forecast(country,type_case,periods):
    ds = df_data[df_data["Country/Region"]==country].index.date
    y = df_data[df_data["Country/Region"]==country][type_case].values
    df_forecast = pd.DataFrame()
    df_forecast["ds"] = ds
    df_forecast["y"] = y
    
    model = Prophet(changepoint_prior_scale=.05,                    interval_width=0.95)
    
    model.fit(df_forecast)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return(model,df_forecast,future,forecast)

def plot_forecast(model,forecast):
    fig = plot_plotly(model,forecast)
    py.iplot(fig)
    
def plot_forecast_components(model,forecast):
    fig = model.plot_components(forecast)
    
def cross_val_forecast(model,horizon,period,initial_training_period):
        df_cv = cross_validation(model=model,                                 horizon=horizon,                                period=period,                                initial=initial_training_period)
        print('Forecast cross validation ')
        print(df_cv.tail(5))

        df_p = performance_metrics(df_cv)
        print('Performance metrics')
        print(df_p.head(5))

        ufig = plot_cross_validation_metric(df_cv, metric='rmse')
        return(df_cv)


# #### Using Prophet to forecast a specific metric in a country

# In[ ]:


### Define a country and a metric to forecast
country,type_case,periods = "Mexico","New_Confirmed",10

model,df_forecast, future, forecast = create_forecast(country,type_case,periods)

plot_forecast(model,forecast)
plot_forecast_components(model,forecast)


# In[ ]:


### Show some statistics about the forecast evaluation
horizon = "10 days"
period = "1 days"
initial_training_period = "30 days"

df_cv = cross_val_forecast(model,horizon,period,initial_training_period)


# #### The root mean squared error (rmse) is in the 130's for one-day forecast using Prophet. This means that there is a mean difference of around 130 cases between the one-day forecast and the actual values

# ### ARIMA

# In[ ]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from scipy.stats import boxcox

### Define some helper functions for stationary tests 
### and data transformation

# stationary test
def adfuller_test(dataset):  
    result = adfuller(dataset)
    print("adFuller Test")
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

# stationary test
def kpss_test(dataset):
    print("KPSS test")
    kpsstest = kpss(dataset, regression='c')
    kpss_output = pd.Series(kpsstest[0:3],                            index=['Test Statistic','p-value',                                   'Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print(kpss_output)
    
# differencing transformation
def difference_transf(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return (np.asarray(diff))

# boxcox transformation
def boxcox_transf(dataset):
    dataset = dataset+delta_positive
    transformed, lam = boxcox(dataset)
    if lam < -5:
        transformed, lam = dataset, 1
    return(transformed, lam)

# sqrt transformation
def sqrt_transf(dataset):
    return(np.sqrt(dataset))

# two types of moving average transformation
def diff_moving_avg(dataset, moving_avg):
    diff = dataset - moving_avg
    diff.dropna(inplace=True)
    return(diff)
def moving_average(dataset, n=3) :
    ret = np.cumsum(dataset, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# ewma transformation
def diff_ewma(dataset):
    expwighted_avg = dataset.ewm(halflife=7).mean()
    diff = dataset - expwighted_avg
    diff.dropna(inplace=True)
    return(diff)


# In[ ]:


### Define helper functions to show time series

def plot_series(data,initial_date,rolling_days=7):
    ts = pd.Series(data, index=pd.date_range(initial_date, 
                                         periods=len(data)))
    rolmean = ts.rolling(window=rolling_days).mean()
    rolstd = ts.rolling(window=rolling_days).std()

    fig = plt.figure(figsize=(12,6))
    orig = plt.plot(ts, color='blue',label=type_case)
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation '+country)
    plt.show(block=False)
    return(ts,rolmean)

#Source:
#https://machinelearningmastery.com/time-series-forecast-case-study-python-monthly-armed-robberies-boston/
def show_qqplot(dataset):
    plt.figure(1, figsize=(12,12))
    # line plot
    plt.subplot(311)
    plt.plot(dataset)
    # histogram
    plt.subplot(312)
    plt.hist(dataset)
    # q-q plot
    plt.subplot(313)
    qqplot(dataset, line='r', ax=plt.gca())
    plt.show()
    
def show_acf_pacf(dataset):
    max_lags=7
    plt.figure(figsize=(12,10))
    plt.subplot(211)
    plot_acf(dataset, ax=plt.gca(),lags=max_lags)
    plt.subplot(212)
    plot_pacf(dataset, ax=plt.gca(),lags=max_lags)
    plt.show()
    acf_res = acf(dataset,nlags=max_lags)
    pacf_res = pacf(dataset,nlags=max_lags)
    return(acf_res,pacf_res)


# #### Plot data distribution and stationarity tests

# In[ ]:


country, type_case = "Mexico", "New_Confirmed"
data = (list_of_positive_cases (country, type_case)).astype("float")
# initial_date that feature type_case started being non-zero
initial_date = get_initial_date(country, type_case)

ts,rolmean = plot_series(data,initial_date)
adfuller_test(data)
kpss_test(data)
show_qqplot(data)


# - Data is not stationary as expected. I will use a box_cox transformation to reduce skewness

# In[ ]:


dataset = data
boxcox_t,lam = boxcox_transf(dataset)
adfuller_test(boxcox_t)
kpss_test(boxcox_t)
show_qqplot(boxcox_t)


# - After several trials using different types of transformations, box_cox was the only one that provided consistent positive results

# #### Define a range to grid search p and q values in the ARIMA model

# In[ ]:


dataset = boxcox_t
acf_res, pacf_res = show_acf_pacf(dataset)
# arbitrary set of thresholds for acf in 0.7, and pacf in 0.2
q_values = [ix for ix,x in enumerate(np.abs(acf_res) > 0.7) if x]
p_values = [ix for ix,x in enumerate(np.abs(pacf_res) > 0.2) if x]


# In[ ]:


import warnings

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(dataset, arima_order):
    train_size = int(len(dataset) * 0.50)
    train, test = dataset[0:train_size], dataset[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=False)
        #yhat = model_fit.forecast()[0]
        yhat = model_fit.predict(start=0,end=0,typ='levels')
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = get_rmse(test, predictions)
    return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                print(p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s RMSE=%.3f' % (order,mse))
                except:
                    print('Error', order)
                    #continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    return(best_cfg)


# #### Perform grid search for p,d,q values

# In[ ]:


get_ipython().run_cell_magic('time', '', 'dataset = boxcox_t\n# evaluate parameters\nd_values = [0,1]\nwarnings.filterwarnings("ignore")\nprint(p_values,d_values,q_values)\nbest_cfg = evaluate_models(dataset, p_values, d_values, q_values)')


# #### Define helper functions

# In[ ]:


def profit_plot(model_fit,data,horizon=10):
    fig, ax = plt.subplots(figsize=(16,8))
    fig = model_fit.plot_predict(1, len(data)+horizon, ax=ax)
    plt.show()
    
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def show_results(y_true,yhat):
    mae = mean_absolute_error(y_true, y_pred)    
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = get_rmse(y_true, y_pred)

    print("Walk-forward validation, Tests = " + "{:d}".format(len(y_true)))
    print("mae = {:4.4f}".format(mae))
    print("rmse = {:4.4f}".format(rmse))
    print("r2 = {:4.4f}".format(r2))
    print("mape = {:4.4f}".format(mape))
    return(rmse)
    
def forecast_plot(forecast,actual,title,conf_int=None):
    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(forecast.index, forecast.values, label="Forecast",             color="red",marker=".",linestyle='dashed',linewidth=1,            alpha=.5)
    ax.plot(actual.index, actual.values, label="Actual",             color="blue",marker="o",linestyle="", alpha=.5)
    if conf_int is not None:
        fill_label = "{0:.0%} confidence interval".format(.95)
        ax.fill_between(forecast.index[-conf_int.shape[0]:], conf_int[:, 0], conf_int[:, 1],
                        color='gray', alpha=.5, label=fill_label)
        
    ax.legend(loc=2)
    ax.set_title(title)
    plt.show()
    
# invert box-cox transform
def boxcox_inverse(value, lam):
    if lam == 0:
        inv = exp(value)
    else:
        inv = exp(log(lam * value + 1) / lam)
    return(inv-delta_positive)

def predict_inverse(data_size,model_fit,lam,horizon,typ=None):
    if typ is not None:
        yhat = model_fit.predict(start=data_size,                                 end=data_size+horizon-1,
                                 typ='levels')
    else:
        yhat = model_fit.predict(start=data_size,                                end=data_size+horizon-1)
    return([boxcox_inverse(x,lam) for x in yhat])


# #### Run the ARIMA model on transformed data and plot on regular data

# In[ ]:


# Arima
dataset = boxcox_t
out = 'AIC: {0:0.3f}, BIC: {1:0.3f}'
# Walk-forward validation after training at least for 30 days
n_train = 30
n_records = len(dataset)
y_pred = []

for i in range(n_train, n_records):
    train = dataset[0:i]
    model = ARIMA(train, order=best_cfg)
    model_fit = model.fit(disp=False)
    #print(out.format(model_fit.aic,model_fit.bic))
    yhat = model_fit.predict(start=i,end=i,typ='levels')
    y_pred.extend(yhat)
    #print(i,dataset[i],yhat[0])
y_true = dataset[n_train:n_records]
metric = show_results(y_true,y_pred)


# In[ ]:


horizon=10
# Run the model in the entire transformed dataset using the fine-tuned parameters
dataset = boxcox_t
model = ARIMA(dataset, order=best_cfg)
model_fit = model.fit(disp=False)
y_pred = model_fit.predict(start=1,end=len(dataset)+horizon,typ='levels')

forecast, stderr, conf_int = model_fit.forecast(steps=horizon+1)
forecast=forecast[1:]
stderr=stderr[1:]
conf_int=conf_int[1:]

y_pred_inv = [boxcox_inverse(x,lam) for x in y_pred]
rmse = get_rmse(data, y_pred_inv[:len(data)])
title = type_case + " " + country + " RMSE = {:4.4f}".format(rmse)

conf_int_inv = np.array([[boxcox_inverse(x,lam),boxcox_inverse(x2,lam)]                 for (x,x2) in conf_int])

ts_forecast = pd.Series(y_pred_inv,                        index=pd.date_range(initial_date,                                                 periods=len(y_pred_inv)))

forecast_plot(ts_forecast,ts,title,conf_int_inv)
fig = plot_data(title,
          [ts,ts_forecast], 
          ["Actual", "Forecast"])


# #### rmse is in the 70's for one-day forecast using a fine-tuned ARIMA model with a box-cox transformed dataset. In previous exercises the mean rmse has been around 30's

# ### Holt-Winters Exponential Smoothing

# #### Grid search of configuration parameters

# In[ ]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

dataset = boxcox_t
out = 'AIC: {0:0.3f}, BIC: {1:0.3f}'
n_train = 30
n_records = len(dataset)

# Define parameters to grid search
l_trend=["add","mul",None]
l_damped=[True,False]
l_seasonal=["add","mul",None]
seasonal_periods=7
optimized=True
use_boxcox=False
l_remove_bias=[True,False]

best_metric = np.inf
best_trend, best_damped, best_remove_bias, best_seasonal = None,None,None,None

for trend in l_trend:
    for damped in l_damped:
        for remove_bias in l_remove_bias:
            for seasonal in l_seasonal:
                print("trend:",trend,", damped:",damped,                ", remove_bias:",remove_bias, ", seasonal:",seasonal)
                y_pred = []
                try:
                    for i in range(n_train, n_records):
                        train = dataset[0:i]
                        model = ExponentialSmoothing(train,trend=trend,damped=damped,                                                     seasonal=seasonal,                                                     seasonal_periods=seasonal_periods)
                        model_fit = model.fit(optimized=optimized,use_boxcox=use_boxcox,                                              remove_bias=remove_bias)
                        #print(out.format(model_fit.aic, model_fit.bic))
                        yhat = model_fit.predict(start=i,end=i)
                        y_pred.extend(yhat)
                    #print(i,dataset[i],yhat[0])
                    y_true = dataset[n_train:n_records]
                    metric = show_results(y_true,y_pred)
                    if metric < best_metric:
                        best_metric = metric
                        best_trend = trend
                        best_damped = damped
                        best_seasonal = seasonal
                        best_remove_bias = remove_bias
                except:
                    break
print("\nBest metric (rmse): " + "{0:4f}".format(best_metric))
print("trend:",best_trend,", damped:",best_damped,      ", remove_bias:",best_remove_bias,      ", seasonal:",seasonal)


# #### Run the Exponential Smoothing  model on transformed data and plot on regular data

# In[ ]:


horizon = 10
dataset = boxcox_t
trend=best_trend
damped=best_damped
seasonal=best_seasonal
remove_bias=best_remove_bias

# Run the model in the entire transformed dataset using the fine-tuned parameters

model = ExponentialSmoothing(dataset,trend=trend,damped=damped,                             seasonal=seasonal,                             seasonal_periods=seasonal_periods)
model_fit = model.fit(optimized=optimized,use_boxcox=use_boxcox,                         remove_bias=remove_bias)
y_pred = model_fit.predict(start=1,end=len(dataset)+horizon)

y_pred_inv = [boxcox_inverse(x,lam) for x in y_pred]
rmse = get_rmse(data, y_pred_inv[:len(data)])
title = type_case + " " + country + " RMSE = {:4.4f}".format(rmse)

ts_forecast = pd.Series(y_pred_inv,                        index=pd.date_range(initial_date,                                                 periods=len(y_pred_inv)))

forecast_plot(ts_forecast,ts,title)
fig = plot_data(title,
          [ts,ts_forecast], 
          ["Actual", "Forecast"])


# #### The rmse is in the 80's for one-day forecast using a fine-tuned Holt-Winters Exponential Smoothing model with a box-cox transformed dataset

# ### Long Short Term Memory (LSTRM) networks

# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import ConvLSTM2D


# #### Define helper functions to prepare data and different LSTM models

# In[ ]:


#https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return (np.array(X), np.array(y))

# Vainilla LSTM
def simple_lstm(num_nodes,n_steps,n_features):
    model = Sequential()
    model.add(LSTM(num_nodes, activation='relu',                    input_shape=(n_steps, n_features)))
    model.add(Dense(num_nodes))
    model.add(Dense(num_nodes*2))
    model.add(Dense(num_nodes))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return(model)

# Stacked LSTM
def stacked_lstm(num_nodes,n_steps,n_features):
    model = Sequential()
    model.add(LSTM(num_nodes, activation='relu',                    return_sequences=True,                    input_shape=(n_steps, n_features)))
    model.add(LSTM(num_nodes, activation='relu'))
    model.add(Dense(num_nodes))
    model.add(Dense(num_nodes*2))
    model.add(Dense(num_nodes))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return(model)

# Bidirectional LSTM
def biderectional_lstm(num_nodes,n_steps,n_features):
    model = Sequential()
    model.add(Bidirectional(LSTM(num_nodes, activation='relu'),                             input_shape=(n_steps, n_features)))
    model.add(Dense(num_nodes))
    model.add(Dense(num_nodes*2))
    model.add(Dense(num_nodes))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return(model)

def cnn_lstm(num_nodes,n_steps,n_features):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1,                                     activation='relu'),                              input_shape=(None, n_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(num_nodes, activation='relu'))
    model.add(Dense(num_nodes))
    model.add(Dense(num_nodes*2))
    model.add(Dense(num_nodes))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return(model)

def conv2d_lstm(num_nodes,n_steps,n_features,n_seq):
    model = Sequential()
    model.add(ConvLSTM2D(filters=64,                          kernel_size=(1,2), activation='relu',                          input_shape=(n_seq, 1, n_steps, n_features)))
    model.add(Flatten())
    model.add(Dense(num_nodes))
    model.add(Dense(num_nodes*2))
    model.add(Dense(num_nodes))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return(model)


# #### Identify the best type of LSTM network based on rmse

# In[ ]:


# define input layer sequence
raw_seq = data
n_features = 1


num_nodes=64
n_epochs=100

models = {"Simple LSTM":simple_lstm,          "Stacked LSTM":stacked_lstm,         "Bidirectional LSTM":biderectional_lstm,         "CNN LSTM":cnn_lstm,         "Conv2D LSTM":conv2d_lstm}

best_rmse = np.inf
best_lstm_model=None

for model_type in models.keys():
    if (model_type == "CNN LSTM"):
        n_steps = 4
        X, y = split_sequence(raw_seq, n_steps)
        # reshape from [samples, timesteps] into :
        #[samples, subsequences, timesteps, features]
        n_seq = 2
        n_steps = 2
        X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
        params = (num_nodes,n_steps,n_features)
    elif (model_type == "Conv2D LSTM"):
        n_steps = 4
        X, y = split_sequence(raw_seq, n_steps)
        # reshape from [samples, timesteps] into :
        #[samples, subsequences, timesteps, features]
        n_seq = 2
        n_steps = 2        
        X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
        params = (num_nodes,n_steps,n_features,n_seq)
    else:
        n_steps = 7
        X, y = split_sequence(raw_seq, n_steps)
        # reshape from [samples, timesteps] into :
        # [samples, timesteps, features]
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        params = (num_nodes,n_steps,n_features)

    model = models[model_type](*params)
    model.fit(X, y, epochs=n_epochs, verbose=0)
    
    y_pred=[]
    for ix in range(X.shape[0]):
        x_input = X[ix]
        if (model_type == "CNN LSTM"):            
            x_input = x_input.reshape((1, n_seq, n_steps, n_features))
        elif (model_type == "Conv2D LSTM"):            
            x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
        else:
            x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        y_pred.extend(yhat[0])
    rmse = get_rmse(y, y_pred)
    print(model_type + " -  RMSE = {:4.4f}".format(rmse))
    if rmse < best_rmse:
        best_rmse = rmse
        best_n_steps = n_steps
        best_lstm_model_name = model_type
        best_y_pred = y_pred


# #### Run the winning LSTM  model on regular data and plot it

# In[ ]:


ts_forecast = pd.Series(best_y_pred,                        index=pd.date_range(ts.index[best_n_steps],                                                 periods=len(best_y_pred)))

title = type_case + " " + country + " RMSE = {:4.4f}".format(best_rmse)
# Run the model in the entire dataset using the best LSTM network
forecast_plot(ts_forecast,ts,title)
fig = plot_data(title,
          [ts,ts_forecast], 
          ["Actual", "Forecast"])


# #### The rmse is in the 90's for one-day forecast using the best LSTM model  with the regular un-transformed dataset. Most of the runs I have tried rmse in above 130's

# ## Conclusions

# - FB Prophet provides a reasonable prediction accuracy without the need to adjust for data stationarity. It also offers some functions to plot and cross-validation testing.
# - Fine-tuned ARIMA model on transformed data provided the best rmse metric on one-day forecasts, but it produces a lot of errors on non-transformed data when it is not stationary.
# - Fine-tuned Holt-Winters Exponential Smoothing on transformed data provided a good rmse. Its performance was still very acceptable even with no fine-tunning configuration and no stationary data.
# - LSTM-based models do not require data to be stationary, and their training was really fast given the few number of records, but their rmse metric results were inconsistent (high variance) and turned-out to be the highest values.

# I would suggest to use Prophet for quick results, then Exponential Smoothing for better accuracy, and finally, ARIMA to maximize the accuracy at the cost of possible transformations and parameters fine-tuning.

# In[ ]:




