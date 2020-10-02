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


# Load specific forecasting tools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose      # for ETS Plots
# from pmdarima import auto_arima                              # for determining ARIMA orders
from pyramid.arima import auto_arima
print("hello")
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.stattools import adfuller
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/Metro_Interstate_Traffic_Volume.csv')
df_comp=df.copy()


# In[ ]:


df_comp.head()


# In[ ]:


plt.figure(figsize = (8,6))
sns.countplot(y='weather_description', data = df_comp)
plt.show()


# In[ ]:


plt.hist(df_comp.rain_1h.loc[df_comp.rain_1h<1])
plt.show()


# In[ ]:


df_comp["year"]=df['date_time'].str.slice(0,4).astype(int)
df_comp["month"]=df['date_time'].str.slice(5,7).astype(int)
df_comp["day"]=df['date_time'].str.slice(8,10).astype(int)
df_comp["year_month_day"]=df_comp["date_time"].str.slice(0,10)


# In[ ]:


df_comp.date_time = pd.to_datetime(df_comp.date_time, dayfirst = True)
df_comp.drop_duplicates(["date_time"],inplace=True)
df_comp.set_index("date_time", inplace=True)


# In[ ]:


df_comp=df_comp.loc[:,["holiday","temp","rain_1h","snow_1h","traffic_volume","year","month","day","year_month_day"]]
df_comp=df_comp.loc[(df_comp["year"]==2018) & df_comp["month"].isin([3,4,5,6,7])]


# In[ ]:


df_comp.loc[:,'holiday']=df_comp['holiday'].str.replace("None","1")
df_comp.loc[df_comp["holiday"]!='1','holiday']='0'
df_comp.loc[:,"holiday"]=df_comp['holiday'].astype(int)


# In[ ]:


df_comp.info()


# In[ ]:


pg=df_comp[((df_comp["holiday"]==0))]
pg=pg.loc[:,["month","day"]]
for i,j in pg.iterrows():
    #print(str(i))
    for m,n in df_comp.iterrows():
        sj=str(i)
        sj1=sj[:10]
        if(sj1==str(n['year_month_day'])):
            df_comp.loc[m,'holiday']=0
            


# In[ ]:


def getday(x):
    return int(pd.to_datetime(x).weekday())
df_comp["week_day"]=df_comp["year_month_day"].map(lambda x: getday(x))


# In[ ]:


df_comp[df_comp["year_month_day"]=="2018-07-04"]


# In[ ]:


df_comp.info()


# In[ ]:


# for i,j in df_comp.iterrows():
#     if(df_comp.loc[i,"week_day"]>4 or df_comp.loc[i,"holiday"]==0):
#         df_comp.at[i,"weekday_encoded"]=int(0)
#     else:
#         df_comp.at[i,"weekday_encoded"]=int(1) 
for i,j in df_comp.iterrows():
    if(df_comp.loc[i,"week_day"]>4):
        df_comp.at[i,"weekday_encoded"]=0
    else:
        df_comp.at[i,"weekday_encoded"]=1


# In[ ]:


df_comp.loc[:,"weekday_encoded"]


# In[ ]:


df_comp=df_comp.asfreq('h')
# print(df_comp.isna().sum())
df_comp=df_comp.fillna(method='ffill')
# print(df_comp.isna().sum())
# df_comp.index.freq = 'h'
print(df_comp.index.freq)


# In[ ]:


print(df_comp.isna().sum())


# In[ ]:


df_comp=df_comp.loc[:,["traffic_volume","holiday","weekday_encoded","month","year"]]


# In[ ]:


df_comp


# In[ ]:


df_comp_train=df_comp[df_comp["month"].isin([3,4,5,6])]
df_comp_test=df_comp[df_comp["month"].isin([7])]


# In[ ]:


df_comp_train.iloc[1056:1104,0]=df_comp_train.iloc[888:936,0].values


# In[ ]:


df_comp_train["traffic_volume"].plot(figsize=(20,5))


# In[ ]:



from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("AAAAAAAAAAAAAAAa",result[1])
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        
        
adf_test(df_comp_train["traffic_volume"])


# In[ ]:


result = seasonal_decompose(df_comp_train['traffic_volume'],model="additive")


# In[ ]:


result.trend.plot();


# In[ ]:


result.seasonal.plot(figsize=(20,5))


# In[ ]:


plot_pacf(df_comp_train["traffic_volume"],lags=40);


# In[ ]:


plot_acf(df_comp_train["traffic_volume"],lags=170);


# In[ ]:


# auto_arima(df_comp_train["traffic_volume"],seasonal=True,start_p=0, start_q=0,max_p=24, max_q=24, m=24,d=1,trace=True,error_action='ignore',stepwise=True).summary()  
#            # we don't want to know if an order does not worksuppress_warnings=True,  # we don't want convergence warnings
                          


# In[ ]:


model_sarima_no_exog=SARIMAX(df_comp_train["traffic_volume"],order=(3,1,7),seasonal_order=(1,1,1,24),enforce_invertibility=False)
result_sarima_no_exog=model_sarima_no_exog.fit()
result_sarima_no_exog.summary()


# In[ ]:


result_sarima_no_exog.resid.plot(kind='kde')


# In[ ]:


start=len(df_comp_train)
end=len(df_comp_train)+len(df_comp_test)-1
predictions_sarima_no_exog=result_sarima_no_exog.predict(start=start,end=end,dynamic=False)


# In[ ]:


df_comp_test["traffic_volume"].plot(figsize=(20,5))
predictions_sarima_no_exog.plot()


# In[ ]:


from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

error_mse_noex = mean_squared_error(df_comp_test['traffic_volume'], predictions_sarima_no_exog)
error_rmse_noex = rmse(df_comp_test['traffic_volume'], predictions_sarima_no_exog)

print("MSE",error_mse_noex)
print("RMSE",error_rmse_noex)


# In[ ]:


model_sarima_exog_holiday = SARIMAX(df_comp_train['traffic_volume'],order=(3,0,1),seasonal_order=(2,0,1,24),enforce_invertibility=False,exog=df_comp_train[["holiday"]])
results_sarima_exog_holiday = model_sarima_exog_holiday.fit()
results_sarima_exog_holiday.summary()


# In[ ]:


start=len(df_comp_train)
end=len(df_comp_train)+len(df_comp_test)-1
predictions_sarima_exog_holiday = results_sarima_exog_holiday.predict(start=start, end=end, dynamic=False,exog=df_comp_test[["holiday"]]).rename('SARIMA(1,0,1)(1,0,1,24) Predictions')


# In[ ]:


df_comp_test["traffic_volume"].plot(figsize=(20,5))
predictions_sarima_exog_holiday.plot()


# In[ ]:


results_sarima_exog_holiday.resid.plot()


# In[ ]:


results_sarima_exog_holiday.resid.plot(kind='kde')


# In[ ]:


model_sarima_exog_weekend = SARIMAX(df_comp_train['traffic_volume'],order=(2,1,1),seasonal_order=(1,0,1,24),enforce_invertibility=False,exog=df_comp_train[["weekday_encoded"]])
results_sarima_exog_weekend = model_sarima_exog_weekend.fit()
results_sarima_exog_weekend.resid


# In[ ]:


results_sarima_exog_weekend.resid.plot(kind='kde')


# In[ ]:


import scipy.stats
import pylab


# In[ ]:


scipy.stats.probplot(results_sarima_exog_weekend.resid,plot=pylab)
plt.show()


# In[ ]:


start=len(df_comp_train)
end=len(df_comp_train)+len(df_comp_test)-1
predictions_sarima_exog_weekend = results_sarima_exog_weekend.predict(start=start, end=end, dynamic=False,exog=df_comp_test[["weekday_encoded"]]).rename('SARIMA(1,0,1)(1,0,1,24) Predictions')


# In[ ]:


df_comp_test["traffic_volume"].plot(figsize=(20,5))
predictions_sarima_exog_weekend.plot()


# In[ ]:


error_mse_ex_weekend = mean_squared_error(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend)
error_rmse_ex_weekend= rmse(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend)

print("MSE",error_mse_ex_weekend)
print("RMSE",error_rmse_ex_weekend)


# In[ ]:


model_sarima_exog_holiday_weekend = SARIMAX(df_comp_train['traffic_volume'],order=(1,0,1),seasonal_order=(1,0,1,24),enforce_invertibility=False,exog=df_comp_train[["weekday_encoded","holiday"]])
results_sarima_exog_holiday_weekend = model_sarima_exog_holiday_weekend.fit()
results_sarima_exog_holiday_weekend.summary()


# In[ ]:


start=len(df_comp_train)
end=len(df_comp_train)+len(df_comp_test)-1
predictions_sarima_exog_holiday_weekend = results_sarima_exog_holiday_weekend.predict(start=start, end=end, dynamic=False,exog=df_comp_test[["weekday_encoded","holiday"]]).rename('SARIMA(1,0,1)(1,0,1,24) Predictions')


# In[ ]:


df_comp_test["traffic_volume"].plot(figsize=(20,5))
predictions_sarima_exog_holiday_weekend.plot()


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mean_absolute_percentage_error(df_comp_test["traffic_volume"],predictions_sarima_exog_holiday_weekend)


# In[ ]:


error_mse_ex_holiday_weekend = mean_squared_error(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend)
error_rmse_ex_holiday_weekend= rmse(df_comp_test['traffic_volume'], predictions_sarima_exog_weekend)

print("MSE",error_mse_ex_holiday_weekend)
print("RMSE",error_rmse_ex_holiday_weekend)


# In[ ]:


predictions_sarima_exog_weekend.mean()


# In[ ]:


df_comp_test.traffic_volume.mean()


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2_score(df_comp_test.traffic_volume, predictions_sarima_exog_weekend)


# In[ ]:




