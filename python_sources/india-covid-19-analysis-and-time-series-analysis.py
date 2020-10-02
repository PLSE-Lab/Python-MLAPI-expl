#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[ ]:


# import os
# print(os.listdir("../input/time-series/case.xlsx"))


# In[ ]:


dataset = pd.read_csv('../input/datacovid19/case_time_series.csv')


# In[ ]:


dataset.info()


# In[ ]:


dataset['date'] = pd.to_datetime(dataset['Date'])


# In[ ]:


dataset.info()


# In[ ]:


import seaborn as sns
plt.figure(figsize=(20,10))
g = sns.jointplot(dataset['date'],dataset['Total Confirmed'], kind="kde", height=7, space=0)
plt.gcf().autofmt_xdate()
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
sns.set(style="darkgrid")
sns.lineplot(x='date', y='Total Confirmed',
             data=dataset,  linewidth=3)
plt.gcf().autofmt_xdate()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime as dt
plt.figure(figsize=(15,10))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.plot(dataset['date'],dataset['Total Confirmed'], linewidth=3)
plt.gcf().autofmt_xdate()
plt.show()


# In[ ]:


df = dataset.groupby(['date'])[['Daily Confirmed','Daily Recovered','Daily Deceased']].sum()

plt.figure(figsize=(15,10))
plt.title('DAILY CASES OF COVID-19 CASES IN INDIA', fontsize=30)
plt.xlabel('date')
plt.ylabel('NO OF CASES')
plt.plot(df.index,df['Daily Confirmed'], label='daily_confirmed', linewidth=3)
plt.plot(df.index,df['Daily Recovered'], label='daily_recovered', linewidth=3, color='green')
plt.plot(df.index,df['Daily Deceased'], label='daily_deceased', linewidth=3, color='red')
plt.bar(df.index,df['Daily Confirmed'], alpha=0.2, color='c')
plt.style.use('ggplot')
plt.legend()


# In[ ]:


# total cases
df_total = dataset.groupby(['date'])[['Total Confirmed','Total Recovered','Total Deceased']].sum()
plt.figure(figsize=(15,10))
plt.title('TOTAL CASES OF COVID-19 CASES IN INDIA', fontsize=30)
plt.xlabel('date', fontsize=20)
plt.ylabel('Total Number of Cases', fontsize=20)
plt.plot(df_total.index,df_total['Total Confirmed'], label='Total Confirmed',linewidth=3)
plt.plot(df_total.index,df_total['Total Recovered'], label='Total Recovered', linewidth=3)
plt.plot(df_total.index,df_total['Total Deceased'], label='Total Deceased', linewidth=3)
plt.bar(df_total.index,df_total['Total Confirmed'], label='Total Confirmed', alpha=0.2, color='c')
plt.style.use('ggplot')
plt.legend(loc='best')


# In[ ]:


#df
labels = 'Total Recovered','Total Deceased'
recovered = dataset['Total Recovered']
deceased = dataset['Total Deceased']
sizes = [recovered.sum() , deceased.sum()]
explode = [0,0.1]
colors = ['yellowgreen','lightcoral']
plt.figure(figsize=(10,20))

plt.title('Dsitribution of confirmed cases Till 4th MAY', fontsize=20)
plt.pie(sizes, autopct='%1.1f%%', labels=labels,explode=explode,colors=colors, shadow=True)
plt.legend(labels, loc='best')
plt.show()


# ***TIME SEREIS ANALYIS FOR UPCOMING NEXT 10 DAYS******

# In[ ]:


df = pd.read_excel('../input/time-series/case.xlsx')


#  **WE WILL TAKE ONLY TWO COLUMNS FOR ANALYSIS - 'date' and 'Total Confirmed'**

# In[ ]:


df.head()


# **WE WILL CHANGE INDEX TO DATE INDEX**

# In[ ]:


df = df.set_index(['date'])


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(10,8))
plt.plot(df)


# In[ ]:


df.isnull().sum()


# In[ ]:


# rolling mean

rolmean = df.rolling(window=6).mean()
rolstd = df.rolling(window=6).std()


# In[ ]:


print(rolmean,rolstd)


# In[ ]:


plt.figure(figsize=(10,8))
plt.plot(df, color='blue', label='original cases')
plt.plot(rolmean, color='red', label='rolling mean')
plt.plot(rolstd, color='black', label='rolling standard deviation')
plt.legend(loc='best')
plt.show()


# **WE WILL DO DICKEY-FULLER TEST TO CHECK STATIONARITY OF DATA
# AND TO REJECT NULL HYPOTHESIS**

# In[ ]:


from statsmodels.tsa.stattools import adfuller
def test(data):
    rolmean = data.rolling(window=4).mean()
    rolstd = data.rolling(window=4).std()
    plt.figure(figsize=(10,8))
    plt.plot(data, color='blue', label='original cases')
    plt.plot(rolmean, color='red', label='rolling mean')
    plt.plot(rolstd, color='black', label='rolling standard deviation')
    plt.legend(loc='best')
    plt.show()
    
    dftest = adfuller(data['Total Confirmed'], autolag = 't-stat')
    dfoutput = pd.Series(dftest[0:4], index=['test statitics','p_value','lags used','number of observations'])
    for key,value in dftest[4].items():
        dfoutput['critcal value (%s)'%key] = value
        
    print(dfoutput)


# In[ ]:


test(df)


# **HERE P_VALUE IS LARGER THAN 0.5 SO WE CANNOT REJECT NULLL HYPOTHESIS THAT DATA IS NOT STATIONARY**

# **# TREND ANALYSIS USING LOG TRANSFORMATION**

# In[ ]:


df_log = np.log(df)
plt.figure(figsize=(10,8))
plt.plot(df_log)
plt.gcf().autofmt_xdate()
plt.show()
df_log.tail()


# In[ ]:


test(df_log)


# In[ ]:


movingaverage = df_log.rolling(window=4).mean()
rolstd = df_log.rolling(window=4).std()
plt.figure(figsize=(10,8))
plt.plot(df_log, color='blue', label='original cases')
plt.plot(movingaverage, color='red', label='rolling mean')
plt.plot(rolstd, color='black', label='rolling standard deviation')
plt.legend(loc='best')
plt.show()


# In[ ]:


df_log_minus = df_log - movingaverage
df_log_minus.dropna(inplace=True)
df_log_minus.tail(12)


# In[ ]:


test(df_log_minus)


# **HERE P_VALUE IS FAR LESS THAN CRITICAL VALUE SO WE CAN REJECT NULL HYPOTHESIS AND WE CAN SAY OUR DATA IS STATIONARY**

# In[ ]:


data_shift = df_log_minus - df_log_minus.shift()
plt.figure(figsize=(10,8))
plt.plot(data_shift)


# In[ ]:


data_shift.dropna(inplace=True)
test(data_shift)


# **WE WILL USE ARIMA MODEL **

# In[ ]:


# TO CALCULATE P AND Q VALUE FOR ARIMA MODEL
# TO CALCULATE ACF AND PACF

from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(data_shift, nlags=20)
lag_pacf = pacf(data_shift, nlags=20, method='ols')

plt.figure(figsize=(10,8))
#plot acf
plt.subplot(211)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_shift)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_shift)), linestyle='--', color='gray')
plt.title('ACF')
plt.legend(loc='best')

#plot pacf
plt.subplot(212)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_shift)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_shift)), linestyle='--', color='gray')
plt.title('PACF')
plt.legend(loc='best')


# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings('ignore')
model = ARIMA(df_log, order=(4,1,4))
result = model.fit(disp=-1)
plt.figure(figsize=(10,8))
plt.plot(data_shift)
plt.plot(result.fittedvalues, color='blue')
plt.title('RSS %-4F'% sum((result.fittedvalues- data_shift['Total Confirmed'])**2))


# In[ ]:


pred_arima_diff = pd.Series(result.fittedvalues, copy=True)
pred_arima_diff


# In[ ]:


# we will take cumsum
pred_arima_diff_cumsum = pred_arima_diff.cumsum()
pred_arima_diff_cumsum.tail()


# In[ ]:


prediction = pd.Series(df_log['Total Confirmed'].iloc[0], index=df_log.index)
prediction = prediction.add(pred_arima_diff_cumsum, fill_value=0)
prediction.tail()


# In[ ]:


pred = np.exp(prediction)
plt.figure(figsize=(10,8))
plt.plot(df)
plt.plot(pred, color='green')


# In[ ]:


pred.tail()


# In[ ]:


df


# **WE WILL DO 10 DAYS PREDICTION 5 MAY TO 15 MAY , 2020**

# In[ ]:



result.plot_predict(1,106)
plt.figure(figsize=(10,8))


# In[ ]:


x = result.forecast(steps=10)


# In[ ]:


x = np.exp(x[0])


# In[ ]:


for i in x:
    print(i)


# **5 may actual case = 49,391**
# **Predicted one = 50,246**
#    
