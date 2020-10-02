#!/usr/bin/env python
# coding: utf-8

# ## IMPORT THE NECESSARY LIBRARIES

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from itertools import combinations

from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA as ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
pd.options.display.float_format = '{:.2f}'.format


# ## IMPORT THE DATASET

# In[ ]:


data = pd.read_csv('../input/avocado-prices/avocado.csv')
data = data.drop('Unnamed: 0',axis = 1)
data.head()


# ## VISUALIZE THE DATASET

# ### AVOCADO TYPES

# In[ ]:


sns.countplot('type',data = data)


# ### PERCENTAGE OF EACH TYPE OF AVOCADO

# In[ ]:


df = data.groupby(data['type']).sum()
plt.pie(df['Total Volume'],data = df,labels = ['CONVENTIONAL','ORGANIC'])
plt.show()


# ### TOTAL VOLUME vs YEAR

# In[ ]:


sns.barplot(x = 'year',y = 'Total Volume',data = data,palette= 'Blues')


# ## HEATMAP FOR FINDING CORRELATIONS BETWEEN AVERAGEPRICE & OTHER FEATURES

# In[ ]:


df = data.drop(columns = ['year','region'])
sns.heatmap(df.corr())


# ### AVERAGEPRICE ACCORDING TO CITIES ALONG THE YEARS

# In[ ]:


sns.factorplot('AveragePrice','region',data=data,hue='year',size=18,aspect=0.7,palette='Blues',join=False)


# ### AVERAGEPRICE VARIATION ALONG THE YEARS 

# In[ ]:


sns.boxplot('year','AveragePrice',data = data)


# ### DISTRIBUTION OF AVERAGEPRICE

# In[ ]:


sns.distplot(data['AveragePrice'])


# ### PLOTS WITH RESPECT TO TIME

# In[ ]:


sns.lineplot('Date','AveragePrice',hue = 'year',data = data,)


# In[ ]:


sns.lineplot('Date','Total Volume',hue = 'year',data = data)


# ### SOME MORE VISUALIZATIONS

# In[ ]:


sns.swarmplot('Date','AveragePrice',data = data,hue = 'type')


# In[ ]:


sns.catplot('year','Total Volume',data = data)


# In[ ]:


sns.catplot('year','Total Bags',data = data)


# # TIMESERIES ANALYSIS

# ### FUNCTIONS FOR ANALYSIS

# In[ ]:


def test_stationarity(timeseries):
    #Determing rolling statistics
    MA = timeseries.rolling(window=12).mean()
    MSTD = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(15,5))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(MA, color='red', label='Rolling Mean')
    std = plt.plot(MSTD, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# In[ ]:


def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


# ## PREPROCESSING

# - DROPPING THE UNNECESSARY COLUMNS
# - CONVERTING DATES TO DATETIME AND SETTING IT AS INDEX
# - RESAMPLE [GROUPBY] THE DATA ACCORDING TO YOUR DATASET

# In[ ]:


data = data.drop(columns = ['Total Volume','Total Bags','year','4046','4225','4770','Small Bags','Large Bags','XLarge Bags'])
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data.resample('W').sum()
data.head()


# ## TIMESERIES ANALYSIS

# In[ ]:


test_stationarity(data['AveragePrice'])


# In[ ]:


dec = sm.tsa.seasonal_decompose(data['AveragePrice'],period = 52).plot()
plt.show()


# - DATA HAS TREND & SEASONALITY 
# - DATA HAS AN INCREASING TREND
# - DATA HAS A YEARLY SEASONAL PATTERN 
# - DATA IS NOT STATIONARY 

# # DIFFERENCING

# - 1st DIFFERENCING TO ELIMINATE THE TREND 

# In[ ]:


data_diff = data['AveragePrice'].diff()
data_diff = data_diff.dropna()
dec = sm.tsa.seasonal_decompose(data_diff,period = 52).plot()
plt.show()


# In[ ]:


test_stationarity(data_diff)


# - TREND HAS NOW BEEN ELIMINATED BUT SEASONALITY IS PRESENT BUT NOT CLEARLY VISIBLE
# - DICKEY-FULLER TEST TELLS US THAT OUR DATA IS STATIONARY
# - DETERMINATION IF DATA  IS STATIONARY:
# 
# - 1] P-VALUE < 0.05
# - 2] TEST STATISTIC < CRITICAL VALUE
# - 3] THE MOVING AVERAGE OF THE DATA IS ALSO NEARLY 0 AND ROTATES AROUND 0

# # ARIMA 

# -  THE PREDICTIONS OF ARIMA MODEL ARE NOT GOING TO BE GOOD BECAUSE OUR DATA IS SEASONAL
# -  WE HAVE NOT CONSIDERED THE SEASONAL PATTERN IN THIS PATTERN
# -  p = PARTIAL AUTOCORRELATION = 0 -> PLOT SHOWS NO INSIGNIFICANT LAGS
# -  d = DIFFERENCING = 0 -> DIFFERENCING DONE 1 TIME
# -  q = AUTOCORRELATION = 0 -> PLOT SHOWS NO INSIGNIFICANT LAGS

# In[ ]:


tsplot(data_diff)


# In[ ]:


model = ARIMA(data['AveragePrice'],order = (0,1,0))
model_fit = model.fit()
print(model_fit.summary())


# In[ ]:


data['FORECAST'] = model_fit.predict(start = 130,end = 170,dynamic = True)
data[['AveragePrice','FORECAST']].plot(figsize = (10,6))


# In[ ]:


exp = [data.iloc[i,0] for i in range(130,len(data))]
pred = [data.iloc[i,1] for i in range(130,len(data))]
data = data.drop(columns = 'FORECAST')
error = mean_absolute_error(exp,pred)
error


# - THE MODEL OUPUTS POOR PREDICTIONS
# - MEAN ABSOLUTE ERROR VALUE ALSO HAS A HIGH VALUE
# - THEREFORE,WE REJECT THIS MODEL AND MOVE TOWARDS SEASONAL ARIMA [SARIMA]

# # SARIMAX

# - HERE WE ARE WORKING ON WEEKLY DATA AND OUR SEASONAL PERIOD IS OF 1 YEAR 
# - FOR OUR DATA, S = 52 WEEKS , 52 ENTRIES OF OUR DATAFRAME
# - HENCE WE DIFFERENCE BY 52

# In[ ]:


data_diff_seas = data_diff.diff(52)
data_diff_seas = data_diff_seas.dropna()
dec = sm.tsa.seasonal_decompose(data_diff_seas,period = 52).plot()
plt.show()


# In[ ]:


tsplot(data_diff_seas)


# ## SEASONAL ORDER

# ## SIMILAR TO ARIMA ORDER
# -  P ->(0-2) 
# -  D ->1
# -  Q ->(0-2)
# - SELECT ORDER HAVING THE LEAST AIC MODEL VALUE

# In[ ]:


model = sm.tsa.statespace.SARIMAX(data['AveragePrice'],order = (0,1,0),seasonal_order = (1,1,0,52))
results = model.fit()
print(results.summary())


# In[ ]:


data['Forecast'] = results.predict(start = 130,end = 169,dynamic = True)
data[['AveragePrice','Forecast']].plot(figsize = (12,8))


# In[ ]:


exp = [data.iloc[i,0] for i in range(130,len(data))]
pred = [data.iloc[i,1] for i in range(130,len(data))]

error = mean_absolute_error(exp,pred)
error


# - MEAN ABSOLUTE ERROR OF OUR MODEL IS 11.64
# - IT IS BETTER THAN ARIMA MODEL AND CAN BE IMPROVED BY DIFFERENING OR USING THE SAME TECHNIQUES 
# - WITH DIFFERENT TRANSFORMATIONS FOR FORECASTING PURPOSE

# # FORECASTING

# In[ ]:


from pandas.tseries.offsets import DateOffset
future_dates = [data.index[-1] + DateOffset(weeks = x)for x in range(0,52)]


# - ADDING DATES FOR FORECASTING

# In[ ]:


df = pd.DataFrame(index = future_dates[1:],columns = data.columns)


# In[ ]:


forecast = pd.concat([data,df])
forecast['Forecast'] = results.predict(start = 170,end = 222,dynamic = True)
forecast[['AveragePrice','Forecast']].plot(figsize = (12,8))


# # THANK YOU
