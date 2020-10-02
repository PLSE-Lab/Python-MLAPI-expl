#!/usr/bin/env python
# coding: utf-8

# **Time series:** 
#    
#    Let consider the few persons had taken the loan from the company. So the people who taken the loan depends on the family size and income level won't depends on past value. 
#     
#    Consider the dataset of percentage of nitrogen level which are recorded every day. So Time series is defined as the data which is recorded consistently with time i.e hourly,daily,yearly,weekly etc. 
#     
#    For forecasting the time series data we require the past data. We can't apply regression model because regression model won't consider the relationship between the past values.
#     
# **  Models can be applied on Time series Data: **
# 
# **1. Naive Approch:**
# 
# The future data is predicted by the previous value. But the forecasting will be flat line.
# 
# **2. Simpe Average Approch:**
# 
# The future data is predicted by the considering all the average of previous value.So by this forecastiong won't be flat line. But every time we won't require the all previous values for forecasting.
# 
# **3.Moving Average Approch:**
# 
# The future value is predicted by considering N previous values. 
# 
# **4. Weighted Moving Average Approch:**
# 
# The future value is predicted by considering the different weights for previous N values. 
# 
# **5. Simple Exponentail Moving:**
# 
# The funture value is predicted by considering the more weights for most recent previous values and than the distant observations.
# 
# **6.Holt's Linear Trend model:**
# 
# In the above approches the trend is not considered.  consider the booking of hotel are increasing every year, which means increasing trend.
# 
# **7.Holt winter Method:**
# 
# Similar to holt's linear trend model also it consider seasonality. Not only booking of hotel are increasing every year but weekdays the bookings are decreasing and increasing at the weekend. 
# 
# **8.ARIMA:**
# 
# ARIMA is the most widely used  method for time series. Auto Regression integrated moving average.ARIMA provides the correlation between the datapoints. So ARIMA considers the data as the stationay. 
# 
# ** *Stationay Data- Mean and Covarience must be constant according time.* **
# Stationary can be made by using log transformation. Also data provided input must be as the univarient,since arima uses the past values for prediction of fututre of values. ARIMA has 3 components
# AR-Auto regression. I- differencing term, MA- moving average.
# 
# **  *AR- term uses the past value for forecasting the next value. So it is defined by 'p' and determined by PACF plot.* **  
# 
# **   *MA- uses the past forecast errors used to predict the fututre values. So it is defined as 'q' and determined by ACF plot.* **  
# 
# **  *d- differencing order specifies the number of times differencing operation is performed on series to make it stationary.* **  
# 
# **9. AUTO ARIMA:**
# 
# It is same to ARIMA model but there is no need to find the values of 'p' and 'q'. These values are calucluated automatically by the auto arima model.    
# 
# **We will use ARIMA and build an AUTO ARIMA function here**
#    

# In[ ]:



get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.core import datetools


# In[ ]:



df=pd.read_csv('../input/international-airline-passengers.csv')


# In[ ]:


df.head()


# In[ ]:


start = datetime.datetime.strptime("1949-01-01", "%Y-%m-%d")
type(start)


# In[ ]:


print(start)


# In[ ]:


len(df)


# In[ ]:


date_list = [start + relativedelta(months=x) for x in range(0,df.shape[0])]


# In[ ]:


print(date_list[0:4])


# In[ ]:


for x in range(0,144):
    print(x)


# In[ ]:





# In[ ]:


df['index'] =date_list
df.set_index(['index'], inplace=True)
df.index.name=None


# In[ ]:


df.columns


# In[ ]:


df.columns=('Month','AirPassengers')


# In[ ]:


#del df['Unnamed: 0']
del df['Month']
df.head()


# In[ ]:


df.info()


# In[ ]:


np.count_nonzero(df)


# In[ ]:


df['AirPassengers'] = df.AirPassengers*1000


# In[ ]:


#plotting the data
import matplotlib.pyplot as plt

df.AirPassengers.plot(figsize=(12,8), title= 'Monthly Passengers', fontsize=14)
plt.savefig('month_ridership.png', bbox_inches='tight')


# In[ ]:


#building the model by first decomposing time series in trend and seasonality and irregularity
decomposition = seasonal_decompose(df.AirPassengers, freq=12)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)


# In[ ]:


# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)
print(p)


# In[ ]:


import itertools
import warnings


# In[ ]:


# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
print(pdq)


# In[ ]:


# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[ ]:


y=df


# Finding the best model with lowest AIC and parsing its results into a model

# In[ ]:


warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
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


warnings.filterwarnings("ignore") # specify to ignore warning messages
c3=[]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            c3.append( results.aic)
        except:
            continue


# In[ ]:


c3


# In[ ]:


import numpy as np
index_min = np.argmin(c3)


# In[ ]:


#minimum AIC for the models
np.min(c3)


# In[ ]:


index_min


# In[ ]:


c3[index_min]


# In[ ]:


warnings.filterwarnings("ignore") # specify to ignore warning messages
c4=[]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            c4.append('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[ ]:


c4[index_min]


# In[ ]:


order1=c4[index_min][6:13]
order1


# In[ ]:


order1=[int(s) for s in order1.split(',')]
order1


# In[ ]:


seasonal_order1=c4[index_min][16:27]
seasonal_order1


# In[ ]:


seasonal_order1=[int(s) for s in seasonal_order1.split(',')]
seasonal_order1


# In[ ]:


from statsmodels.tsa.x13 import x13_arima_select_order


# In[ ]:


mod = sm.tsa.statespace.SARIMAX(df.AirPassengers, trend='n', order=order1, seasonal_order=seasonal_order1)


# In[ ]:


results = mod.fit()
print (results.summary())


# In[ ]:


df[121:144]


# In[ ]:


results.predict(start=120,end=144)


# In[ ]:


type(df)


# In[ ]:


type(results.predict(start=120,end=144))


# In[ ]:


y.tail()


# In[ ]:


df2=pd.DataFrame(results.predict(start=120,end=144))


# In[ ]:


y.head()


# In[ ]:


df2.head()


# In[ ]:


ax = df.plot(figsize=(12,8), title= 'Monthly Passengers', fontsize=14)
df2.plot(ax=ax)


# **PLEASE VOTE FOR ENCOURAGEMENT**
