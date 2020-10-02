#!/usr/bin/env python
# coding: utf-8

# **This is a Time Analysis Model of Russian Airlines**
# This dataset contains Yearwise number of passenger,cargo & parcels traveled over Russian Airlines.
# 
# # Content:
# * Cargo and Parcel dataset Preprocessing
# * Exploratory Data Analysis
# * Visualizing the trend of Airlines over Russia
# * Rolling Test & visualization for checking stationarity of the data
# * Dicky Fuller Test
# * Shifting of Data to reduce Seasonality
# * Autocorrelation And Partial Autocorrelation 
# * Taking required data for proper implementation of ARIMA model
# * Implementing SEASONAL ARIMA MODEL
# * Testing our prediction with previous data
# * Making Future Prediction for 2021

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ***This is our dataset for RUSSIAN AIR SERVICE CARGO AND PARCELS***

# In[ ]:


df=pd.read_csv("/kaggle/input/russian-passenger-air-service-20072020/russian_air_service_CARGO_AND_PARCELS.csv")
df.head()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


print('Number of rows:', df.shape[0])
print('Number of Airports:', df['Airport name'].nunique())
print('First Year:', df['Year'].min())
print('Last Year:', df['Year'].max())


# # **DATA PREPROCESSING**

# In[ ]:


months = df.columns[~df.columns.isin([
    'Airport name',
    'Airport coordinates',
    'Whole year', 'Year'
])]
mapping= {v: k for k,v in enumerate(months, start=1)} 


# In[ ]:


time_series = df.melt(
    id_vars=['Airport name', 'Year'],
    value_vars=months,
    var_name='Month',
    value_name="cargo_&_parcel"
)
time_series.head()


# We will make a list of months & map those with corresponding Values

# In[ ]:


time_series['date'] = time_series.apply(lambda x: f"{x['Year']}-{mapping[x['Month']]:02d}", axis=1)

time_series['date'] = pd.to_datetime(time_series['date']) 


# In[ ]:



# Covert type
time_series = (
    time_series
    .rename(columns={'Airport name': 'airport', 'value': 'cargo_&_parcel'})
    .drop(columns=['Year', 'Month'])
)
time_series.head()


# # EDA

# In[ ]:


rpass=time_series.groupby(["date"])["cargo_&_parcel"].sum().loc[:'2020-01-01'] 
rpass.head()


# # USE THE SLIDER FOR DETAIL VIEW

# In[ ]:


import plotly.express as px
fig = px.line(rpass.reset_index(),x="date",y="cargo_&_parcel",title="TREND OF AIRLINES IN RUSSIA")
fig.update_xaxes(rangeslider_visible=True)
fig.show()


# ROLLING TEST
# 
# This test is to check stationarity of the Data
# 1. If Mean & Standard deviation both are straight then the data will be stationary
# 2. If any one of those are not straight then data the will be Not stationary
# 
# As per the visualization 
# * Mean is not straight
# * Std is straight
# So overall the data is Not stationary

# In[ ]:


plt.figure(figsize=(20,7))
rpassrolling=time_series.groupby(["date"])["cargo_&_parcel"].sum().loc[:'2019-12-01'] 
rolmean = rpassrolling.rolling(window=13).mean()
rolstd = rpassrolling.rolling(window=13).std()
original=plt.plot(rpassrolling,color="blue",label="Original")
mean=plt.plot(rolmean,color="red",label="Mean")
std=plt.plot(rolstd,color="black",label="Std")
plt.legend()
plt.title("ROLLING TEST")
plt.show()


# # Dicky Fuller Test
# 
# This is the similar test for Rolling test
# from this using P-value we can confirm wheather our data is stationary or not
# If p-value<0.05 data is stationary

# In[ ]:


from statsmodels.tsa.stattools import adfuller
def adfuller_test(passenger):
    res=adfuller(passenger)
    labels=["ADF TEST STATISTICS","P-VALUE","LAGS USED","NUMBER OF OBSERVATION USED"]
    for value,label in zip(res,labels):
        print(label+' : '+str(value))
    if(res[1]<=0.05):
        print("Stationary")
    else:
        print("Not Stationary")
adfuller_test(rpass.reset_index()["cargo_&_parcel"].dropna())

So our data is not stationary and now we have to make our data stationary
For that we will be using Differencing or Shifting Method over a prefferable Time window
# # **Differencing**

# In[ ]:


rpass=rpass.reset_index()
rpass["seasonal difference"]=rpass["cargo_&_parcel"]-rpass["cargo_&_parcel"].shift(13)
rpass.tail()


# In[ ]:


plt.figure(figsize=(20,5))
adfuller_test(rpass.reset_index()["seasonal difference"].dropna())
px.line(rpass,y="seasonal difference",x="date",title="TREND AFTER SHIFTING")


# In[ ]:


plt.figure(figsize=(10,5))
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(rpass["cargo_&_parcel"])
plt.show()


# # Autocorrelation And Partial Autocorrelation

# In[ ]:


import statsmodels as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# FROM AUTOCORRELATION we will find upto which lag exponential decrease is there
# for this case it is 10
# So P=10
# FROM PARTIAL AUTOCORRELATION we find shutdown point
# here q=0

# In[ ]:


fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig=plot_acf(rpass["seasonal difference"].dropna(),lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=plot_pacf(rpass["seasonal difference"].dropna(),lags=40,ax=ax2)


# In[ ]:


#p=0,q=10,d=1
from statsmodels.tsa.arima_model import ARIMA


# In[ ]:


#rpass.set_index("date",inplace=True)
import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(rpass["cargo_&_parcel"],order=(0,1,10),seasonal_order=(0,1,10,12))
results=model.fit()


# In[ ]:


rpass["Future Prediction"]=results.predict(start=130,end=154,dynamic=True)
rpass[["cargo_&_parcel","Future Prediction"]].plot(figsize=(20,8))
plt.title("TESTING PREDICTED VALUE WITH DATASET")


# Now It is time for making prediction

# In[ ]:


from pandas.tseries.offsets import DateOffset
future_dates=[rpass.index[-1]+ DateOffset(months=x) for x in range(0,24)]


# In[ ]:


future_datest_df=pd.DataFrame(index=future_dates[1:],columns=rpass.columns)
future_datest_df.tail()


# In[ ]:


future_df=pd.concat([rpass,future_datest_df])
future_df['Future Prediction']=results.predict(start=156,end=200,dynamic=True)
future_df[["cargo_&_parcel","Future Prediction"]].plot(figsize=(20,8))


# In[ ]:




