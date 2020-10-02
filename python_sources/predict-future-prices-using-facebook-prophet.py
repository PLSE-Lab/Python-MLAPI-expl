#!/usr/bin/env python
# coding: utf-8

# ## In this case study, we will predict prices of avocados using Facebook Prophet.
# ## Prophet is an open source tool used for time series forecasting.

# ### Import Libraries and Dataset

# In[ ]:


# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for  data visualisation
import random
import seaborn as sns
from fbprophet import Prophet


# In[ ]:


# dataframes creation for both training and testing datasets 
avocado_df = pd.read_csv('../input/avocado-prices/avocado.csv')


# About Data set-
# 
# - Date: The date of the observation
# - AveragePrice: the average price of a single avocado
# - type: conventional or organic
# - year: the year
# - Region: the city or region of the observation
# - Total Volume: Total number of avocados sold
# - 4046: Total number of avocados with PLU 4046 sold
# - 4225: Total number of avocados with PLU 4225 sold
# - 4770: Total number of avocados with PLU 4770 sold

# In[ ]:


# Let's view the head of the training dataset
avocado_df.head()


# In[ ]:


# Let's view the last elements in the training dataset
avocado_df.tail()


# In[ ]:


avocado_df.describe()


# In[ ]:


avocado_df.info()


# In[ ]:


avocado_df.isnull().sum()


# ### Exploratory Data Analysis

# In[ ]:


avocado_df = avocado_df.sort_values('Date')


# In[ ]:


# Plot date and average price
plt.figure(figsize=(10,10))
plt.plot(avocado_df['Date'],avocado_df['AveragePrice'])


# In[ ]:


# Plot distribution of the average price
plt.figure(figsize=(10,6))
sns.distplot(avocado_df['AveragePrice'],color='b')


# In[ ]:


# Plot a violin plot of the average price vs. avocado type
sns.violinplot(y='AveragePrice',x='type',data=avocado_df)


# In[ ]:


# Bar Chart to indicate the number of regions 

sns.set(font_scale=0.7) 
plt.figure(figsize=[20,8])
sns.countplot(x = 'region', data = avocado_df)
plt.xticks(rotation = 45)


# In[ ]:


# Bar Chart to indicate the count in every year
sns.set(font_scale=1.5) 
plt.figure(figsize=[16,8])
sns.countplot(x = 'year', data = avocado_df)
plt.xticks(rotation = 45)


# In[ ]:


# plot the avocado prices vs. regions for conventional avocados
conventional = sns.catplot('AveragePrice','region',data=avocado_df[avocado_df['type']=='conventional'],hue='year',
                          height=15)


# In[ ]:


# plot the avocado prices vs. regions for organic avocados
conventional = sns.catplot('AveragePrice','region',data=avocado_df[avocado_df['type']=='organic'],hue='year',
                         height=15)
  


# ## Data Preprocessing

# In[ ]:


avocado_df


# In[ ]:


avocado_prophet_df = avocado_df[['Date', 'AveragePrice']]


# In[ ]:


avocado_prophet_df


# In[ ]:


avocado_prophet_df.rename(columns={'Date':'ds','AveragePrice' : 'y'},inplace='true')


# In[ ]:


avocado_prophet_df


# ### Understanding the intuition Behind Facebook Prophet

# * Prophet is open source software released by Facebook's Core Data Science team.
# * Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
# * Prophet works best with time series that hace strong seasonal effects and several seasons of historical data.
# * For more information, please check this out:
# https://facebook.github.io/prophet/docs/quick_start.html

# ## Develop model and make Prediction

# In[ ]:


m = Prophet()
m.fit(avocado_prophet_df)


# In[ ]:


# Forcasting into the future
future = m.make_future_dataframe(periods = 365)
forecast = m.predict(future)


# In[ ]:


forecast


# In[ ]:


figure = m.plot(forecast, xlabel='Date', ylabel='Price')


# In[ ]:



figure2 = m.plot_components(forecast)


# ### Develop model and make region specific predictions

# In[ ]:


# dataframes creation for both training and testing datasets 
avocado_df = pd.read_csv('../input/avocado-prices/avocado.csv')


# In[ ]:


# Select specific region
avocado_df_sample = avocado_df[avocado_df['region']=='West']


# In[ ]:


avocado_df_sample = avocado_df_sample.sort_values('Date')


# In[ ]:


plt.plot(avocado_df_sample['Date'], avocado_df_sample['AveragePrice'])


# In[ ]:


avocado_df_sample = avocado_df_sample.rename(columns={'Date':'ds', 'AveragePrice':'y'})


# In[ ]:


m1 = Prophet()
m1.fit(avocado_df_sample)
# Forcasting into the future
future1 = m1.make_future_dataframe(periods=365)
forecast1 = m1.predict(future)


# In[ ]:


figure = m1.plot(forecast1, xlabel='Date', ylabel='Price')


# In[ ]:


figure3 = m.plot_components(forecast)


# ## Conclusion
# 
# ### Using Facebook Prophet model we succesfully predicted the prices of avocados for all regions for the period of next one year, we observered that prices will be in the range of 1 to 1.5
# ### Also for region "West" prediction we can see the prices will be in the range of 1.25 to 1.75.

# In[ ]:




