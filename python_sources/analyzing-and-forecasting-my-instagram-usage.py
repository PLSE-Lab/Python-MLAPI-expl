#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet


# # Introduction
# * Doing analysis and forecasting on my own Instagram data downloaded from the app
# * Settings -> Security -> Download Data
# * You will get a JSON file titled 'likes.json'
# * Convert to [CSV](https://www.convertcsv.com/json-to-csv.htm) for easy data processing
# ![App](https://raw.githubusercontent.com/vee-upatising/Instagram-Forecasting/master/instagram%20data.png)

# # Load Data

# In[ ]:


df = pd.read_csv('../input/convertcsv.csv')


# # Counting Likes Per Account

# In[ ]:


#counting each time I give an account a like
order_data = df['media_likes/1'].value_counts().iloc[:10].index
sns.catplot(x='media_likes/1',
           kind='count',
            height=8, 
            aspect=4,
           data=df, order = order_data)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel('Instagram Account', size = 20)
plt.ylabel('Likes Given', size = 20)


# # Counting Likes Per Day

# In[ ]:


#dropping which account I liked, and comment likes
df = df.drop(['comment_likes/0','media_likes/1','comment_likes/1'], axis=1)


# In[ ]:


#Assigning one like instance to each date
df['Likes'] = 1
df.columns = ['Date','Likes']
df.Date = pd.to_datetime(df.Date)
df.set_index('Date', inplace = True)
df.head()


# # Data Processing for FB Prophet
# * The library requires a dataframe with two columns (ds and y)
# * ds column must be in DateTime format

# In[ ]:


#summing up every like that happens in a given 24 hour period
likes_df = df.resample('D').size().reset_index()
df = pd.DataFrame(likes_df)
df.columns = ['ds','y']
df.head()


# In[ ]:


p = Prophet(yearly_seasonality=True,changepoint_prior_scale=0.9)
p.fit(df)


# In[ ]:


future = p.make_future_dataframe(periods = 365, include_history = True)
forecast = p.predict(future)


# # Trends
# * With weekly and yearly seasonality enabled

# In[ ]:


figure2 = p.plot_components(forecast)


# # Forecasting
# Predicting the trend of my Instgram usage into 2021

# In[ ]:


figure = p.plot(forecast, xlabel='Date', ylabel='Instagram Likes Given')

