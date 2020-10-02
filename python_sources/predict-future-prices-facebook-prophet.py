#!/usr/bin/env python
# coding: utf-8

# # 1. Import Libraries and Dataset

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from fbprophet import Prophet


# In[ ]:


avocado_df = pd.read_csv('../input/avocado-prices/avocado.csv')


# In[ ]:


avocado_df.head()


# In[ ]:


avocado_df.tail(10)


# In[ ]:


avocado_df.describe()


# In[ ]:


avocado_df.info()


# In[ ]:


avocado_df.isnull().sum()


# # 2. Explore Dataset  

# In[ ]:


avocado_df = avocado_df.sort_values('Date')


# In[ ]:


plt.figure(figsize = (10, 10))
plt.plot(avocado_df['Date'], avocado_df['AveragePrice'])


# In[ ]:


plt.figure(figsize = (10, 6))
sns.distplot(avocado_df['AveragePrice'], color = 'b')


# In[ ]:


sns.violinplot(y = 'AveragePrice', x = 'type', data = avocado_df)


# In[ ]:


sns.set(font_scale=0.7) 
plt.figure(figsize=[25,12])
sns.countplot(x = 'region', data = avocado_df)
plt.xticks(rotation = 45)


# In[ ]:


sns.set(font_scale=1.5) 
plt.figure(figsize=[25,12])
sns.countplot(x = 'year', data = avocado_df)
plt.xticks(rotation = 45)


# In[ ]:


conventional = sns.catplot('AveragePrice', 'region', data = avocado_df[avocado_df['type']=='conventional'], 
                           hue = 'year',
                           height = 20)


# In[ ]:


organic = sns.catplot('AveragePrice', 'region', data = avocado_df[avocado_df['type']=='organic'],
                      hue = 'year',
                      height = 20)


# # 3. Prepare the data before aplying Facebook Prophet Tool 

# In[ ]:


avocado_df


# In[ ]:


avocado_prophet_df = avocado_df[['Date', 'AveragePrice']]


# In[ ]:


avocado_prophet_df


# In[ ]:


avocado_prophet_df = avocado_prophet_df.rename(columns = {'Date':'ds', 'AveragePrice':'y'})


# In[ ]:


avocado_prophet_df


# # 4. Develop model and make Prediction

# In[ ]:


m = Prophet()
m.fit(avocado_prophet_df)


# In[ ]:


# Forcasting into the future
future = m.make_future_dataframe(periods = 365)
forecast = m.predict(future)


# In[ ]:


figure = m.plot(forecast, xlabel = 'Date', ylabel = 'Price')


# In[ ]:


figure2 = m.plot_components(forecast)


# # 5.  Develop Model and make Prediction(Region Specific)

# In[ ]:


avocado_df = pd.read_csv('../input/avocado-prices/avocado.csv')


# In[ ]:


# Select specific region
avocado_df_sample = avocado_df[avocado_df['region']=='West']


# In[ ]:


avocado_df_sample = avocado_df_sample.sort_values('Date')


# In[ ]:


plt.plot(avocado_df_sample['Date'], avocado_df_sample['AveragePrice'])


# In[ ]:


avocado_df_sample = avocado_df_sample.rename(columns = {'Date':'ds', 'AveragePrice':'y'})


# In[ ]:


m = Prophet()
m.fit(avocado_df_sample)
# Forcasting into the future
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[ ]:


figure = m.plot(forecast, xlabel='Date', ylabel='Price')


# In[ ]:


figure3 = m.plot_components(forecast)


# ## Similarly We can predict for all other Regions

# In[ ]:




