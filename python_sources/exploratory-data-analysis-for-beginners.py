#!/usr/bin/env python
# coding: utf-8

# # Global AI Challenge 2020
# ### Assessment of Economic Impact of COVID-19

# ## Exploratory data analysis for Beginners 

# #### 1. Import the required libraries

# In[ ]:


import numpy as np             
import pandas as pd            
import matplotlib.pylab as plt
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# #### 2. Load data

# In[ ]:


df_crude_oil = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend_From1986-10-16_To2020-03-31.csv', header=0, index_col=0, parse_dates=True, squeeze=False)
df_train = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_train.csv', header=0, index_col=0, parse_dates=True, squeeze=False)
df_test = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_test.csv', header=0, index_col=0, parse_dates=True, squeeze=False)


# #### 3. Check the basic information about the data

# In[ ]:


#  To check the shape of the dataset 
print('The shape of crude oil price data:',df_crude_oil.shape)
print('The shape of train data :',df_train.shape)
print('The shape of test data:' , df_test.shape)


# **Observation**
# 
# The date feature is indexed.
# 
# The features are more than instance. our task is to find those features really impact the price rate.

# In[ ]:


df_crude_oil.tail()


# In[ ]:


df_train.tail()


# In[ ]:


df_test.tail()


# **Obervation**
# 
# The above visualization of few instances gives information that the train data and crude oil price data contains same date and price values. our take is how to use time series data and covid 19 data to predict crude oil price. 

# #### 3. Exploratory data analysis

# In[ ]:


df_crude_oil.describe()


# In[ ]:


plt.figure(figsize=(20,10))
plt.plot(df_crude_oil, color = 'red')
plt.xlabel('Date in year')
plt.ylabel('Oill Price ')
plt.show()


# **Observation**
# 
# The minimum price rate is 13.363850 and the max price rate is 112.504950. Between 2008 and 2009 the price rate reaches maximum due to issue in international market. The below link explains the huge fluctuation in curde oil price.
# 
# <https://oilprice.com/Energy/Energy-General/Why-This-Oil-Crisis-Is-Different-To-2008.html>

# In[ ]:


columns = [844]
df = df_train[df_train.columns[columns]]
plt.figure(figsize=(20,10))
plt.plot(df, color='blue')
plt.xlabel('Date')
plt.ylabel('price')


# **Observation**
# 
# The above plot shows how the curde oil price reduced due to COVID-19.

# In[ ]:


cols = [843]
df_1 = df_train[df_train.columns[cols]]
plt.figure(figsize=(20,10))
plt.plot(df_1, color='red')
plt.xlabel('Date')
plt.ylabel('world_new_deaths')


# In[ ]:


cols = [842]
df_2 = df_train[df_train.columns[cols]]
plt.figure(figsize=(20,10))
plt.plot(df_2, color='red')
plt.xlabel('Date')
plt.ylabel('World_total_deaths')


# In[ ]:


cols = [841]
df_3 = df_train[df_train.columns[cols]]
plt.figure(figsize=(20,10))
plt.plot(df_3, color='red')
plt.xlabel('Date')
plt.ylabel('World_new_cases')


# In[ ]:


cols = [840]
df_3 = df_train[df_train.columns[cols]]
plt.figure(figsize=(20,10))
plt.plot(df_3, color='red')
plt.xlabel('Date')
plt.ylabel('World_total_cases')


# # Corelation 

# In[ ]:


cols = [840,841,842,843,844]
df_corr= df_train[df_train.columns[cols]]

corr = df_corr.corr()

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df_corr.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df_corr.columns)
ax.set_yticklabels(df_corr.columns)
plt.show()


# In[ ]:


cols = [836,837,838,839, 844]
df_corr= df_train[df_train.columns[cols]]

corr = df_corr.corr()
fig =  plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df_corr.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df_corr.columns)
ax.set_yticklabels(df_corr.columns)
plt.show()


# In[ ]:


cols = [404,405,406,407, 844]
df_corr= df_train[df_train.columns[cols]]

corr = df_corr.corr()
fig =  plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df_corr.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df_corr.columns)
ax.set_yticklabels(df_corr.columns)
plt.show()


# #### The above corelation shown for few features
