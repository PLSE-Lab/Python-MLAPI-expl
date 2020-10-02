#!/usr/bin/env python
# coding: utf-8

# The purpose of this kernel is to visualize the sample data to get hopefully get some inspiration for feature engineering.
# 
# Summary:
# Though the sample is relatively smaller than the entire train dataset, but it can be observed that the click count and average attributed rate for other click count for each variable matters.
# Specifically, the attribution rate is positively correlated to the average attributed rate for other click count for that variable value, while negatively correlated to the click count for that variable value
# For click time, the day of the week, day of the year and the hour matt

# ## 1. Load

# In[2]:


# 1.1 Load Library--------------------------------------------------------------
# data analysis and wrangling
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


# In[3]:


# 1.2 Load data--------------------------------------------------------------
data_dir = '../input/'
dtypes = {
    'click_id'      : 'uint32',
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8'}
train_df = pd.read_csv(data_dir + 'train.csv', skiprows = range(500000, 184403890), dtype=dtypes) # 184,903,891 in total
print('train_df shape: ' + str( train_df.shape))


# ## 2. Data exploration and visualization

# In[4]:


#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.patches as mpatches
import seaborn as sns

import plotly.offline as py
import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')

#Configure Visualization Defaults
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set_style('white')

def get_barplot(df, factor):
    rate_df = df.groupby(factor)['is_attributed'].mean()
    print(rate_df.iplot(kind='bar', xTitle=factor, title='attributed rate'))
    count_df = df.groupby(factor)['is_attributed'].count()
    print(count_df.iplot(kind='bar', xTitle=factor, title='frequency count'))


# ### 2a. ip (click count and attributed rate)

# In[5]:


agg = train_df.groupby('ip').agg(dict(is_attributed = 'sum', app = 'count')).reset_index()
agg = agg.rename(columns = dict(is_attributed = 'ip_attribute_sum', app = 'ip_click_count'))
train_df = train_df.merge(agg, on = 'ip')
train_df['ip_click_count'] / 50
train_df['ip_click_count_fix'] = (round(train_df['ip_click_count'] / 1)).clip(0,30).astype(int)
get_barplot(train_df, 'ip_click_count_fix')
del agg
gc.collect()


# In[6]:


# attribute rate for the ip for other clicks
train_df['ip_other_click_attribute_rate'] = (train_df['ip_attribute_sum'] - train_df['is_attributed']) / (train_df['ip_click_count'] - 1)
attribute_rate_mean = train_df['ip_other_click_attribute_rate'].mean()
train_df['ip_other_click_attribute_rate'] = train_df.apply(lambda r:     r['ip_other_click_attribute_rate'] if (r['ip_click_count']> 10)     else attribute_rate_mean, axis = 1)
train_df['ip_other_click_attribute_rate'] = train_df['ip_other_click_attribute_rate'].fillna(attribute_rate_mean)

train_df['ip_other_click_attribute_rate_fix'] = round(train_df['ip_other_click_attribute_rate'], 2).clip(0,0.1)
get_barplot(train_df, 'ip_other_click_attribute_rate_fix')


# ### 2a. app (click count and attributed rate)

# In[11]:


agg = train_df.groupby('app').agg(dict(is_attributed = 'sum', ip = 'count')).reset_index()
agg = agg.rename(columns = dict(is_attributed = 'app_attribute_sum', ip = 'app_click_count'))
train_df = train_df.merge(agg, on = 'app')
train_df['app_click_count_fix'] = (round(train_df['app_click_count'] / 2000)).clip(0,30).astype(int)
get_barplot(train_df, 'app_click_count_fix')

del agg
gc.collect()


# In[ ]:


# attribute rate for the app for other clicks
train_df['app_other_click_attribute_rate'] = (train_df['app_attribute_sum'] - train_df['is_attributed']) / (train_df['app_click_count'] - 1)
attribute_rate_mean = train_df['app_other_click_attribute_rate'].mean()
train_df['app_other_click_attribute_rate'] = train_df.apply(lambda r:     r['app_other_click_attribute_rate'] if (r['app_click_count']> 1000)     else attribute_rate_mean, axis = 1)
train_df['app_other_click_attribute_rate'] = train_df['app_other_click_attribute_rate'].fillna(attribute_rate_mean)
train_df['app_other_click_attribute_rate_fix'] = round(train_df['app_other_click_attribute_rate'], 4).clip(0,0.0005)
get_barplot(train_df, 'app_other_click_attribute_rate_fix')


# ### 2c. device (click count and attributed rate)

# In[12]:


agg = train_df.groupby('device').agg(dict(is_attributed = 'sum', ip = 'count')).reset_index()
agg = agg.rename(columns = dict(is_attributed = 'device_attribute_sum', ip = 'device_click_count'))
train_df = train_df.merge(agg, on = 'device')
train_df['device_click_count_fix'] = (round(train_df['device_click_count'] / 5000)).clip(0,10).astype(int)
get_barplot(train_df, 'device_click_count_fix')

del agg
gc.collect()


# In[16]:


train_df['device_other_click_attribute_rate'] = (train_df['device_attribute_sum'] - train_df['is_attributed']) / (train_df['device_click_count'] - 1)
attribute_rate_mean = train_df['device_other_click_attribute_rate'].mean()
train_df['device_other_click_attribute_rate'] = train_df.apply(lambda r:     r['device_other_click_attribute_rate'] if (r['device_click_count']> 1000)     else attribute_rate_mean, axis = 1)
train_df['device_other_click_attribute_rate'] = train_df['device_other_click_attribute_rate'].fillna(attribute_rate_mean)
train_df['device_other_click_attribute_rate_fix'] = round(train_df['device_other_click_attribute_rate'], 4).clip(0,0.001)
get_barplot(train_df, 'device_other_click_attribute_rate_fix')


# ### 2d. os (click count and attributed rate)

# In[17]:


agg = train_df.groupby('os').agg(dict(is_attributed = 'sum', ip = 'count')).reset_index()
agg = agg.rename(columns = dict(is_attributed = 'os_attribute_sum', ip = 'os_click_count'))
train_df = train_df.merge(agg, on = 'os')
train_df['os_click_count_fix'] = (round(train_df['os_click_count'] / 1000)).clip(0,10).astype(int)
get_barplot(train_df, 'os_click_count_fix')

del agg
gc.collect()


# In[ ]:


# attribute rate for the os for other clicks
train_df['os_other_click_attribute_rate'] = (train_df['os_attribute_sum'] - train_df['is_attributed']) / (train_df['os_click_count'] - 1)
attribute_rate_mean = train_df['os_other_click_attribute_rate'].mean()
train_df['os_other_click_attribute_rate'] = train_df.apply(lambda r:     r['os_other_click_attribute_rate'] if (r['os_click_count']> 1000)     else attribute_rate_mean, axis = 1)
train_df['os_other_click_attribute_rate'] = train_df['os_other_click_attribute_rate'].fillna(attribute_rate_mean)
train_df['os_other_click_attribute_rate_fix'] = round(train_df['os_other_click_attribute_rate'], 4).clip(0,0.001)
get_barplot(train_df, 'os_other_click_attribute_rate_fix')


# ### 2e. channel (click count and attributed rate)

# In[ ]:


agg = train_df.groupby('channel').agg(dict(is_attributed = 'sum', ip = 'count')).reset_index()
agg = agg.rename(columns = dict(is_attributed = 'channel_attribute_sum', ip = 'channel_click_count'))
train_df = train_df.merge(agg, on = 'channel')
train_df['channel_click_count_fix'] = (round(train_df['channel_click_count'] / 1000)).clip(0,10).astype(int)
get_barplot(train_df, 'channel_click_count_fix')

del agg
gc.collect()


# In[ ]:


# attribute rate for the channel for other clicks
train_df['channel_other_click_attribute_rate'] = (train_df['channel_attribute_sum'] - train_df['is_attributed']) / (train_df['channel_click_count'] - 1)
attribute_rate_mean = train_df['channel_other_click_attribute_rate'].mean()
train_df['channel_other_click_attribute_rate'] = train_df.apply(lambda r:     r['channel_other_click_attribute_rate'] if (r['channel_click_count']> 1000)     else attribute_rate_mean, axis = 1)
train_df['channel_other_click_attribute_rate'] = train_df['channel_other_click_attribute_rate'].fillna(attribute_rate_mean)
train_df['channel_other_click_attribute_rate_fix'] = round(train_df['channel_other_click_attribute_rate'], 4).clip(0,0.001)
get_barplot(train_df, 'channel_other_click_attribute_rate_fix')


# ### 2f. click_time

# In[18]:


train_df['attributed_time_fix']= pd.to_datetime(train_df['attributed_time'])
train_df['click_time_fix'] = pd.to_datetime(train_df['click_time'])


# In[19]:


train_df['click_time_dayofweek'] = train_df['click_time_fix'].map(lambda x: x.dayofweek).astype(str)
train_df['click_time_dayofyear'] = train_df['click_time_fix'].map(lambda x: x.dayofyear).astype(int)
train_df['click_time_hour'] = train_df['click_time_fix'].map(lambda x: x.hour).astype(int)


# In[20]:


get_barplot(train_df,'click_time_dayofweek')


# In[21]:


get_barplot(train_df,'click_time_dayofyear')


# In[ ]:


get_barplot(train_df,'click_time_hour')

