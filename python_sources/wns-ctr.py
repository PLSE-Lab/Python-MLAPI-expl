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
import datetime as dt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.style.use('ggplot')

import warnings
warnings.filterwarnings(action='ignore')


# In[ ]:


get_ipython().system('ls ..')


# In[ ]:


parse_date = lambda val : pd.datetime.strptime(val, '%Y-%m-%d %H:%M:%S')


# In[ ]:


train = pd.read_csv('../input/wns2019/train.csv', parse_dates=['impression_time'], date_parser=parse_date)
test = pd.read_csv('../input/wns2019/test.csv', parse_dates=['impression_time'], date_parser=parse_date)
sample = pd.read_csv('../input/wns2019/sample_submission.csv')


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# Target variable is 'is_click' which is 1 if there is a click and 0 otherwise

# In[ ]:


train.dtypes


# In[ ]:


df = pd.concat([train,test])


# In[ ]:


print(train.shape)
print(test.shape)
print(df.shape)


# In[ ]:


df.isna().sum()


# No empty values found

# In[ ]:


train.is_click.value_counts()/len(train)


# In[ ]:


positive_click_count = train.is_click.sum()
negative_click_count = train.is_click.count() - train.is_click.sum()
patches, texts = plt.pie([positive_click_count, negative_click_count], explode=(0.1,0))
plt.legend(patches, ['Ad Clicked', 'No click'], loc='best')
plt.show()


# # Univariate

# ## Exploration : impression_id

# In[ ]:


df.impression_id.nunique()


# Impression ID is unique for each entry, hence nothing much to explore here

# ## Exploration : app_code

# In[ ]:


df.app_code.nunique()


# In[ ]:


clicks_per_app = train.groupby('app_code').is_click.sum()
count_per_app = train.groupby('app_code').is_click.count()
ctr_per_app = clicks_per_app/count_per_app * 100


# In[ ]:


clicks_per_app.sort_values(ascending=False)[:50].plot(kind='bar', figsize=(12,6))
plt.ylabel('No. of Clicks')
plt.show()


# In[ ]:


ctr_per_app.sort_values(ascending=False)[:50].plot(kind='bar', figsize=(12,6))
plt.ylabel('CTR')
plt.show()


# In[ ]:


df[df.app_code==109]


# ## Exploration : is_4G

# In[ ]:


train.is_4G.value_counts()/len(train)


# In[ ]:


train.groupby('is_4G').is_click.sum().plot(kind='bar', figsize=(4,6))
plt.ylabel('CTR')
plt.show()


# Most of the clicks were by non-4G devices

# ## Exploration : os_version

# In[ ]:


print(train.os_version.value_counts()/len(train))


# In[ ]:


df.groupby('os_version').is_click.count().plot(kind='bar')
plt.ylabel('No. of clicks')
plt.show()


# In[ ]:


df.groupby('os_version').is_click.sum().plot(kind='bar')
plt.ylabel('CTR')
plt.show()


# ## Exploration : user_id

# In[ ]:


train.user_id.nunique()


# In[ ]:


train[train.is_click==1].user_id.value_counts()[:50].plot(kind='bar', figsize=(12,4))
plt.show()


# In[ ]:


train[train.user_id==54611]


# ## Exploration : impression_time

# In[ ]:


train.impression_time.describe()


# In[ ]:


df['impression_date'] = df.impression_time.apply(lambda x : x.date())
df['impression_hour'] = df.impression_time.apply(lambda x : x.hour)


# In[ ]:


df.head()


# In[ ]:


df.iloc[0].impression_date==dt.date(2018, 11, 15)


# In[ ]:


df['day_of_click'] = df.impression_time.apply(lambda x : x.weekday_name)
cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df.groupby('day_of_click').agg({'is_click':'sum'}).reindex(cats).plot()
plt.title('Clicks trend by week day')
(df.groupby('day_of_click').agg({'is_click':'sum'}).reindex(cats)/df.groupby('day_of_click').agg({'is_click':'count'}).reindex(cats)).plot()
ticks = list(range(0, 7, 1)) # points on the x axis where you want the label to appear
plt.title('CTR per week day')
plt.show()


# In[ ]:


df.groupby(['day_of_click', 'is_click']).size().unstack().plot(kind='bar', stacked=True, figsize=(12,5))
plt.show()


# Eventhough Tuesdays are having most clicks but CTR is lowest for Tuesdays and highest during Saturdays and Sundays

# In[ ]:


df.groupby(['impression_date']).is_click.sum().plot(figsize=(16,6))
plt.show()


# The drop in the curve around 2018-12-13 is due to the fact the from 2018-12-13 onwards is our test set and is unlabelled.

# In[ ]:


df.groupby(['impression_date', 'is_click']).size().unstack().plot(kind='bar', stacked=True, figsize=(12,6))
plt.show()


# In[ ]:


df.groupby('impression_hour').is_click.sum().plot(figsize=(20,6))
plt.xticks([i for i in range(24)])
plt.xlabel('Hour of day')
plt.ylabel('No. of clicks')
plt.title('Impression time')
plt.show()


# In[ ]:


df.groupby(['impression_hour', 'is_click']).size().unstack().plot(kind='bar', stacked=True, figsize=(12,5))
plt.show()


# Most of the clicks were observed during midnight and between 19:00-22:00 hrs

# In[ ]:


df.head()


# ## Log Analysis

# In[ ]:


view_log = pd.read_csv('../input/wns2019/view_log.csv', parse_dates=['server_time'], date_parser=parse_date)
item_data = pd.read_csv('../input/wns2019/item_data.csv')


# In[ ]:


view_log.head()


# In[ ]:


item_data.head()


# In[ ]:


print(view_log.shape)
print(item_data.shape)


# In[ ]:


view_log.head()


# In[ ]:


print(view_log.shape)
print(item_data.shape)


# In[ ]:


log_data = view_log.merge(item_data, how='left', on='item_id')


# In[ ]:


log_data.shape


# In[ ]:


session_count = view_log.groupby('user_id').session_id.nunique()
item_count = view_log.groupby('user_id').item_id.nunique()


# In[ ]:


view_log = view_log.merge(session_count, how='inner', on='user_id')
view_log = view_log.merge(item_count, how='inner', on='user_id')


# In[ ]:


view_log.head()


# In[ ]:


view_log.columns = ['server_time', 'device_type', 'session_id', 'user_id', 'item_id',
       'session_count', 'item_count']


# In[ ]:


view_log.head()


# In[ ]:


merger_log = view_log.drop(['server_time', 'session_id', 'item_id'], axis=1)


# In[ ]:


merger_log.drop_duplicates(inplace=True)


# In[ ]:


merger_log


# In[ ]:


df.head()


# In[ ]:


merger_log.shape


# In[ ]:


df_final = df.merge(merger_log, how='inner', on='user_id')


# In[ ]:


df.shape


# In[ ]:


df_final.shape


# In[ ]:


df_final.head(20)


# In[ ]:


df_final.drop(['impression_time', 'impression_date', ], axis=1, inplace=True)


# In[ ]:


df.isna().sum()


# In[ ]:


df_final.isna().sum()


# In[ ]:


df_final.head(20)


# In[ ]:


df_final.shape


# In[ ]:


df_final.dtypes


# In[ ]:


df_final[['app_code', 'impression_id', 'user_id', 'is_4G', 'is_click', 'impression_hour', 'day_of_click', 'device_type']] = df_final[['app_code', 'impression_id', 'user_id', 'is_4G', 
                                                                                                                     'is_click', 'impression_hour', 
                                                                                                                     'day_of_click', 'device_type']].astype('category')


# In[ ]:


df_final.dtypes


# In[ ]:


y= df_final.is_click


# In[ ]:


x_train = 

