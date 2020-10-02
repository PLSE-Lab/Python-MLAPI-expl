#!/usr/bin/env python
# coding: utf-8

# # Simple model: Just using the last known 28 days, which should be the most useful since they happened most recently, we use the average demand, grouped by weekday. 

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime, timedelta

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


PATH = '/kaggle/input/m5-forecasting-accuracy/'
cal = pd.read_csv(f'{PATH}calendar.csv')
# sell_prices = pd.read_csv(f'{PATH}sell_prices.csv')
ss = pd.read_csv(f'{PATH}sample_submission.csv')
stv = pd.read_csv(f'{PATH}sales_train_validation.csv')


# In[ ]:


# We need to select only the last 28 days
last_28 = stv.iloc[:, pd.np.r_[0,-28:0]]
last_28.head()


# In[ ]:


# melt the days into the d column and the values into the demand cloumn
last = last_28.melt('id', var_name='d', value_name='demand')
last.head()


# In[ ]:


# merge with calander to use the dates to aggregate
last = last.merge(cal)
last.head()


# In[ ]:


# get the demand for each product, grouped by weekday
by_weekday = last.groupby(['id','wday'])['demand'].mean()


# In[ ]:


# make a copy of the sample submission
sub = ss.copy()
# change the column names to match the last 28 days
sub.columns = ['id'] + ['d_' + str(1914+x) for x in range(28)]
# select only the rows with an id with the validation tag
sub = sub.loc[sub.id.str.contains('validation')]


# In[ ]:


# melt this dataframe and merge it with the calendar so we can join it with by_weekday dataframe
sub = sub.melt('id', var_name='d', value_name='demand')
sub = sub.merge(cal)[['id', 'd', 'wday']]
df = sub.join(by_weekday, on=['id', 'wday'])
df.head()


# In[ ]:


# pivot df to get it into the proper format for submission
df = df.pivot(index='id', columns='d', values='demand')
# need to reset index to take care of columns. comment next line out to see what i mean 
df.reset_index(inplace=True)
df.head()


# I notice that I have the id column out of order. Therefore I will merge a copy of the submission file to df so that the items will be in the right order. I'm not sure if this is necessary.

# In[ ]:



submission = ss[['id']].copy()


# In[ ]:


submission = submission.merge(df)


# In[ ]:


# we must copy the dataframe to match the format of the submission file which is twice as long as what we have
submission = pd.concat([submission, submission], axis=0)


# In[ ]:


# reset the id colum to have the same values as the sample submission
submission['id'] = ss.id.values


# In[ ]:


# rename the columns to match the sample submission format 
submission.columns = ['id'] + ['F' + str(i) for i in range(1,29)]
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




