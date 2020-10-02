#!/usr/bin/env python
# coding: utf-8

# ** Version 1**
# 
# Aim: 
# To see the data from various perspective,
#   * To see the difference between the pattern of visitors on days of holidays and non-holidays
#   * to see the restaurant wise visitor behavior

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
avd = pd.read_csv('../input/air_visit_data.csv')
asi = pd.read_csv('../input/air_store_info.csv')
hsi = pd.read_csv('../input/hpg_store_info.csv')
ar = pd.read_csv('../input/air_reserve.csv')
hr = pd.read_csv('../input/hpg_reserve.csv')
sid = pd.read_csv('../input/store_id_relation.csv')
tes = pd.read_csv('../input/sample_submission.csv')
hol = pd.read_csv('../input/date_info.csv')


# In[ ]:


plt.rcParams['figure.figsize'] = 16, 8


# In[ ]:


len(avd)


# In[ ]:





# In[ ]:


hol.head()


# In[ ]:


avd.head()


# In[ ]:


air_visit_date = pd.merge(avd, hol, how='left', left_on='visit_date', right_on='calendar_date')


# In[ ]:


air_visit_date.head()


# In[ ]:


air_visit_date.loc[air_visit_date['holiday_flg'] != 0].sort_values('visitors', ascending=False).head(10)


# In[ ]:


day_wise_df = air_visit_date.loc[air_visit_date['holiday_flg'] != 0].groupby('day_of_week').agg(sum)


# In[ ]:


sorter = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
sorterIndex = dict(zip(sorter,range(len(sorter))))
day_wise_df['day_id'] = day_wise_df.index
day_wise_df['day_id'] = day_wise_df['day_id'].map(sorterIndex)


# In[ ]:


day_wise_df.plot()
day_wise_df.plot(kind='bar')


# In[ ]:


day_wise_df


# In[ ]:


day_wise_df.head()


# In[ ]:


day_wise_df.sort_values('day_id', inplace=True)


# In[ ]:


day_wise_df


# ## The above Data was to see how visitors show pattern on days of holidays
# 
# # Observation: 
# * On days of holidays specially on Monday and Thrusday the visitors were high.
# 
# ## **Now we will see visitors behaviour on all days without holiday**

# In[ ]:


air_visit_date.loc[air_visit_date['holiday_flg'] == 0].sort_values('visitors', ascending=False).head(10)


# In[ ]:


day_wise_no_holiday_df = air_visit_date.loc[air_visit_date['holiday_flg'] == 0].groupby('day_of_week').agg(sum)


# In[ ]:


day_wise_no_holiday_df


# In[ ]:


day_wise_no_holiday_df['day_id'] = day_wise_no_holiday_df.index
day_wise_no_holiday_df['day_id'] = day_wise_no_holiday_df['day_id'].map(sorterIndex)


# In[ ]:


day_wise_no_holiday_df.sort_values('day_id', inplace=True)


# In[ ]:


day_wise_no_holiday_df


# In[ ]:


day_wise_no_holiday_df.plot()
day_wise_no_holiday_df.plot(kind='bar')


# ## Observation:
# 
# * Very clearly as we have expected, the visitors are higher on the weekends and incremental on weekdays

# In[ ]:




