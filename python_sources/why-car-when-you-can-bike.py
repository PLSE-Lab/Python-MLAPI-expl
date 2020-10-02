#!/usr/bin/env python
# coding: utf-8

# In this kernel we will explore the dataset.Dataset has data for two years spanning 19 days in each month.Then we will predict the bike Count using Machine learning.This kernel is a work in process.If you like my work please do vote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Importing the Dataset**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


# In[ ]:


df=pd.read_csv('../input/bike-sharing-demand/train.csv',parse_dates=['datetime'],index_col=0)
df_test=pd.read_csv('../input/bike-sharing-demand/test.csv',parse_dates=['datetime'],index_col=0)
#df.head()


# In[ ]:


#df.info()


# **Summary of Dataset**

# In[ ]:


print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())
print('\nMissing values    :',df.isnull().values.sum())
print('\nUnique values :  \n',df.nunique())


# We have no missing values in the Dataset

# **Converting to Datetime to Month,day,hour etc**

# In[ ]:


def add_feature(df):
    df['year']=df.index.year
    df['month']=df.index.month
    df['day']=df.index.day
    df['dayofweek']=df.index.dayofweek
    df['hour']=df.index.hour


# In[ ]:


add_feature(df)
add_feature(df_test)


# In[ ]:


df.tail(1)


# **Plot the count**

# In[ ]:


plt.title('Rental Count - Gaps')
df['2011-02':'2011-03']['count'].plot()
plt.show()


# We can see that data is available for first 19 days of the week.Rest of the days of the month data is not available.

# **Hourly Rental Change**

# In[ ]:


plt.plot(df['2011-01-01']['count'])
plt.xticks(fontsize=14,rotation=45)
plt.xlabel('Date')
plt.ylabel('Rental Count')
plt.title('Hourly Rentals for 01-Jan-2011')
plt.show()


# We can clearly see that the rental count is diiferent at different time of the day.

# **Monthly Demand**

# In[ ]:


plt.plot(df['2011-01']['count'])
plt.xticks(fontsize=14,rotation=45)
plt.xlabel('Date')
plt.ylabel('Rental Count')
plt.title('Rental for One Month')
plt.show()


# So demand shows lot of seasonality in the data.Fewer rentals on weekends compared to weekdays.

# In[ ]:


y=df.groupby('hour')['count'].mean()
plt.plot(y.index,y);
plt.xlabel('Hour')
plt.ylabel('Rental Count')
plt.xticks(np.arange(24));
plt.grid(True)
plt.title('Average Hourly Rental Count')
plt.show()


# So we can clearly see that the count increases at 8 am and 5 pm.This is possibly due to office timinings are 8 AM to 5 PM.

# **Year to Year Trend**

# In[ ]:


plt.plot(df['2011']['count'],label='2011')
plt.plot(df['2012']['count'],label='2012')
plt.xticks(fontsize=14,rotation=45)
plt.xlabel('Date')
plt.ylabel('Rental Count')
plt.title('2011 and 2012 Rentals (Year to Year)')
plt.show()


# So we can see from the above plot year on year the ride count is increasing.

# **How is ride count based on Month?**

# In[ ]:


ax=df.groupby('month')['count'].mean().plot('bar',color='blue',figsize=(15,6))
ax.set_xlabel('Month',fontsize=15)
ax.set_ylabel('Ride Count',fontsize=15)
ax.set_title('Number of Rides for per Month',fontsize=15)
ax.set_xticklabels(('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))
plt.show()


# We can see that the count of rides is more in May,Jun,Jul then steadly decreases in Nov and Dec.So more rides are happening in Summer months.In winter it will be difficult to do cycling.

# **Rides Per Year**

# In[ ]:


ax=df.groupby('year')['count'].mean().plot('bar',color='blue',figsize=(15,6))
ax.set_xlabel('Month',fontsize=15)
ax.set_ylabel('Number of Rides',fontsize=15)
ax.set_title('Number of Rides per Year',fontsize=15)
ax.set_xticklabels(('2011','2012'))
plt.show()


# We have more rides in Year 2012
