#!/usr/bin/env python
# coding: utf-8

# **Importing all the necessary Libraries** 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.


# **Reading the file**

# In[ ]:


crime_data = pd.read_csv('../input/crime.csv', encoding="cp1252" )


# In[ ]:


crime_data.head()


# In[ ]:


print('There are '+str(crime_data.shape[0])+' incidents.')


# Filtering Crimes in year 2016-2017

# In[ ]:


crime_data = crime_data.loc[crime_data['YEAR'].isin([2016,2017])]
#droping unused column
crime_data = crime_data.drop(['INCIDENT_NUMBER','OFFENSE_CODE','UCR_PART', 'Location'], axis=1)
crime_data['OCCURRED_ON_DATE'] = pd.to_datetime(crime_data['OCCURRED_ON_DATE'])
crime_data.SHOOTING.fillna('N', inplace=True)
crime_data.DAY_OF_WEEK = pd.Categorical(crime_data.DAY_OF_WEEK, categories = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], ordered=True )
crime_data.Lat.replace(-1,None,inplace=True)
crime_data.Long.replace(-1,None,inplace=True)
rename = {'OFFENSE_CODE_GROUP':'Group',
         'OFFENSE_DESCRIPTION':'Description',
         'DISTRICT':'District',
         'REPORTING_AREA':'Area',
         'SHOOTING':'Shooting',
         'OCCURRED_ON_DATE':'Date',
         'YEAR':'Year',
         'MONTH':'Month',
         'DAY_OF_WEEK':'Day',
         'HOUR':'Hour',
         'STREET':'Street'}
crime_data.rename(index=str, columns=rename, inplace=True)
crime_data.head()


# Top

# In[ ]:


crime = crime_data.Group.value_counts()
crime.head()
crime = crime.to_frame()
crime.columns
plt.figure(figsize=(16,6))
sns.barplot(x=crime.index[:5],y='Group', data=crime.head())
plt.title('Most Popular Crime 2016-2017',fontsize=30)
plt.show()


# Crime Commited in a day

# In[ ]:


crime_week = crime_data[['Group','Year','Month','Day','Hour','Date']]
plt.figure(figsize=(16,6))
sns.countplot(x='Day', data=crime_week)
plt.title('Crime Commited in a day',fontsize=30)
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
sns.countplot(x='Hour', data = crime_week)
plt.title('Crime distribution in 24 hours')
plt.show()


# In[ ]:


sns.catplot(y='Group',data=crime_data,height=10,aspect=2, kind='count', order=crime_data.Group.value_counts().index)
plt.show()


# Where do crime mostly occur?

# In[ ]:


sns.scatterplot(x='Lat',y='Long', data=crime_data, alpha=0.01)
plt.show()


# In[ ]:


sns.scatterplot(x='Lat',y='Long', data=crime_data, alpha=0.01, hue='District')
plt.show()


# Summary from EDA:
# 1. Motor vechicle accident is most occured crime
# 2. Larceny is second most occured crime
# 3. Crime mostly occured in evening
# 4. Crime mostly occured in friday 
# 5. least in sunday

# In[ ]:




