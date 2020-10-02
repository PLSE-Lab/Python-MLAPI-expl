#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df=pd.read_csv('../input/us-border-crossing-data/Border_Crossing_Entry_Data.csv')
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **1 - Data Overview**

# In[ ]:


df.info()


# In[ ]:


df['Date'].unique()


# Since there are no registers of hours (all registers are equal 00:00), we will not consider hours in analysis. The same is true for days.

# In[ ]:


#Transforming column date to date type in pandas
df['Date']=pd.to_datetime(df['Date'])
df['Date'].head()


# In[ ]:


#Spliting date column in separated date (year, month and day)
df['Year']=df['Date'].dt.year
df['Month']=df['Date'].dt.month


# In[ ]:


#Dropping Date collumn
df.drop(columns='Date',inplace=True)


# In[ ]:


#Replacing some values
df['Border'].replace({'US-Canada Border':'US-CAN','US-Mexico Border':'US-MEX'},inplace=True)


# In[ ]:


df.head()


# # **2 - Analysing data**

# ** Amount of people crossing the borders over the years**

# In[ ]:


sum_crossing=df.groupby(['Year']).sum()['Value'].reset_index()


# In[ ]:


plt.figure(figsize=(15,5))
plt.grid()
sns.set_style('dark')
sns.barplot(x='Year',y='Value',data=sum_crossing)
plt.title('Amount per year')


# The number of people crossing has reduced in the last years. Maybe because USA government have made more difficult the entrance in the country.

# **Flow per Month**

# In[ ]:


sum_month=df.groupby('Month').sum()['Value'].reset_index()


# In[ ]:


plt.figure(figsize=(15,5))
plt.grid()
sns.set_style('dark')
sns.barplot(x='Month',y='Value',data=sum_month)


# It looks the higher flow occurs between months 6 and 9

# **Comparing Borders**

# In[ ]:


border=df.groupby('Border').sum()['Value'].reset_index()


# In[ ]:


plt.figure(figsize=(7,4))
sns.barplot(x='Border',y='Value',data=border)
plt.title('Amount per Border')


# In[ ]:


mexico_sum=sum(border[border['Border']=='US-MEX']['Value'])
canada_sum=sum(border[border['Border']=='US-CAN']['Value'])
total=mexico_sum+canada_sum


# In[ ]:


#Percentage related to mexico entries
mexico_sum/total*100


# In[ ]:


#Percentage related to canada entries
canada_sum/total*100


# Most of the crossing are from borders between USA and Mexico

# **Type of vehicle**

# In[ ]:


vehicle_sum=df.groupby(['Measure']).sum()['Value'].reset_index().sort_values('Value',ascending=False)


# In[ ]:


sns.barplot(x='Value',y='Measure',data=vehicle_sum)


# **Type of vehicle per Border**

# In[ ]:


vehicle_sum=df.groupby(['Measure','Border']).sum()['Value'].reset_index().sort_values('Value',ascending=False)


# In[ ]:


plt.figure(figsize=(15,5))
plt.grid()
sns.set_style('dark')
sns.barplot(x='Value',y='Measure',hue='Border',data=vehicle_sum)


# Mexico has a considerable number of pedestrians entrance in relation to Canada

# **Which Port is more used?**

# In[ ]:


port_sum=df.groupby(['Port Name','Border']).sum()['Value'].reset_index().sort_values('Value',ascending=False)


# In[ ]:


#20 more used considering MEX and CAN
plt.figure(figsize=(15,7))
plt.grid()
sns.set_style('dark')
sns.barplot(x='Value',y='Port Name',hue='Border',data=port_sum.iloc[:20])


# In[ ]:




