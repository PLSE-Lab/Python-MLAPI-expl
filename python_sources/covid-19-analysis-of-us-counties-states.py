#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Data of US corona virus cases uptil 2020-07-09

# In[ ]:


pd.set_option('max_rows',100)
df=pd.read_csv("/kaggle/input/us-counties-covid-19-dataset/us-counties.csv")

df['date']=pd.to_datetime(df['date'],yearfirst=True)
df.set_index('date',inplace=True)
df.head(100)


# In[ ]:


total_cases=df.loc[pd.Timestamp('2020-07-09')]
total_cases.head(10)


# **I have grouped data by State and calculated the total cases of each State up till 2020-07-09**

# In[ ]:


total=total_cases.groupby(['state']).agg({'cases':['sum']})
total.columns=['total_cases']


# In[ ]:


cases=total.nlargest(10,['total_cases'])
plt.figure(figsize=(8,8))

sns.barplot(x='total_cases',y=cases.index,data=cases,orient='h')
plt.title('Top 10 States in terms of number of cases')
plt.ylabel('State')
plt.xlabel('Total number of Cases')
ax=plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# **I have grouped data by State and calculated the total deaths each State uptil 2020-07-09**

# In[ ]:


death=total_cases.groupby(['state']).agg({'deaths':['sum']})
death.columns=['total_deaths']
death.head(10)


# In[ ]:


deaths=death.nlargest(10,['total_deaths'])
plt.figure(figsize=(8,8))

sns.barplot(x='total_deaths',y=deaths.index,data=deaths,orient='h')
plt.title('Top 10 States in terms of number of deaths')
plt.ylabel('State')
plt.xlabel('Total number of deaths')
ax=plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# In[ ]:


dates=[]
data=[]
dates.append('January')
data.append(df.loc[df.index==pd.Timestamp('2020-01-31')]['cases'].sum())

dates.append('February')
data.append(df.loc[df.index==pd.Timestamp('2020-02-29')]['cases'].sum()-data[0])

dates.append('March')
data.append(df.loc[df.index==pd.Timestamp('2020-03-31')]['cases'].sum()-data[1])

dates.append('April')
data.append(df.loc[df.index==pd.Timestamp('2020-04-30')]['cases'].sum()-data[2])

dates.append('May')
data.append(df.loc[df.index==pd.Timestamp('2020-05-31')]['cases'].sum()-data[3])

dates.append('June')
data.append(df.loc[df.index==pd.Timestamp('2020-06-30')]['cases'].sum()-data[4])

dates.append('July')
data.append(df.loc[df.index==pd.Timestamp('2020-07-09')]['cases'].sum()-data[5])


# In[ ]:


plt.figure(figsize=(8,8))

sns.barplot(x=dates,y=data)
plt.xlabel('Month')
plt.ylabel('Total number of cases')
plt.title('Cases per Month')

for i in range(7):
    plt.text(x=i,y=data[i]+10000,s=data[i],ha='center')
    
    
ax=plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# In[ ]:


dates1=[]
data1=[]
dates1.append('January')
data1.append(df.loc[df.index==pd.Timestamp('2020-01-31')]['deaths'].sum())

dates1.append('February')
data1.append(df.loc[df.index==pd.Timestamp('2020-02-29')]['deaths'].sum()-data1[0])

dates1.append('March')
data1.append(df.loc[df.index==pd.Timestamp('2020-03-31')]['deaths'].sum()-data1[1])

dates1.append('April')
data1.append(df.loc[df.index==pd.Timestamp('2020-04-30')]['deaths'].sum()-data1[2])

dates1.append('May')
data1.append(df.loc[df.index==pd.Timestamp('2020-05-31')]['deaths'].sum()-data1[3])

dates1.append('June')
data1.append(df.loc[df.index==pd.Timestamp('2020-06-30')]['deaths'].sum()-data1[4])

dates1.append('July')
data1.append(df.loc[df.index==pd.Timestamp('2020-07-09')]['deaths'].sum()-data1[5])


# In[ ]:


plt.figure(figsize=(8,8))
sns.barplot(x=dates1,y=data1)
plt.xlabel('Month')
plt.ylabel('Total number of deaths')
#plt.ylim(0,30000)
plt.title('Deaths per month')

for i in range(7):
    plt.text(x=i,y=data1[i]+1000,s=data1[i],ha='center')
    i+=1
    
ax=plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# In[ ]:


df.reset_index(inplace=True)
date=df.groupby(pd.Grouper(key='date',freq='1D')).agg({'cases':['sum'],'deaths':['sum']})
date.columns=['total_cases','total_deaths']
#date['total_cases']=np.cumsum(date['total_cases'])
#date['total_deaths']=np.cumsum(date['total_deaths'])
date


# In[ ]:


plt.figure(figsize=(12,12))

sns.lineplot(x=date.index,y='total_cases',data=date)
plt.title('Number of cases up till now')
plt.xticks(np.array(pd.date_range('2020-01-21','2020-07-09',freq='5D')),rotation=90)
plt.xlabel('Date')
plt.ylabel('Number of Cases')
ax=plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# In[ ]:


counties=total_cases.groupby(['county']).agg({'cases':['max']})
counties.columns=['total_cases']
counties

