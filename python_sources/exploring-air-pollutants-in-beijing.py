#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.metrics import mean_squared_error,r2_score
import glob
import missingno as msno
from fbprophet import Prophet
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10, 7)
from datetime import datetime, timedelta

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


colors = ["windows blue", "amber", "faded green", "dusty purple"]
sns.set(rc={"figure.figsize": (20,10), "axes.titlesize" : 18, "axes.labelsize" : 12, 
            "xtick.labelsize" : 14, "ytick.labelsize" : 14 })


# In[ ]:


path =r'../input/beijing-multisite-airquality-data-set/' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)

cols = ['No', 'year', 'month', 'day', 'hour', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP',
       'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station']
frame = frame[cols]
frame = frame.sort_values(['station', 'year'])

frame.tail(3)


# In[ ]:


frame.describe()


# In[ ]:


frame.info()


# In[ ]:


frame.shape


# In[ ]:


#summarising number of missing values in each column
frame.isnull().sum()


# In[ ]:


# percentage of missing values in each column
round(frame.isnull().sum()/len(frame.index), 2)*100


# In[ ]:


frame['PM2.5'].describe()


# In[ ]:


frame['PM2.5'].fillna(frame['PM2.5'].median(), inplace=True)


# In[ ]:


# percentage of missing values in each column
round(frame.isnull().sum()/len(frame.index), 2)*100


# In[ ]:


frame['PM10'].describe()


# # Imputing Missing Values

# In[ ]:


frame['PM10'].fillna(frame['PM10'].median(), inplace=True)
frame['SO2'].fillna(frame['SO2'].median(), inplace=True)
frame['NO2'].fillna(frame['NO2'].mean(), inplace=True)
frame['CO'].fillna(frame['CO'].median(), inplace=True)
frame['O3'].fillna(frame['O3'].median(), inplace=True)


# In[ ]:


# percentage of missing values in each column
round(frame.isnull().sum()/len(frame.index), 2)*100


# In[ ]:


frame.head(15)


# In[ ]:


df = frame[['SO2','year','station']].groupby(["year"]).median().reset_index().sort_values(by='year',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='year', y='SO2', data=df)


# In[ ]:


df = frame[['NO2','year','station']].groupby(["year"]).median().reset_index().sort_values(by='year',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='year', y='NO2', data=df,markers='o', color='red')


# In[ ]:


df = frame[['CO','year','station']].groupby(["year"]).median().reset_index().sort_values(by='year',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='year', y='CO', data=df,markers='o', color='olive')


# In[ ]:


df = frame[['O3','year','station']].groupby(["year"]).median().reset_index().sort_values(by='year',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='year', y='O3', data=df,markers='o', color='blue')


# In[ ]:


df = frame[['PM2.5','year','station']].groupby(["year"]).median().reset_index().sort_values(by='year',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='year', y='PM2.5', data=df,markers='o', color='red')


# In[ ]:


df = frame[['PM10','year','station']].groupby(["year"]).median().reset_index().sort_values(by='year',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='year', y='PM10', data=df,markers='o', color='red')


# In[ ]:


frame.head()


# In[ ]:


df_2017 = frame[frame['year']==2017]


# In[ ]:


df_2017.head()


# # Hourly Analysis of Pollutants in 2017

# In[ ]:


df = df_2017[['SO2','hour','station']].groupby(["hour"]).median().reset_index().sort_values(by='hour',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='hour', y='SO2', data=df)


# In[ ]:


df = df_2017[['NO2','hour','station']].groupby(["hour"]).median().reset_index().sort_values(by='hour',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='hour', y='NO2', data=df)


# In[ ]:


df = df_2017[['CO','hour','station']].groupby(["hour"]).median().reset_index().sort_values(by='hour',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='hour', y='CO', data=df)


# In[ ]:


df = df_2017[['O3','hour','station']].groupby(["hour"]).median().reset_index().sort_values(by='hour',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='hour', y='O3', data=df)


# In[ ]:


df = df_2017[['PM2.5','hour','station']].groupby(["hour"]).median().reset_index().sort_values(by='hour',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='hour', y='PM2.5', data=df)


# In[ ]:


df = df_2017[['PM10','hour','station']].groupby(["hour"]).median().reset_index().sort_values(by='hour',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='hour', y='PM10', data=df)


# In[ ]:


df = df_2017[['WSPM','hour','station']].groupby(["hour"]).median().reset_index().sort_values(by='hour',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='hour', y='WSPM', data=df)


# In[ ]:


df = df_2017[['TEMP','hour','station']].groupby(["hour"]).median().reset_index().sort_values(by='hour',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='hour', y='TEMP', data=df)


# In[ ]:


df = df_2017[['PRES','hour','station']].groupby(["hour"]).median().reset_index().sort_values(by='hour',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='hour', y='PRES', data=df)


# In[ ]:


df = df_2017[['WSPM','hour','station']].groupby(["hour"]).median().reset_index().sort_values(by='hour',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='hour', y='WSPM', data=df)


# In[ ]:


df = df_2017[['DEWP','hour','station']].groupby(["hour"]).median().reset_index().sort_values(by='hour',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='hour', y='DEWP', data=df)


# In[ ]:


df_2017.dtypes


# In[ ]:


# creating date field for further analysis by extracting day of the week, month etc.
df_2017['date']=pd.to_datetime(df_2017[['year', 'month', 'day']])


# In[ ]:


df_2017.dtypes


# In[ ]:


df_2017.head()


# In[ ]:


# function to find day of the week based on the date field
import calendar
def findDay(date): 
    dayname = calendar.day_name[date.weekday()]
    return dayname


# In[ ]:


df_2017['day_week'] = df_2017['date'].apply(lambda x: findDay(x))
df_2017.head()


# In[ ]:


df_2017.tail()


# # Analysis of Pollutants in day of the week

# In[ ]:


custom_day = {'Monday':0, 'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}


# In[ ]:



df = df_2017[['SO2','day_week','station']].groupby(["day_week"]).median().reset_index().sort_values(by='day_week',ascending=True)
df = df.iloc[df['day_week'].map(custom_day).argsort()]
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='day_week', y='SO2', data=df)


# In[ ]:


df = df_2017[['NO2','day_week','station']].groupby(["day_week"]).median().reset_index().sort_values(by='day_week',ascending=True)
df = df.iloc[df['day_week'].map(custom_day).argsort()]
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='day_week', y='NO2', data=df)


# In[ ]:


df = df_2017[['O3','day_week','station']].groupby(["day_week"]).median().reset_index().sort_values(by='day_week',ascending=True)
df = df.iloc[df['day_week'].map(custom_day).argsort()]
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='day_week', y='O3', data=df)


# In[ ]:


df = df_2017[['CO','day_week','station']].groupby(["day_week"]).median().reset_index().sort_values(by='day_week',ascending=True)
df = df.iloc[df['day_week'].map(custom_day).argsort()]
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='day_week', y='CO', data=df)


# # Analysis of Pollutants monthwise in a year 2017

# converting month values to month name for better understanding

# In[ ]:


frame.month.replace([1,2,3,4,5,6,7,8,9,10,11,12], ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], inplace=True)


# In[ ]:


frame['month'].value_counts()


# In[ ]:


custom_dict = {'Jan':0, 'Feb':1,'Mar':2,'Apr':3,'May':4,'Jun':5,'Jul':6,'Aug':7,'Sep':8,'Oct':9,'Nov':10,'Dec':11}
df = frame[['SO2','month','station']].groupby(["month"]).median().reset_index().sort_values(by='month',ascending=True)
df = df.iloc[df['month'].map(custom_dict).argsort()]
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='month', y='SO2', data=df)


# In[ ]:


df = frame[['NO2','month','station']].groupby(["month"]).median().reset_index().sort_values(by='month',ascending=True)
df = df.iloc[df['month'].map(custom_dict).argsort()]
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='month', y='NO2', data=df)


# In[ ]:


df = frame[['O3','month','station']].groupby(["month"]).median().reset_index().sort_values(by='month',ascending=True)
df = df.iloc[df['month'].map(custom_dict).argsort()]
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='month', y='O3', data=df)


# In[ ]:


df = frame[['CO','month','station']].groupby(["month"]).median().reset_index().sort_values(by='month',ascending=True)
df = df.iloc[df['month'].map(custom_dict).argsort()]
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='month', y='CO', data=df)


# In[ ]:


df = frame[['PM2.5','month','station']].groupby(["month"]).median().reset_index().sort_values(by='month',ascending=True)
df = df.iloc[df['month'].map(custom_dict).argsort()]
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='month', y='PM2.5', data=df)


# In[ ]:


df = frame[['PM10','month','station']].groupby(["month"]).median().reset_index().sort_values(by='month',ascending=True)
df = df.iloc[df['month'].map(custom_dict).argsort()]
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='month', y='PM10', data=df)


# In[ ]:


frame['station'].value_counts()


# In[ ]:


f = plt.figure(figsize=(30,8))
sns.boxplot(x='station', y='SO2', data=frame.dropna(axis=0).reset_index())


# In[ ]:


f = plt.figure(figsize=(30,8))
sns.boxplot(x='station', y='NO2', data=frame.dropna(axis=0).reset_index())


# In[ ]:


f = plt.figure(figsize=(30,8))
sns.boxplot(x='station', y='CO', data=frame.dropna(axis=0).reset_index())


# In[ ]:


f = plt.figure(figsize=(30,8))
sns.boxplot(x='station', y='O3', data=frame.dropna(axis=0).reset_index())


# In[ ]:




