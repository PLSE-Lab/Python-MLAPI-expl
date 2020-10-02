#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime
from types import *


# In[2]:


train = pd.read_csv('../input/train.csv')
train.info()


# In[3]:


test = pd.read_csv('../input/test.csv')
test.info()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


# split numbers using commas and show only 2 decimal points
pd.set_option('display.float_format', '{:,.2f}'.format)
print(train.describe())


# In[ ]:


print(train.count())


# In[ ]:


train.describe(include='all')


# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# In[4]:


train.sort_values(by='revenue', ascending=False).head(20)[['title','revenue','release_date']]


# In[5]:


sns.jointplot(x="popularity", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


train['status'].value_counts()


# In[ ]:


test['status'].value_counts()


# In[6]:


test.loc[test['release_date'].isnull() == True, 'release_date'] = '01/01/98'


# In[7]:


def date_features(df):
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_day'] = df['release_date'].dt.day
    df['release_quarter'] = df['release_date'].dt.quarter
    df['release_dayofweek'] = df['release_date'].dt.dayofweek
    ''' year = df['release_year']
    now = datetime.datetime.now()
    if year> now.year:
        year=year-100
    else:
        year=year '''
    #df['budget_adj'] = float((float(df['budget'])*1000)/1900)
    #dummies = pd.get_dummies(df['release_dayofweek'])
    #print(dummies)
    ''' df['mon','tue','wed','thu','fri','sat','sun'] = 0
    dayOfWeek = df['release_date'].dt.dayofweek
    dayOfWeek = dayOfWeek.astype(int) ''' 
    #print('hi')
    #print(dayOfWeek,"hi"+str(dayOfWeek))
    #print('hi')
    #print(type(dayOfWeek))
    '''
    print(dayOfWeek)
    if dayOfWeek.astype(int)==0:
        df['mon']=1
    elif dayOfWeek.astype(int)==1:
        df['tue']=1   
    elif dayOfWeek.astype(int)==1:
        df['wed']=1
    elif dayOfWeek.astype(int)==1:
        df['thu']=1
    elif dayOfWeek.astype(int)==1:
        df['fri']=1
    elif dayOfWeek.astype(int)==1:
        df['sat']=1
    elif dayOfWeek.astype(int)==1:
        df['sun']=1 
         '''
    #df.drop(columns=['release_date'], inplace=True)
    return df

train=date_features(train)
#test=date_features(test)
train.sort_values(by='release_year',ascending=False).head(10)[['title','revenue','release_year']]


# In[9]:


#Train
now = datetime.datetime.now()
train.loc[train['release_year'] > now.year, 'release_year'] = train.loc[train['release_year'] >  now.year, 'release_year'].apply(lambda x: x - 100)
train.sort_values(by='release_year',ascending=False).head(10)[['title','revenue','release_year']]
#dummies = pd.get_dummies(train['release_dayofweek'])
#train=pd.concat([train,dummies],axis=1)
names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

for i, x in enumerate(names):
    train[x] = (train['release_dayofweek'] == i).astype(int)


# In[ ]:


''' train = train.drop(['0'], axis=1)
train = train.drop(['1'], axis=1)
train = train.drop(['2'], axis=1)
train = train.drop(['3'], axis=1)
train = train.drop(['4'], axis=1)
train = train.drop(['5'], axis=1)
train = train.drop(['6'], axis=1)
train = train.drop(['7'], axis=1) '''
train.info()


# In[10]:


sns.jointplot(x="release_year", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="release_month", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="release_dayofweek", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Monday", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Tuesday", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Wednesday", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Thursday", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Friday", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Saturday", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Sunday", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[11]:


names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for i, x in enumerate(names):
    train[x] = (train['release_month'] == i+1).astype(int)
train.info()


# In[ ]:


sns.jointplot(x="Jan", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Feb", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Mar", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Apr", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="May", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Jun", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Jul", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Aug", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Sep", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Oct", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Nov", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[ ]:


sns.jointplot(x="Dec", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()


# In[23]:


train['budget_inflation_varient_val']=train['budget']/train['release_year']
#for i in train:
  #  print(i['budget'])
    #train['budget_inflation_varient_val'] = x/


# In[24]:


sns.jointplot(x="budget_inflation_varient_val", y="revenue", data=train, height=11, ratio=4, color="g")
plt.show()

