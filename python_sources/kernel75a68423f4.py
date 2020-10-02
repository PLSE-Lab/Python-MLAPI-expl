#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/renfe.csv", index_col =0, parse_dates = ['insert_date', 'start_date', 'end_date'], infer_datetime_format = True)


# In[ ]:


df


# In[ ]:


df.dtypes


# In[ ]:


df.isna().sum()


# In[ ]:


df = df.dropna(how = 'any',subset = ['price'])


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize = (10,8))
sns.barplot(x='origin', y = 'price', data = df)
plt.xlabel('Origin')
plt.ylabel('Avg. Price')


# Highest Ticket prices from *Barcelona*

# Station by Station Analysis of costs.

# In[ ]:


plt.figure(figsize = (10,8))
sns.countplot(x='origin', data = df)
plt.xlabel('Origin')
plt.title('Station wise purchase of tickets')


# Most tickets sold at Madrid

# In[ ]:


mad = df.loc[(df['origin']=='MADRID'),:]

sev = df.loc[(df['origin']=='SEVILLA'),:]

pon = df.loc[df['origin']=='PONFERRADA',:]

barca = df.loc[(df['origin']=='BARCELONA'),:]

val = df.loc[(df['origin']=='VALENCIA'),:]


# In[ ]:


sns.set(style="whitegrid")
plt.figure(figsize = (20,30))
g = sns.catplot(x = 'train_type', y='price',hue = 'train_class', data=mad, col='destination', col_wrap = 2, kind = 'bar')
g.set_xticklabels(rotation=90)


# Maadrid - Barcelona highest cost and high-speed trains run

# In[ ]:


sns.set(style="whitegrid")
plt.figure(figsize = (20,30))
g = sns.catplot(x = 'train_type', y='price', hue = 'train_class', data=sev, col='destination', col_wrap = 2, kind = 'bar')
g.set_xticklabels(rotation=90)


# Preferente is costliest at Seville. AVE train costliest.

# In[ ]:


sns.set(style="whitegrid")
plt.figure(figsize = (20,30))
g = sns.catplot(x = 'train_type', y='price', hue='train_class', data=pon, col='destination', col_wrap = 2, kind = 'bar')
g.set_xticklabels(rotation=90)


# Cama G.Clase tickets costiest at Ponferrada. Threnotel train costliest

# In[ ]:


sns.set(style="whitegrid")
plt.figure(figsize = (20,30))
g = sns.catplot(x = 'train_type', y='price', hue='train_class', data=barca, col='destination', col_wrap = 2, kind = 'bar')
g.set_xticklabels(rotation=90)


# Preferente costliest at barca. Also, ave and ave-tgv costs dont vary much.

# In[ ]:


sns.set(style="whitegrid")
plt.figure(figsize = (20,30))
g = sns.catplot(x = 'train_type', y='price', hue='train_class', data=val, col='destination', col_wrap = 2, kind = 'bar')
g.set_xticklabels(rotation=90)


# *Turista Plus* tickets for AVE only, at highest costs at valencia.

# In[ ]:


plt.figure(figsize = (16,8))
sns.barplot(x='train_class', y = 'price', data = df)
plt.xlabel('Train Class')
plt.ylabel('Avg. Price')


# Cama G. Clase with highest average cost

# In[ ]:


plt.figure(figsize = (16,8))
g = sns.countplot(x = 'train_class', data = df)
plt.xticks(rotation=90)
plt.xlabel('Train Class')
plt.title('Popularity of Train Classes')


# Turista class is most popular. The trend is being followed that price of tickets most sold is *in between* cheapest and the middle of the distribution (OR) neither cheap nor very expensive.

# In[ ]:


plt.figure(figsize = (16,8))
g = sns.countplot(x = 'train_type', data = df)
plt.xticks(rotation=90)
plt.xlabel('Train Type')
plt.title('Popularity of Train Types')


# Most number of AVE trains run.

# In[ ]:


plt.figure(figsize = (16,8))
sns.barplot(x='train_type', y = 'price', data = df)
plt.xticks(rotation = 90)
plt.xlabel('Train Type')
plt.ylabel('Avg. Price')
plt.title('Price distribution of Train types')


# TGV train is the costliest (high-speed trains)

# In[ ]:


plt.figure(figsize = (16,8))
sns.barplot(x='fare', y = 'price', data = df)
plt.xticks(rotation = 90)
plt.xlabel('Fare Type')
plt.ylabel('Avg. Price')
plt.title('Price Distribution of Fare Types')


# Mesa Fare type is the most expensive followed by individual-flexible

# In[ ]:


plt.figure(figsize = (16,8))
sns.countplot(x='fare', data = df)
plt.xticks(rotation = 90)
plt.xlabel('Fare Type')
plt.title('Popularity of Fare Types')


# The trend for Fare types is it maintains 'central tendency'. Mesa is too costly --> used by very less traveller
# Similarly, 'Aduito ida' is very cheap
# 
# 'Promo' gets highest popularity among travellers followed by 'Flexible'

# In[ ]:


plt.figure(figsize = (10,8))
sns.distplot(df['price'])
plt.xlabel('Price')
plt.title('Distribution of Price')


# The distribution of Price is left - skewed. This is the general trend, i.e., cheap tickets are more popular.

# In[ ]:


def eta(z):
    start = z['start_date']
    end = z['end_date']
    td = end - start
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    total_hours = hours + minutes/60
    return total_hours
df['travelTime'] = df.apply(eta, axis = 1)


# In[ ]:


plt.figure(figsize=(10,8))
sns.distplot(df['travelTime'])
plt.xlabel('Travel Time')
plt.title('Distribution of Travel Times')


# Most of the journeys are short ones, peaking at 2.5 hrs. Frequency drops drastically at 3+ hrours

# In[ ]:


def booking(z):
    start = z['start_date']
    book = z['insert_date']
    td = start - book
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    total_days = days + hours/24 + minutes/60
    return total_days
df['bookingProximity'] = df.apply(booking, axis = 1)


# Knowledge of real world domains says the travel tickets become costlier as journey dates come close. Applying that knowledge here to check if this is a factor here.

# In[ ]:


df = df.loc[df['bookingProximity']>0,:]


# Removing updates to db after journey starts. These maybe server discrepancies, as journey ticket can *not* be booked after journey commences.

# In[ ]:


plt.figure(figsize=(10,8))
sns.lineplot(x='bookingProximity', y ='price', data = df)


# In[ ]:


plt.figure(figsize=(10,8))
sns.distplot(df['bookingProximity'])


# Most of the tickets are booked between 10 and 25 days. The trend of price is different. It increases tremendously at 1-2 days before journey starts.
# Prices vary in a small range after that, showing a slight downwards trend with increasing number of days. However, at the rightmost of the graph (near 60 days), the prices are spiking. 
# This may be attributed to greater opening prices of tickets and higher class tickets being booked early on. It however gradually lowers, due to cancellations and not buying of expensive tickets and more purchase of lower class tickets for journeys.

# Now we perform some ML operations for prediction of Ticket Price.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[ ]:


miT = min(x['travelTime'])
maT = max(x['travelTime'])
x['normTravelTime']=[(i-miT)/(maT-miT) for i in x['travelTime']]


# In[ ]:


miB = min(x['bookingProximity'])
maB = max(x['bookingProximity'])
x['normbookingProximity']=[(i-miB)/(maB-miB) for i in x['bookingProximity']]


# In[ ]:


X = x.drop(['bookingProximity', 'travelTime'], axis = 1)


# In[ ]:


X = pd.get_dummies(X)


# In[ ]:


mi = min(y)
ma = max(y)
y = [(i-mi)/(ma-mi) for i in y]


# In[ ]:


x = df[['train_type', 'train_class', 'fare', 'origin', 'destination', 'travelTime', 'bookingProximity']]
y = df['price']


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.30, random_state=4)


# In[ ]:


rf = RandomForestRegressor(n_estimators = 100)
rf.fit(x_train, y_train)


# In[ ]:


pred = rf.predict(x_test)


# In[ ]:


from sklearn import metrics
mse = metrics.mean_squared_error(y_test, pred)
print(mse)
r2 = metrics.r2_score(y_test, pred)
print(r2)

