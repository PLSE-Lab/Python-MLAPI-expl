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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import cufflinks as cf
import tensorflow as tf
import pickle
#from fastai.structured import *
#from fastai.column_data import *
from IPython.display import HTML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
cf.go_offline()
init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


Path = "../input/"
tables = ['train','store']


# In[ ]:


table = [pd.read_csv(f'{Path}{fname}.csv',low_memory=False) for fname in tables]


# In[ ]:


train, store = table


# # Data

# ## Data Info

# Data source: https://www.kaggle.com/c/rossmann-store-sales/data
# 
# Most of the fields are self-explanatory. The following are descriptions for those that aren't.
# 
# Id - an Id that represents a (Store, Date) duple within the test set
# 
# Store - a unique Id for each store
# 
# Sales - the turnover for any given day (this is what you are predicting)
# 
# Customers - the number of customers on a given day
# 
# Open - an indicator for whether the store was open: 0 = closed, 1 = open
# 
# StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
# 
# SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
# 
# StoreType - differentiates between 4 different store models: a, b, c, d
# 
# Assortment - describes an assortment level: a = basic, b = extra, c = extended
# 
# CompetitionDistance - distance in meters to the nearest competitor store
# 
# CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
# 
# Promo - indicates whether a store is running a promo on that day
# 
# Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
# 
# Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
# 
# PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

# In[ ]:


for t in table:
    display(t.head())


# In[ ]:


# for t in table:
#     display(DataFrameSummary(t).summary())


# In[ ]:


print(len(train))
len(store)


# # Exploratory Data Analysis

# In[ ]:


# Open
fig, (axis1) = plt.subplots(1,1,figsize=(15,8))
sns.countplot(x='Open',hue='DayOfWeek', data=train,palette="husl", ax=axis1)


# Drop Open column
# train.drop("Open", axis=1, inplace=True)


# In[ ]:


train_store = pd.merge(train, store, how = 'inner', on = 'Store')

train_store['Date'] = pd.to_datetime(train_store['Date']).apply(lambda x: x.date())
train_store['Year'] = pd.to_datetime(train_store['Date']).apply(lambda x: x.year)
train_store['Month'] = pd.to_datetime(train_store['Date']).apply(lambda x: x.month)
train_store['Day'] = pd.to_datetime(train_store['Date']).apply(lambda x: x.day)


# In[ ]:


c = '#386B7F' #Basic RGB
sns.factorplot(data = train_store, x = 'Month', y = "Sales", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo', # per promo in the store in rows
              ) 


# In[ ]:


# sales trends
sns.catplot(data = train_store, x = 'Month', y = "Customers", 
               col = 'StoreType', # per store type in cols
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo', # per promo in the store in rows
               color = c,
           kind = 'boxen') 


# In[ ]:


# Compute the correlation matrix 
# exclude 'Open' variable
corr_all = train_store.drop('Open', axis = 1).corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_all, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_all, mask = mask,
            square = True, linewidths = .5, ax = ax, cmap = "BuPu")      
plt.show()


# In[ ]:


# sale per customer trends
sns.factorplot(data = train_store, x = 'DayOfWeek', y = "Sales", 
               col = 'Promo', 
               row = 'Promo2',
               hue = 'Promo2',
               palette = 'RdPu') 


# In[ ]:



train_store.head(2)


# In[ ]:


train_store[:10000].iplot(kind='scatter',y='Sales',x='Customers',mode='markers',size=10,
                          xTitle='Number of Customers', yTitle='Sales')


# In[ ]:


# train_store[:10000].iplot(kind='scatter3d',y='Sales',x='Customers',
#                           z='CompetitionDistance',mode='markers',size=10 )


# In[ ]:


# SchoolHoliday

# Plot
sns.countplot(x='SchoolHoliday', data=train)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='SchoolHoliday', y='Sales', data=train, ax=axis1)
sns.barplot(x='SchoolHoliday', y='Customers', data=train, ax=axis2)


# In[ ]:


train_store[:1050].dropna().iplot(kind='spread',x='Date',y=['Customers','Sales']
                                  ,xTitle='Date',yTitle='Sales')


# In[ ]:


train.iplot()


# In[ ]:


train_store[:1050].dropna().iplot(kind='spread',x='Date',y='Sales',
                                  xTitle='Date',yTitle='Sales',dash='dashdot',theme='white')


# In[ ]:


#train_store.iplot(kind='histogram',x)


# # Models

# In[ ]:


merged = train_store


# In[ ]:


#Working with missing data

imp = Imputer(missing_values='NaN',strategy='mean')
imp.fit(merged['CompetitionDistance'].values.reshape(-1,1))
merged['CompetitionDistance'] = imp.transform(merged['CompetitionDistance'].values.reshape(-1,1))

imp1 = Imputer(strategy='median')
imp1.fit(merged['CompetitionOpenSinceYear'].values.reshape(-1,1))
merged['CompetitionOpenSinceYear'] = imp.transform(merged['CompetitionOpenSinceYear'].values.reshape(-1,1))

imp1.fit(merged['CompetitionOpenSinceMonth'].values.reshape(-1,1))
merged['CompetitionOpenSinceMonth'] = imp.transform(merged['CompetitionOpenSinceMonth'].values.reshape(-1,1))


# In[ ]:


#Dropping columns with excessive null values
merged = merged.drop(['Promo2SinceWeek','Promo2SinceYear','PromoInterval'],axis=1)


# In[ ]:


merged = pd.concat([merged,pd.get_dummies(merged['StateHoliday'],prefix='StateHoliday',drop_first=True)],axis=1)
merged = pd.concat([merged,pd.get_dummies(merged['StoreType'],prefix='StoreType',drop_first=True)],axis=1)
merged = pd.concat([merged,pd.get_dummies(merged['Assortment'],prefix='Assortment',drop_first=True)],axis=1)
merged.drop(['StateHoliday','StoreType','Assortment','Date'],axis=1,inplace=True)


# In[ ]:


merged.head()


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
lr = LinearRegression()
rf = RandomForestRegressor()


# In[ ]:


merged.columns


# In[ ]:


X = ['Store', 'DayOfWeek', 'Open', 'Promo',
       'SchoolHoliday', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
       'CompetitionOpenSinceYear', 'Promo2', 'Year', 'Month', 'Day',
       'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c', 'StoreType_b',
       'StoreType_c', 'StoreType_d', 'Assortment_b', 'Assortment_c']
y = ['Sales']


# In[ ]:


x_train, x_eval, y_train, y_eval = train_test_split(merged[X], merged[y],
                                                    test_size=0.3, random_state=101)


# In[ ]:


print(x_train.shape)
print(y_train.shape)

print(x_eval.shape)
print(y_eval.shape)


# In[ ]:


lr.fit(x_train,y_train)


# In[ ]:


prediction = lr.predict(x_eval)


# In[ ]:


from sklearn.metrics import r2_score, mean_squared_error,explained_variance_score
print(r2_score(y_eval, prediction))
print(mean_squared_error(y_eval, prediction))
print(explained_variance_score(y_eval, prediction))


# In[ ]:


rf = RandomForestRegressor()
rf.fit(x_train,y_train)


# In[ ]:


rf_pred = rf.predict(x_eval)


# In[ ]:


print(r2_score(y_eval, rf_pred))
print(mean_squared_error(y_eval, rf_pred))
print(explained_variance_score(y_eval, rf_pred))


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=10)


# In[ ]:


knn.fit(x_train,y_train)


# In[ ]:


knn_pred = knn.predict(x_eval)


# In[ ]:


print(r2_score(y_eval, knn_pred))
print(mean_squared_error(y_eval, knn_pred))
print(explained_variance_score(y_eval, knn_pred))


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()


# In[ ]:


dt.fit(x_train,y_train)


# In[ ]:


dt_pred = dt.predict(x_eval)


# In[ ]:


print(r2_score(y_eval, dt_pred))
print(mean_squared_error(y_eval, dt_pred))
print(explained_variance_score(y_eval, dt_pred))


# In[ ]:


# Will take so much time
# from sklearn.neural_network import MLPRegressor
# mlp = MLPRegressor()


# In[ ]:


#mlp.fit(x_train,y_train)


# In[ ]:


# mlp_pred = mlp.predict(x_eval)


# In[ ]:


# print(r2_score(y_eval, mlp_pred))
# print(mean_squared_error(y_eval, mlp_pred))
# print(explained_variance_score(y_eval, mlp_pred))


# In[ ]:




