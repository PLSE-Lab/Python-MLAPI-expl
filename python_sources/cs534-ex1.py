#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import os
print(os.listdir("../input"))


# In[ ]:


train =  pd.read_csv('../input/train.csv', nrows = 100_000)
test =  pd.read_csv('../input/test.csv')
test.shape


# ## Overview

# In[ ]:


train.head()


# In[ ]:


print(train.shape)


# In[ ]:


train.dtypes


# In[ ]:


train.describe()


# In[ ]:


print(train.isnull().sum())


# ## Data Processing

# In[ ]:


# drop nan
train = train.dropna()
test = test.dropna()
print(train.shape)


# In[ ]:


# drop zero passenger
train = train[train.passenger_count!=0]
test = test[test.passenger_count!=0]
print(train.shape)


# In[ ]:


# drop negative fares
train = train[train.fare_amount>=0]
print(train.shape)


# ## Feature

# In[ ]:


# calculate distance
def cal_dis(df):
    delta_longitude = np.fabs((df.dropoff_longitude - df.pickup_longitude))
    delta_latitude = np.fabs((df.dropoff_latitude - df.pickup_latitude))
    dis = np.sqrt(delta_longitude**2 + delta_latitude**2)
    
    return dis
      
train['dis'] = train.apply(cal_dis, axis='columns')
test['dis'] = test.apply(cal_dis, axis='columns')
train.head()


# In[ ]:


# extract 'year' and 'hour' 
def get_Year(df):
    y = pd.to_datetime(df.pickup_datetime).year

    return y

def get_Hour(df):
    h = pd.to_datetime(df.pickup_datetime).hour

    return h

train['year'] = train.apply(get_Year, axis='columns')
train['hour'] = train.apply(get_Hour, axis='columns')
test['year'] = test.apply(get_Year, axis='columns')
test['hour'] = test.apply(get_Hour, axis='columns')
train.head()


# ## Visualization

# In[ ]:


sns.heatmap(train.corr(),annot = True)


# In[ ]:


plt = sns.boxplot(
    x='passenger_count',
    y='fare_amount',
    data=train
)
plt.axes.set_ylim([0, 25])


# In[ ]:


train.plot(x='dis',y='fare_amount',kind='scatter')
train.dis.plot.hist()


# In[ ]:


# divide the distance region
train = train[train.dis<0.3]
train.dis.plot.hist()


# In[ ]:


# drop distance outliers
train = train[train.dis<=0.3]
test = test[test.dis<=0.3]
print(train.shape)
train.plot(x='dis',y='fare_amount',kind='scatter')


# In[ ]:


# drop distance outliers 
train = train[train.dis>=0.001]
test = test[test.dis>=0.001]
print(train.shape)
train.plot(x='dis',y='fare_amount',kind='scatter')


# In[ ]:


plt = sns.boxplot(
    x='year',
    y='fare_amount',
    data=train
)

plt.axes.set_ylim([0, 30])


# In[ ]:


plt = sns.boxplot(
    x='hour',
    y='fare_amount',
    data=train
)

plt.axes.set_ylim([0, 25])


# ## Linear Regression

# In[ ]:


print(train.dtypes)
train = train.drop(columns=['key','pickup_datetime'])
train.dtypes
train_y = train['fare_amount']
train_x = train.drop(columns=['fare_amount'])


# In[ ]:


SaveX = train_x


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score


# In[ ]:


# standarization
col_names = train_x.columns

scaler = StandardScaler()
scaler.fit(train_x) 
train_x = scaler.transform(train_x)
train_x = pd.DataFrame(train_x,columns=col_names)

b0 = train_y.mean(axis=0)
train_y -= b0


# In[ ]:





# In[ ]:




print("Regular linear regression:")
lm = linear_model.LinearRegression(fit_intercept=False)
#print("R2: %8.6f" 
#      % (np.mean(cross_val_score(lm, train_x[['dis','year','hour', 'passenger_count']], train_y, cv=10, scoring='r2')))) 
lm.fit(train_x[['dis','year','hour', 'passenger_count']], train_y)
print(train_x[['dis','year','hour', 'passenger_count']].columns)
print(lm.coef_)


# In[ ]:


print("Ridge regression:")
alpha_list = np.arange(0,1000,10)
r2_list = []
for a in alpha_list:
    ridge = linear_model.Ridge(fit_intercept=False, alpha=a)
    r2_list.append(np.mean(cross_val_score(ridge, train_x, train_y, cv=10, scoring='r2')))

Error = pd.DataFrame(np.stack((alpha_list,r2_list)),index = ['alpha','R2']).T
Error.plot(x='alpha',y='R2',kind='line')


# In[ ]:


print("Lasso regression:")
alpha_list = np.arange(0.1,10,0.1)
r2_list = []
for a in alpha_list:
    lasso = linear_model.Lasso(fit_intercept=False, alpha=a)
    r2_list.append(np.mean(cross_val_score(lasso, train_x, train_y, cv=10, scoring='r2')))

Error = pd.DataFrame(np.stack((alpha_list,r2_list)),index = ['alpha','R2']).T
Error.plot(x='alpha',y='R2',kind='line')


# In[ ]:


lm = linear_model.LinearRegression(fit_intercept=False)
selector = RFECV(lm, step=1, cv=10)
selector = selector.fit(train_x, train_y)
print(selector.support_)
print(selector.ranking_)


# In[ ]:


print("Regular linear regression:")
lm = linear_model.LinearRegression(fit_intercept=False)
print("R2: %8.6f" 
      % (np.mean(cross_val_score(lm, pd.DataFrame(train_x['dis']), train_y, cv=10, scoring='r2'))))
print("R2: %8.6f" 
      % (np.mean(cross_val_score(lm, pd.DataFrame(train_x[['dis','year']]), train_y, cv=10, scoring='r2'))))
print("R2: %8.6f" 
      % (np.mean(cross_val_score(lm, pd.DataFrame(train_x[['dis','year','hour']]), train_y, cv=10, scoring='r2'))))
print("R2: %8.6f" 
      % (np.mean(cross_val_score(lm, pd.DataFrame(train_x[['dis','year','hour', 'passenger_count']]), 
                                 train_y, cv=10, scoring='r2'))))
print("R2: %8.6f" 
      % (np.mean(cross_val_score(lm, pd.DataFrame(train_x[['dis','year','hour', 'passenger_count']]), 
                                 train_y, cv=10, scoring='r2'))))
print("R2: %8.6f" 
      % (np.mean(cross_val_score(lm, pd.DataFrame(train_x[['dis','year','hour', 'passenger_count',
                                                          'pickup_longitude', 'pickup_latitude']]), 
                                 train_y, cv=10, scoring='r2'))))
print("R2: %8.6f" 
      % (np.mean(cross_val_score(lm, pd.DataFrame(train_x[['dis','year','hour', 'passenger_count',
                                                          'pickup_longitude', 'pickup_latitude',
                                                          'dropoff_longitude', 'dropoff_latitude']]), 
                                 train_y, cv=10, scoring='r2'))))


# In[ ]:


lm.coef_


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

depth = np.arange(2,10,1)
for i in depth:
    reg = DecisionTreeRegressor(max_depth=i)
    print("max_depth:%d R2: %8.6f" % (i, np.mean(cross_val_score(reg, train_x, train_y, cv=10, scoring='r2'))))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

depth = np.arange(2,10,1)
for i in depth:
    rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=i, n_estimators=100)
    print("max_depth:%d R2: %8.6f" % (i, np.mean(cross_val_score(rf, train_x, train_y, cv=10, scoring='r2'))))


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
      max_depth=1, random_state=0, loss='ls')

print("R2: %8.6f" % (np.mean(cross_val_score(gbr, train_x, train_y, cv=10, scoring='r2'))))


# In[ ]:





# In[ ]:





# In[ ]:





# ## Reference

# https://www.kaggle.com/dster/nyc-taxi-fare-starter-kernel-simple-linear-model
# 
# https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration

# In[ ]:





# In[ ]:





# In[ ]:




