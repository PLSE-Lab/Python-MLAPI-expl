#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_log_error
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/bike-sharing-demand/train.csv')
test = pd.read_csv('../input/bike-sharing-demand/test.csv')
train.head()


# Now we will convert the current datetime column into Machine Learning friendly format i.e Year, Month, Day, Hour.

# In[ ]:


train['year'] = [t.year for t in pd.DatetimeIndex(train.datetime)]
train['month'] = [t.month for t in pd.DatetimeIndex(train.datetime)]
train['day'] = [t.day for t in pd.DatetimeIndex(train.datetime)]
train['hour'] = [t.hour for t in pd.DatetimeIndex(train.datetime)]

test['year'] = [t.year for t in pd.DatetimeIndex(test.datetime)]
test['month'] = [t.month for t in pd.DatetimeIndex(test.datetime)]
test['day'] = [t.day for t in pd.DatetimeIndex(test.datetime)]
test['hour'] = [t.hour for t in pd.DatetimeIndex(test.datetime)]


# In[ ]:


train.head()


# In above output we can see we converted the datetime column in Machine Learning firendly format now we will drop the datetime column

# In[ ]:


train.drop('datetime',axis=1,inplace=True)
test.drop('datetime',axis=1,inplace=True)


# In[ ]:


train.head()


# **Data Visualisation**

# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
fig, ax = plt.subplots(2,2)
sns.barplot(train['season'],train['count'],ax=ax[0,0]);
sns.barplot(train['holiday'],train['count'],ax=ax[0,1]);
sns.barplot(train['workingday'],train['count'],ax=ax[1,0]);
sns.barplot(train['weather'],train['count'],ax=ax[1,1]);


# The above plots show us intuitively how count parameter differs with workingday, weather, season, holiday

# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
fig, ax = plt.subplots(2,2)
sns.distplot(train['temp'],ax=ax[0,0]);
sns.distplot(train['atemp'],ax=ax[0,1]);
sns.distplot(train['humidity'],ax=ax[1,0]);
sns.distplot(train['windspeed'],ax=ax[1,1]);


# This are distribution plot of humidity, windspeed, temp and atemp

# In[ ]:


sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(train.corr(),annot=True,linewidths=0.5);


# We cans see that registered/casual is highly correlated with the count which means most of the bike were registered.

# In[ ]:


train.drop(['casual','registered'],axis=1,inplace=True)


# **Correlation Plot**

# In[ ]:


sns.set(rc={'figure.figsize':(20,5)})
sns.barplot(x=train['month'],y=data['count']);


# The above plot explains the demand of the bicycle according to month.

# **Data Transformation**

# In[ ]:


season = pd.get_dummies(train['season'],prefix='season')
train = pd.concat([train,season],axis=1)


# In[ ]:


train.drop('season',axis=1,inplace=True)
train.head()


# In[ ]:


weather = pd.get_dummies(train['weather'],prefix='weather')

train = pd.concat([train,weather],axis=1)

train.drop('weather',axis=1,inplace=True)
train.head()


# **Splitting data into Train and Test split**

# In[ ]:


train.columns.to_series().groupby(data.dtypes).groups


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[ ]:


X = train.drop('count',axis=1)
y = train['count']
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# **Model Building**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
model_rf = rf.fit(X_train,y_train)
y_pred_rf = model_rf.predict(X_test)
np.sqrt(mean_squared_log_error(y_test,y_pred_rf))


# The above Random Forest model is using default parameter of the Random forest model to reduse RMSLE more we will do hyperparameter tuning the parameters considered are listed below

# In[ ]:


n_estimators = [int(x) for x in range(200,2000,100)]
max_feature = ['auto','sqrt']
min_sample_split = [2,5,10]
min_sample_leaf = [1,2,4]
max_depth = [int(x) for x in range(10,110,11)]
max_depth.append(None)


# In[ ]:


random_grid = {'n_estimators': n_estimators,
              'max_depth': max_depth,
              'max_features': max_feature,
              'min_samples_leaf': min_sample_leaf,
              'min_samples_split': min_sample_split}


# In[ ]:


random_grid


# Now we will find the optimal solution using RandomizedSearchCV, the estimator will be Random Forest and parameters will be all the parameters in random_grid

# In[ ]:


rf_tune = RandomForestRegressor()
from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator=rf_tune,param_distributions=random_grid,n_iter=100,cv=5,verbose= 2,n_jobs=-1)


# by running **rf_random(X_train,y_train)** we will get the optimal parameter as below
# 
# * max_depth=87
# * max_features='auto'
# * min_samples_leaf=1
# * min_samples_split=2
# * n_estimators=1300
# 
# now we will make final Random forest model with hyperparameters

# In[ ]:


final_rf = RandomForestRegressor(max_depth=87,max_features='auto',min_samples_leaf=1,min_samples_split=2,n_estimators=1300)
final_model_rf = final_rf.fit(X_train,y_train)
y_final_pred = final_model_rf.predict(X_test)
np.sqrt(mean_squared_log_error(y_test,y_final_pred))

