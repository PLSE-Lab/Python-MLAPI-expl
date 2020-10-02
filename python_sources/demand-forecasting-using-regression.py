#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import pandas as pd
data = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.info()


# Agent, Company, country  have some missing data, 

# In[ ]:


print("Nan in each columns" , data.isna().sum(), sep='\n')


# Country Has 488 missing Values, Agent has 16340 missing value & company has 112593 missing value

# In[ ]:


data = data.drop(['company'], axis = 1)
data = data.dropna(axis = 0)


# As company has maximum missing data lets drop that column
# Aslo drop all the rows that have NaN in them as per above code

# Now we have 31 columns with equal data i.e. 102894

# Lets now check the unique values in each column

# In[ ]:


data.nunique()


# In[ ]:


data['hotel'] = data['hotel'].map({'Resort Hotel':0, 'City Hotel':1})
data['hotel'].unique()


# with the above code line we have converted object values to integer values of 0 & 1
# With below codes we will convert all the object type data into integer values which machine can read

# In[ ]:


data['arrival_date_month'] = data['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,
                                                            'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
data['arrival_date_month'].unique()


# In[ ]:


# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column. 
data['customer_type']= label_encoder.fit_transform(data['customer_type']) 
data['assigned_room_type'] = label_encoder.fit_transform(data['assigned_room_type'])
data['deposit_type'] = label_encoder.fit_transform(data['deposit_type'])
data['reservation_status'] = label_encoder.fit_transform(data['reservation_status'])
data['meal'] = label_encoder.fit_transform(data['meal'])
data['country'] = label_encoder.fit_transform(data['country'])
data['distribution_channel'] = label_encoder.fit_transform(data['distribution_channel'])
data['market_segment'] = label_encoder.fit_transform(data['market_segment'])
data['reserved_room_type'] = label_encoder.fit_transform(data['reserved_room_type'])
data['reservation_status_date'] = label_encoder.fit_transform(data['reservation_status_date'])
  


# We have converted strings and object data into machine readable format

# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import AdaBoostRegressor


# Let's now use Regression modles to check the best one.

# In[ ]:


X = data.drop(['previous_cancellations'], axis = 1)
y = data['previous_cancellations']


# Our Target is y with previous_cancellations, & X contains all the data except previous_cancellation
# with below codes we will train_test_split the data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
y_pred = regressor.predict(X_test)

print('Mean Absolute Error_lng:', metrics.mean_absolute_error(y_test, y_pred).round(3))  
print('Mean Squared Error_lng:', metrics.mean_squared_error(y_test, y_pred).round(3))  
print('Root Mean Squared Error_lng:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
print('r2_score_lng:', r2_score(y_test, y_pred).round(3))

## Linear Regression above##


# In[ ]:


ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train) #training the algorithm

y_pred = ridge.predict(X_test)

print('Mean Absolute Error_ridge:', metrics.mean_absolute_error(y_test, y_pred).round(3))  
print('Mean Squared Error_ridge:', metrics.mean_squared_error(y_test, y_pred).round(3))  
print('Root Mean Squared Error_ridge:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
print('r2_score_ridge:', r2_score(y_test, y_pred).round(3))

## Ridge Regression above##


# In[ ]:


clf = Lasso(alpha=0.1)

clf.fit(X_train, y_train) #training the algorithm

y_pred = clf.predict(X_test)

print('Mean Absolute Error_lasso:', metrics.mean_absolute_error(y_test, y_pred).round(3))  
print('Mean Squared Error_lasso:', metrics.mean_squared_error(y_test, y_pred).round(3))  
print('Root Mean Squared Error_lasso:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
print('r2_score_lasso:', r2_score(y_test, y_pred).round(3))

## Lasso Regression above##


# In[ ]:


logreg = LogisticRegression(solver = 'lbfgs')
# fit the model with data
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

print('Mean Absolute Error_logreg:', metrics.mean_absolute_error(y_test, y_pred).round(3))  
print('Mean Squared Error_logreg:', metrics.mean_squared_error(y_test, y_pred).round(3))  
print('Root Mean Squared Error_logreg:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
print('r2_score_logreg:', r2_score(y_test, y_pred).round(3))

## Logistics Regression above ##


# In[ ]:


# Ridge Regression with Gridsearch ##
from sklearn.model_selection import GridSearchCV

parameters= {'alpha':[50,75,100,200, 230, 250], 'random_state':[5,10,20,50,], 'max_iter':[0.1,0.5,1,2,3,5]}

grid = GridSearchCV(ridge, parameters, cv=5)
grid.fit(X_train, y_train)
print ("Best_Score_Ridge : ", grid.best_score_)
print('best_para_Ridge:', grid.best_params_)


# In[ ]:


# Lasso Regression with Gridsearch ##
from sklearn.model_selection import GridSearchCV

parameters= {'alpha':[200, 230, 250,265, 270, 275, 290, 300], 'random_state':[2,5,10,20,50,], 'max_iter':[5,10,15,20,30,50,100]}

grid = GridSearchCV(clf, parameters, cv=5)
grid.fit(X_train, y_train)
print ("Best_Score_Lasso : ", grid.best_score_)
print('best_para_Lasso:', grid.best_params_)


# In[ ]:


# create regressor object 
rfe = RandomForestRegressor(n_estimators = 100, random_state = 42) 
 
# fit the regressor with x and y data 
rfe.fit(X, y)   
y_pred=rfe.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r2_score_RFE:', r2_score(y_test, y_pred).round(3))


# In[ ]:


ABR = AdaBoostRegressor(n_estimators = 100, random_state = 42) 
  
# fit the regressor with x and y data 
ABR.fit(X, y)   
y_pred=ABR.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r2_score_ABR:', r2_score(y_test, y_pred).round(3))

