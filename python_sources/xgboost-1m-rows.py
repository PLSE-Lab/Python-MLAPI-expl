#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

#Getting train and test data
train_set = pd.read_csv('../input/train.csv', nrows = 10 ** 6)
test_set = pd.read_csv('../input/test.csv')


# In[ ]:


train_set.head()


# In[ ]:


#Removing the redundant columns
train_set = train_set.drop(train_set.columns[[0, 2]], axis = 1)
test_set = test_set.drop(test_set.columns[[0, 1]], axis = 1)


# In[ ]:


#Removing the nan's and zeros
train_set = train_set.dropna(how = 'any', axis = 0)
train_set = train_set[~(train_set == 0).any(axis = 1)]


# In[ ]:


#Seperating the independent and dependent variables
train_set_y = train_set.iloc[:, 0:1].values
train_set_x = train_set.iloc[:, 1:6].values


# In[ ]:


#Lets make a train set and a development set
from sklearn.model_selection import train_test_split
X_train, X_dev, y_train, y_dev = train_test_split(train_set_x, train_set_y, test_size = 0.1)


# In[ ]:


#Fitting XGBoost to training set
from xgboost import XGBRegressor
regressor = XGBRegressor(max_depth = 10, learning_rate=0.1, n_estimators=200)
regressor.fit(X_train, y_train)


# In[ ]:


#Dev set predictions
predictions = regressor.predict(X_dev)


# In[ ]:


#Evaluation on dev set - root mean squared error
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(y_dev, predictions))
print(rmse)


# In[ ]:


#Test set predictions
test_set = np.array(test_set)
test_set_pred = regressor.predict(test_set)


# In[ ]:


#Putting everything together in a .csv file
keys = pd.read_csv('../input/test.csv')
keys = keys.iloc[:,0:1]

test_predictions = pd.DataFrame(test_set_pred, columns = ['fare_amount'])

final = pd.concat([keys, test_predictions], axis = 1)

final.to_csv('sub2.csv', encoding='utf-8', index = False)


# In[ ]:




