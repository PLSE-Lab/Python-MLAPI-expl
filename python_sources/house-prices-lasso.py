#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# ** Data loading **

# In[2]:


# Load train data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test_ids = test['Id']


# ** Data cleansing **

# In[3]:


# Check train and test shape
print("Train data shape:", train.shape)
print("Test data shape:", test.shape)


# In[4]:


# Check train data
train.head()


# In[5]:


# Check fields summary
train.describe()


# In[6]:


# Removing outliers
train = train[train['GrLivArea']<4000]


# In[7]:


# Select target values
y = train['SalePrice']
y.head()


# In[8]:


y = np.log(y)
y.head()


# In[9]:


# Drop ID column
train_ids = train['Id']
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)


# In[10]:


# Find least correlated features relative to target
corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
corr.iloc[:,-1:]


# In[11]:


# Remove least correlated features
train.drop(corr.iloc[-9:-1,-1:].index, axis=1, inplace=True)
test.drop(corr.iloc[-9:-1,-1:].index, axis=1, inplace=True)


# In[12]:


# Get categorical features
categorical_features = train.select_dtypes(include = ['object']).columns
categorical_features


# In[13]:


# Get numerical features
numerical_features = train.select_dtypes(exclude = ["object"]).columns
numerical_features


# In[14]:


# Differentiate numerical features (minus the target) and categorical features
numerical_features = numerical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
train_num = train[numerical_features]
train_cat = train[categorical_features]
test_num = test[numerical_features]
test_cat = test[categorical_features]


# In[15]:


# Fill null values with median for numerical features
print("NAs for numerical features in train : " + str(train_num.isnull().values.sum()))
train_num = train_num.fillna(train_num.median())
test_num = test_num.fillna(test_num.median())
print("Remaining NAs for numerical features in train : " + str(train_num.isnull().values.sum()))


# In[16]:


# One hot encoding for categorical features
len_train = train_cat.shape[0]
houses_cat = pd.concat([train_cat, test_cat], sort=False)
print("Train: ", houses_cat.isnull().values.sum(), "/ Test:", houses_cat.isnull().values.sum())
houses_cat = pd.get_dummies(houses_cat)
print("After get_dummies - Train: ", houses_cat.isnull().values.sum(), "/ Test:", houses_cat.isnull().values.sum())
train_cat = houses_cat[:len_train]
test_cat = houses_cat[len_train:]


# In[17]:


(train_cat.shape, train_num.shape, test_cat.shape, test_num.shape)


# In[18]:


# Merge categorical and numerical features
train = pd.concat([train_cat, train_num], axis=1)
test = pd.concat([test_cat, test_num], axis=1)
#train = train_cat  # <- use to ignore categorical fields
#test = test_cat    # <- use to ignore categorical fields
(train.shape, test.shape)


# ** Training **

# In[19]:


from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
from sklearn.externals import joblib


# In[20]:


sc = RobustScaler()
x = sc.fit_transform(train)


# In[21]:


model = Lasso(alpha = 0.001, random_state=1)


# In[22]:


model.fit(x, y)


# In[23]:


joblib.dump(model, 'house_prices_lasso.joblib')


# [download](./house_prices_lasso.joblib)

# ** Evaluation **

# In[24]:


# Execute prediction with train data
train_prediction = model.predict(sc.transform(train))


# In[25]:


true_x_pred = pd.DataFrame({'SalePrice': np.exp(y), 'Predicted': np.exp(train_prediction)})
true_x_pred.head()


# In[26]:


# Calculates root mean squared log error
from sklearn.metrics import mean_squared_log_error
from math import sqrt
rmsle = sqrt(mean_squared_log_error(true_x_pred['SalePrice'], true_x_pred['Predicted']))
print("RMSLE:", rmsle)


# In[27]:


import matplotlib.pyplot as plt
plt.scatter(true_x_pred['Predicted'], true_x_pred['SalePrice'], c = "blue",  label = "Training data")
plt.title("Lasso")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10, 660000], [10, 660000], c = "red")
plt.show()


# ** Predict test data **

# In[28]:


test_transf = sc.transform(test)
test_prediction = np.exp(model.predict(test_transf))
output = pd.DataFrame({'Id': test_ids, 'SalePrice': test_prediction})
output.head()


# In[29]:


output.to_csv('submission.csv', index=False)


# [download](./submission.csv)

# In[ ]:




