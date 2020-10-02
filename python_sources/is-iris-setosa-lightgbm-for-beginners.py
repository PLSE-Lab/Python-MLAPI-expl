#!/usr/bin/env python
# coding: utf-8

# I simplified the classical Iris Species problem to a binary classification: Is Iris-setosa.
# This notebook will use LightGBM to determine whether a plant is an Iris-setosa.
# 1 means Is Iris-setosa. 
# 0 means Is Not Iris-setosa.

# In[ ]:


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from subprocess import check_output
get_ipython().run_line_magic('matplotlib', 'inline')
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


iris = pd.read_csv("../input/Iris.csv")


# In[ ]:


iris.columns


# In[ ]:


# Prepare the dataset for binary classification
iris['Is Iris-setosa'] = 0
iris.loc[iris.Species=='Iris-setosa', 'Is Iris-setosa'] = 1


# In[ ]:


print(iris[0:1])
print(iris[50:51])
print(iris[100:101])


# In[ ]:


X = iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
y = iris["Is Iris-setosa"].values


# In[ ]:


print(X[0:1])
print(X[50:51])
print(X[100:101])


# In[ ]:


print(y[0:1])
print(y[50:51])
print(y[100:101])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[ ]:


# lgb params
params = {'objective':'binary', 'metric':'auc'}
  
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

lgb_model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_test], verbose_eval=10)
predictions = lgb_model.predict(X_test)


# In[ ]:


print(predictions)


# In[ ]:


predictions = np.around(predictions)
print(predictions)

