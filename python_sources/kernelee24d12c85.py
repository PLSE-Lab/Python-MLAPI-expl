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


df = pd.read_csv("../input/train.csv")


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


y= df["target"]


# In[ ]:


y.head()


# In[ ]:


df.columns


# In[ ]:


X= df.drop(columns = ["target","ID_code"])


# In[ ]:


X.head()


# In[ ]:


# Splitting the data 
from sklearn.model_selection import train_test_split

# Training data , validation data, training target variable, test target variable 
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[ ]:


# DECISION TREE CLASSIFIER MODEL  for predicting the duration of the claim

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# Define model 
worker_model = XGBClassifier()

# Fit model
worker_model.fit(train_X, train_y)

# get predicted prices on validation data
predictions = worker_model.predict(val_X)

# Checking the R square value 
print("R square value : ",r2_score(val_y, predictions))

# Checking the mean_absolute_percentage_error 
#print("mean absolute % error : ",mean_absolute_percentage_error(val_y, predictions))

# Checking the mean_absolute_percentage_error 
print("Accuracy score : ",accuracy_score(val_y, predictions))


# In[ ]:


train =  pd.read_csv("../input/test.csv")


# In[ ]:


train_index = train["ID_code"]


# In[ ]:


train = train.drop(columns=["ID_code"])


# In[ ]:


predictions = worker_model.predict(train)


# In[ ]:


pd.DataFrame(predictions)


# In[ ]:


submission=train_index 


# In[ ]:


submission=pd.DataFrame(submission)


# In[ ]:


submission['target'] = predictions


# In[ ]:


submission


# In[ ]:


submission.to_csv("Submission.csv", index=False)

