#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# linear algebra
import numpy as np
 
# data processing
import pandas as pd
 
# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
 
# Algorithms
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


train = pd.read_csv("../input/zaloni-techniche-datathon-2019/train.csv")
test = pd.read_csv("../input/zaloni-techniche-datathon-2019/test.csv")


# In[ ]:


train.info()


# In[ ]:


common_value = 'S'
data = [train,test]

for dataset in data:
    dataset['last_name'] = dataset['last_name'].fillna(common_value)
    dataset['first_name'] = dataset['first_name'].fillna(common_value)


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


data = [train,test]
for dataset in data:
    if dataset['first_name'].dtype == type(object):
        le = LabelEncoder()
        dataset['first_name'] = le.fit_transform(dataset['first_name'])


# In[ ]:


data = [train,test]
for dataset in data:
    if dataset['last_name'].dtype == type(object):
        le = LabelEncoder()
        dataset['last_name'] = le.fit_transform(dataset['last_name'])


# In[ ]:


data = [train,test]

for dataset in data:
    dataset['first_name'] = dataset['first_name'].astype(float)
    dataset['last_name'] = dataset['last_name'].astype(float)


# In[ ]:


print("Data types and their frequency\n{}".format(train.dtypes.value_counts()))
print("Data types and their frequency\n{}".format(test.dtypes.value_counts()))


# In[ ]:


cols = ['gender','race']
for name in cols:
    print(name,':')
    print(train[name].value_counts(),'\n')


# In[ ]:


cols = ['first_name','last_name']
for name in cols:
    print(name,':')
    print(train[name].value_counts(),'\n')


# In[ ]:


print(train.shape)


# In[ ]:


train.head(9)


# In[ ]:


cols = ['race']
for name in cols:
    print(name,':')
    print(train[name].value_counts(),'\n')


# In[ ]:


train_Y = train.gender
train_predictor_columns = ['last_name', 'first_name']
train_X = train[train_predictor_columns]
test_X = test[train_predictor_columns]


# In[ ]:


train_Y1 = train.race
train_predictor_columns = ['last_name', 'first_name']
train_X = train[train_predictor_columns]
test_X = test[train_predictor_columns]


# In[ ]:


logreg = LogisticRegression()
logreg.fit(train_X,train_Y)

Y_pred_gender = logreg.predict(test_X)

acc_log = round(logreg.score(train_X, train_Y) * 100, 2)
print (acc_log)


# In[ ]:


logreg = LogisticRegression()
logreg.fit(train_X,train_Y1)

Y_pred_race = logreg.predict(test_X)

acc_log = round(logreg.score(train_X, train_Y1) * 100, 2)
print (acc_log)


# In[ ]:




