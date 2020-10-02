#!/usr/bin/env python
# coding: utf-8

# RedWine Quality Prediction
# ![image.png](attachment:image.png)

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


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
wine = pd.read_csv("../input/winequality-red.csv")
wine.shape


# In[ ]:


wine['quality'] = np.where(wine['quality']<6.5, 'Bad', 'Good')
wine[:10]


# In[ ]:


#One hot encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#Since many ML alogrithms operate with numerical data well than categorical data,encoding the target variable to have 0 & 1 in place of Good and Bad
Encoder = LabelEncoder()
wine['quality'] = Encoder.fit_transform(wine['quality'])
wine[:10]


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


#Target Variable

#wine = wine.drop('Result',axis = 1)

X = wine.drop('quality',axis = 1)

y  = wine.quality
wine[:10]


# In[ ]:


#Train test split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
x_train.shape


# In[ ]:


#Exploring Classification algorithms
#Random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
pred_rfc = rf.predict(x_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, pred_rfc))
#Random forest gives 93% accuracy


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
pred_dt = dt.predict(x_test)
print(classification_report(y_test,pred_dt))


# In[ ]:


#Decision Tree gives 90% accuracy


# In[ ]:


from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(x_train,y_train)


# In[ ]:


logit_pred = logit.predict(x_test)
print(classification_report(y_test,logit_pred))


# In[ ]:


#logistic regression gives 89% accuracy
#Among the three models random forest gave better accuracy.


# Please upvote and fork if you find this kernel helpful.Thanks!
