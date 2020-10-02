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


df = pd.read_csv("../input/HR-Employee-Attrition.csv")


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.describe().T


# In[ ]:


# import pandas_profiling
# pandas_profiling.ProfileReport(df)


# In[ ]:


df.columns


# In[ ]:


df.Over18.value_counts()


# In[ ]:


df.EmployeeCount.value_counts()


# In[ ]:


df.StandardHours.value_counts()


# In[ ]:


df.drop(columns=['Over18', 'EmployeeCount','StandardHours'], inplace=True)


# In[ ]:


df.columns


# In[ ]:


df[['MonthlyIncome','JobLevel']].corr()


# In[ ]:


df.isna().sum()


# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


df.shape


# In[ ]:


df['Attrition'].replace('Yes',1,inplace=True)


# In[ ]:


df['Attrition'].replace('No',0,inplace=True)


# In[ ]:


df['Attrition']


# In[ ]:


num_cols = df.select_dtypes(include = np.number)


# In[ ]:


cat_col = df.select_dtypes(exclude=np.number)


# In[ ]:


encoded_cat_cols = pd.get_dummies(cat_col)


# In[ ]:


encoded_cat_cols


# In[ ]:


preprocessed_df = pd.concat([encoded_cat_cols, num_cols], axis=1)


# In[ ]:


preprocessed_df.head()


# In[ ]:


x = preprocessed_df.drop(columns='Attrition')


# In[ ]:


y = preprocessed_df['Attrition']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3,random_state=12)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model = LogisticRegression()


# In[ ]:


model.fit(train_x, train_y)


# In[ ]:


train_predict = model.predict(train_x)
test_predict = model.predict(test_x)


# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.confusion_matrix(train_y,train_predict)


# In[ ]:


metrics.accuracy_score(train_y,train_predict)


# In[ ]:


metrics.confusion_matrix(test_y,test_predict)


# In[ ]:


metrics.accuracy_score(test_y,test_predict)


# In[ ]:





# **Using KNN in the same dataset**

# In[ ]:


from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


preprocessed_df.head()


# In[ ]:


train_y = train_y.ravel()
train_y = train_y.ravel()


# In[ ]:


for K in range(25):
    K_value = K+1
    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
    neigh.fit(train_x, train_y) 
    predict_y = neigh.predict(test_x)
    print ("Accuracy is ", accuracy_score(test_y,predict_y)*100,"% for K-Value:",K_value)

