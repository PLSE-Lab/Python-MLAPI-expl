#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#Loading our dataset
dataset = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")


# In[ ]:


#Viewing first 5 entries in our dataset
dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


#Checking if any null values are present or not in our dataset
dataset.isnull().sum()


# In[ ]:


#Removing the null values and replacing the null values with zeros in our dataset

dataset.fillna(0, inplace = True)


# In[ ]:


#Finding whether any null valued columns are present or not
dataset.isnull().any()


# As we don't have any null values in our dataset, now we are going to make the prediction of the students who have the chances to placed
# 
# Here we consider the following columns they are as follows:
# 
# ssc_p = Percentage obtained by students in ssc
# 
# hsc_p = Percentage obtained by students in hsc
# 
# degree_p = Percentage obtained by students in degree
# 
# etest_p = Percentage obtained by students in employability test
# 
# mba_p = Percentage obtained by students in mba

# In[ ]:


new_dataset = dataset[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'status']]


# In[ ]:


new_dataset.head()


# In[ ]:


#Dividing the features and labels in X and Y

X = new_dataset.iloc[:, :-1]
Y = new_dataset.iloc[:, -1]


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 3)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


log_reg = LogisticRegression()


# In[ ]:


log_reg.fit(X_train, Y_train)


# In[ ]:


log_reg.intercept_


# In[ ]:


log_reg.coef_


# In[ ]:


Y_pred = log_reg.predict(X_test)


# In[ ]:


df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
df


# In[ ]:


print("The accuracy of the predicted values is: ", metrics.accuracy_score(Y_test, Y_pred))


# In[ ]:




