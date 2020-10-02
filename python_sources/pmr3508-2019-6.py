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


import sklearn


# In[ ]:


adult = pd.read_csv("/kaggle/input/adult_data.txt",
                    names=[
                    "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


adult.head()


# In[ ]:


nAdult = adult.dropna()


# In[ ]:


testAdult = pd.read_csv("/kaggle/input/adult_test.txt",
                    names=[
                    "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


testAdult


# In[ ]:


testAdult.drop([0], axis = 0)


# In[ ]:


nTestAdult = testAdult.dropna()


# In[ ]:


nTestAdult.shape


# In[ ]:


xAdult = nAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]


# In[ ]:


yAdult = nAdult.Target


# In[ ]:


xTestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]


# In[ ]:


yTestAdult = nTestAdult["Target"]


# In[ ]:


yTestAdult.shape


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


scores = cross_val_score(knn, xAdult, yAdult, cv=10)


# In[ ]:


scores


# In[ ]:


knn.fit(xAdult,yAdult)


# In[ ]:


yTestPred = knn.predict(xTestAdult)


# In[ ]:


yTestPred


# In[ ]:


yTestAdult


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


print(accuracy_score(yTestPred, yTestAdult)*100)


# In[ ]:


accuracy_score(yTestPred, yTestAdult)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=30)


# In[ ]:


knn.fit(xAdult,yAdult)


# In[ ]:


scores = cross_val_score(knn, xAdult, yAdult, cv=10)


# In[ ]:


yTestPred = knn.predict(xTestAdult)


# In[ ]:


accuracy_score(yTestAdult,yTestPred)


# In[ ]:


from sklearn import preprocessing


# In[ ]:


numAdult = nAdult.apply(preprocessing.LabelEncoder().fit_transform)


# In[ ]:


numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)


# In[ ]:


xAdult = numAdult[["Age",  "Education-Num",  "Race", "Sex", "Capital Gain", "Capital Loss",]]


# In[ ]:


xTestAdult = numTestAdult[["Age",  "Education-Num",  "Race", "Sex", "Capital Gain", "Capital Loss",]]


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=30)


# In[ ]:


knn.fit(xAdult,yAdult)


# In[ ]:


scores = cross_val_score(knn, xAdult, yAdult, cv=10)


# In[ ]:


yTestPred = knn.predict(xTestAdult)


# In[ ]:


accuracy_score(yTestAdult,yTestPred)


# In[ ]:




