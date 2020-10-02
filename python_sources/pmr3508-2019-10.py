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
import sklearn


# In[ ]:


adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?",
        skiprows=1)


# In[ ]:


adult.shape


# In[ ]:


adult.head()


# In[ ]:


adult["Country"].value_counts()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


adult["Age"].value_counts().plot(kind="bar")


# In[ ]:


adult["Sex"].value_counts().plot(kind="bar")


# In[ ]:


adult["Education"].value_counts().plot(kind="bar")


# In[ ]:


adult["Occupation"].value_counts().plot(kind="bar")


# In[ ]:


nadult = adult.dropna()


# In[ ]:


nadult


# In[ ]:


testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?",
        skiprows=1)


# In[ ]:


testSampleAdult = pd.read_csv("/kaggle/input/adult-pmr3508/sample_submission.csv",
        names=["Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?",
        skiprows=1)


# In[ ]:


testAdult = pd.concat([testAdult, testSampleAdult], axis=1, join='inner').sort_index()


# In[ ]:


nTestAdult = testAdult.dropna()


# In[ ]:


nTestAdult


# In[ ]:


Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]


# In[ ]:


Yadult = nadult.Target


# In[ ]:


XtestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]


# In[ ]:


YtestAdult = nTestAdult.Target


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


scores = cross_val_score(knn, Xadult, Yadult, cv=10)


# In[ ]:


scores


# In[ ]:


knn.fit(Xadult,Yadult)


# In[ ]:


YtestPred = knn.predict(XtestAdult)


# In[ ]:


YtestPred


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


Yadult


# In[ ]:


accuracy_score(YtestAdult,YtestPred)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=30)


# In[ ]:


knn.fit(Xadult,Yadult)


# In[ ]:


scores = cross_val_score(knn, Xadult, Yadult, cv=10)


# In[ ]:


scores


# In[ ]:


YtestPred = knn.predict(XtestAdult)


# In[ ]:


accuracy_score(YtestAdult,YtestPred)


# In[ ]:


from sklearn import preprocessing


# In[ ]:


numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)


# In[ ]:


numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)


# In[ ]:


Xadult = numAdult.iloc[:,0:14]


# In[ ]:


Yadult = numAdult.Target


# In[ ]:


XtestAdult = numTestAdult.iloc[:,0:14]


# In[ ]:


YtestAdult = numTestAdult.Target


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=30)


# In[ ]:


knn.fit(Xadult,Yadult)


# In[ ]:


YtestPred = knn.predict(XtestAdult)


# In[ ]:


accuracy_score(YtestAdult,YtestPred)


# In[ ]:


Xadult = numAdult[["Age", "Workclass", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"]]


# In[ ]:


XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"]]


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=30)


# In[ ]:


knn.fit(Xadult,Yadult)


# In[ ]:


YtestPred = knn.predict(XtestAdult)


# In[ ]:


accuracy_score(YtestAdult,YtestPred)


# In[ ]:


Xadult = numAdult[["Age", "Workclass", "Education-Num", 
        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week"]]


# In[ ]:


XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", 
        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week"]]


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=30)


# In[ ]:


knn.fit(Xadult,Yadult)


# In[ ]:


YtestPred = knn.predict(XtestAdult)


# In[ ]:


accuracy_score(YtestAdult,YtestPred)

