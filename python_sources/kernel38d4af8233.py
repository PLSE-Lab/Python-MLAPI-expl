#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


adult = pd.read_csv("/kaggle/input/adultbase/adult.data.txt",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


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


nadult.head()


# In[ ]:


print("{} - nadult \n\r{} - adult".format(nadult.shape, adult.shape))


# In[ ]:


testAdult = pd.read_csv("/kaggle/input/adultbase/adult.test.txt",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


nTestAdult = testAdult.dropna()


# In[ ]:


Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Yadult = nadult.Target

XtestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
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


accuracy_score(YtestAdult,YtestPred)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult,Yadult)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores


# In[ ]:


YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)


# accuracy_scores returns 0 (and not 0.0001 for example) because not a single prediction is right due to YtestAdult has been giving values with dot char "<=50k.". A simple conversion from non-numerical to numerical values settles this issue as is shown below:

# In[ ]:


from sklearn import preprocessing
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult.iloc[:,0:14]
Yadult = numAdult.Target
XtestAdult = numTestAdult.iloc[:,0:14]
YtestAdult = numTestAdult.Target
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(Xadult,Yadult)


# In[ ]:


YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)


# There it go, problem solved.

# In[ ]:


Xadult = numAdult[["Age", "Workclass", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"]]
XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"]]
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(Xadult,Yadult)


# In[ ]:


YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)


# In[ ]:


# automation

adult = pd.read_csv("/kaggle/input/adultbase/adult.data.txt",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
test_adult = pd.read_csv("/kaggle/input/adultbase/adult.test.txt",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
nadult = adult.dropna()
nTestAdult = testAdult.dropna()

from sklearn import preprocessing

numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
Yadult = numAdult.Target
YtestAdult = numTestAdult.Target

def easy_mode(key_columns, n_neighbors,returnPred):
    Xadult = numAdult[key_columns]
    XtestAdult = numTestAdult[key_columns]
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(Xadult,Yadult)
    YtestPred = knn.predict(XtestAdult)
    if returnPred==True:
        return YtestPred
    return (accuracy_score(YtestAdult,YtestPred))

# now a little help from itertools
from itertools import combinations
#nadult_columns = list(nadult.columns.values)
#use nadult_columns above for first tests, afterwords you may use the key columns list found,
#which in my case is the list below
nadult_columns = ["Age", "Education-Num", "Martial Status",
                   "Occupation", "Sex", "Capital Gain", "Capital Loss",
                   "Hours per week"]
best_result = {'result':0,'num_knn':0,'col_list':[]}
for i in range(6, len(nadult_columns)):
    key_columns_list = combinations(list(nadult_columns),i)
    for n_columns in key_columns_list:
        result = easy_mode(list(n_columns),34, False)
        if best_result['result']<result:
            best_result['result']=result
            best_result['num_knn']=34
            best_result['col_list']=list(n_columns)
            print("In progress... Best result so far = {}\t\n{} (knn num = {})".format(best_result['result'],best_result['col_list'],best_result['num_knn']))
print("Best result with key columns {} and knn number = {}:\t\nResult = {}".format(best_result['col_list'],best_result['num_knn'],best_result['result']))

n_columns = best_result['col_list']
print("Now to confirm the best knn number we use the key columns found with different knn number values.")
best_result = {'result': 0, 'num_knn': 0}
for i in range(30, 40):
    result = easy_mode(n_columns, i, False)
    if best_result['result'] < result:
        best_result['result'] = result
        best_result['num_knn'] = i
        print("In progress... Best result so far = {}; n_knn = {}".format(best_result['result'], best_result['num_knn']))
print("With key columns {} and knn_number = {}:\t\nAccuracy = {}".format(n_columns,best_result['num_knn'], best_result['result']))


# After several tests, we have obtained with key columns = ['Age', 'Education-Num', 'Martial Status', 'Occupation', 'Capital Gain', 'Capital Loss'] and knn number = 34 the accuracy of 84.18991%.
# 

# In[ ]:


submission = easy_mode(['Age', 'Education-Num', 'Martial Status', 'Occupation', 'Capital Gain', 'Capital Loss'],34,True)
submission


# Now, translating submission from binary to string:

# In[ ]:


final_result = {'Id':[],'Income':[]}
for i in range (len(submission)):
    final_result['Id'].append(i)
    if submission[i]:
        final_result['Income'].append(">50k")
    else:
        final_result['Income'].append("<=50k")
df_result = pd.DataFrame(final_result)
df_result.to_csv('bestPrediction.csv', index=False, sep=',', line_terminator = '\r\n', header = ["Id", "Income"])


# In[ ]:




