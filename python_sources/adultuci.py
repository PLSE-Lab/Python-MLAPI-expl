#!/usr/bin/env python
# coding: utf-8

# Import all used libraries

# In[ ]:


import pandas as pd
import sklearn
import numpy as np
import os
cwd = os.getcwd()


# Read the archives train and test

# In[ ]:


train_adult = pd.read_csv("../input/adultb/train_data.csv" ,names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        skiprows = 1,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# In[ ]:


test_adult = pd.read_csv("../input/adultb/test_data.csv" ,names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        skiprows = 1,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")


# Remove missing data

# In[ ]:


natrain_adult = train_adult.dropna()
natest_adult = test_adult.dropna()


# Analysing the data through graphs

# In[ ]:


import matplotlib.pyplot as plt
natest_adult["Capital Gain"].value_counts().plot(kind='bar')
natrain_adult["Hours per week"].value_counts().plot(kind='pie')


# In[ ]:


targetxrace = pd.crosstab(natrain_adult["Race"],natrain_adult["Target"],margins=False)
targetxrace.plot(kind='bar',stacked=False)
targetxrace


# In[ ]:


targetxrace = pd.crosstab(natrain_adult["Sex"],natrain_adult["Target"],margins=False)
targetxrace.plot(kind='bar',stacked=True)


# Parse the data and create a classifier

# In[ ]:


from sklearn import preprocessing
adult_train = natrain_adult[["Age", "Workclass", "Education-Num", "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week"]]
adult_test = natest_adult[["Age", "Workclass", "Education-Num", "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week"]]
Xtrainadult = adult_train.apply(preprocessing.LabelEncoder().fit_transform)
Ytrainadult = natrain_adult.Target
Xtestadult = adult_test.apply(preprocessing.LabelEncoder().fit_transform)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier as knnClassifier
knn = knnClassifier(n_neighbors = 35)


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xtrainadult, Ytrainadult,cv=10)
scores


# In[ ]:


knn.fit(Xtrainadult, Ytrainadult)
predYtest = knn.predict(Xtestadult)
predYtest


# In[ ]:


income = pd.DataFrame(predYtest)
income.to_csv("submission.csv",header = ["income"], index_label = "Id")


# There begins the notebook for the Costa Rican Household Poverty Analysis

# In[ ]:


train = pd.read_csv("../input/costa-rican-household-poverty-prediction/train.csv")
test = pd.read_csv("../input/costa-rican-household-poverty-prediction/test.csv")


# In[ ]:


ntrain = train.dropna()


# In[ ]:


from sklearn import preprocessing
Xtrain = train.iloc[:,0:-1]
Ytrain = train.Target
Xtest = test
nXtrain = Xtrain.apply(preprocessing.LabelEncoder().fit_transform)
nXtest = Xtest.apply(preprocessing.LabelEncoder().fit_transform)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier as KnnC
knn = KnnC(n_neighbors = 30)


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, nXtrain, Ytrain, cv=10)
score = np.mean(scores)


# In[ ]:


knn.fit(nXtrain, Ytrain)
predYtest = knn.predict(nXtest)


# In[ ]:


poverty = pd.DataFrame(predYtest)
poverty.to_csv("submission.csv", header = ["Target"], index_label = 'Id')

