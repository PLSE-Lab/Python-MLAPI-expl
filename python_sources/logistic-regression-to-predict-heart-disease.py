#!/usr/bin/env python
# coding: utf-8

# # LOGISTIC REGRESSION TO PREDICT HEART DISEASE.
# ![](http://)

# # Introduction

# World Health Organization has estimated 12 million deaths occur worldwide, every year due to Heart diseases. Half the deaths in the United States and other developed countries are due to cardio vascular diseases. The early prognosis of cardiovascular diseases can aid in making decisions on lifestyle changes in high risk patients and in turn reduce the complications. This research intends to pinpoint the most relevant/risk factors of heart disease as well as predict the overall risk using logistic regression.
# 

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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
sns.set()


# # reading data file

# In[ ]:


data=pd.read_csv("/kaggle/input/heart-disease-prediction-using-logistic-regression/framingham.csv")
data.head(10)


# In[ ]:


data.describe(include="all")


# # Handling Missing Values

# In[ ]:


data.isna().sum()


# education has 105 missing values.
# 
# cigsPerDay has 29 missing values.
# 
# BPMeds has 53 missing values.
# 
# totChol has 50 missing values.
# 
# BMI has 19 missing values.
# 
# heartRate has 1 missing values
# 
# glucose has 388 missing values.

# In[ ]:


from statistics import mode 


# First we will check whether the information is categorical or continuous.
# 
# If they are continuous null values can be replaced by mean(The Arithmetic Mean is the average of the numbers).
# 
# If they are categorical null values can be replaced by mode(The number which appears most often in a set of numbers).

# In[ ]:


data["education"].unique()


# education is categorical.
# so we replace by mode.

# In[ ]:


data["education"]=data["education"].fillna(mode(data["education"]))


# In[ ]:


data["education"].isna().sum()


# In[ ]:


data["cigsPerDay"].unique()


# cigsPerDay seems to be continuous.
# so we replace by mean.

# In[ ]:


data["cigsPerDay"]=data["cigsPerDay"].fillna(data["cigsPerDay"].mean())


# In[ ]:


data["cigsPerDay"].isna().sum()


# In[ ]:


data["BPMeds"].unique()


# BPMeds is categorical.
# so we replace by mode.

# In[ ]:


data["BPMeds"]=data["BPMeds"].fillna(mode(data["BPMeds"]))


# In[ ]:


data["BPMeds"].isna().sum()


# In[ ]:


data["totChol"].unique()


# totChol seems to be continuous.
# so we replace by mean.

# In[ ]:


data["totChol"]=data["totChol"].fillna(data["totChol"].mean())


# In[ ]:


data["totChol"].isna().sum()


# In[ ]:


data["glucose"].unique()


# glucose seems to be continuous.
# so we replace by mean.

# In[ ]:


data["glucose"]=data["glucose"].fillna(data["glucose"].mean())


# BMI & heartRate still have null values but the number is too less .
# 
# so we simply drop the rows.

# In[ ]:


data=data.dropna()


# we checks if further null values are there.

# In[ ]:


data.isna().sum()


# # Data Exploration

# In[ ]:


data.head(5)


# In[ ]:


data.describe(include="all")


# when we see the dependent feature the mean is quite low around 0.15 ie the dataset is imbalanced.

# we try to see them separately

# In[ ]:


X_r=data.drop(["TenYearCHD"],axis=1)
y_r=data["TenYearCHD"]


# Feature Selection

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[ ]:


bf=SelectKBest(score_func=chi2,k=10)
fit=bf.fit(X_r,y_r)


# In[ ]:


dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(X_r.columns)


# In[ ]:


featurescores=pd.concat([dfcolumns,dfscores],axis=1)
featurescores.columns=["spec","score"]
featurescores


# In[ ]:


print(featurescores.nlargest(10,'score'))


# In[ ]:


X=data[["sysBP","glucose","age","totChol","cigsPerDay","diaBP","prevalentHyp","diabetes","BPMeds","male"]]
y=data["TenYearCHD"]


# In[ ]:


data["TenYearCHD"].value_counts()


# to make it balance we have two option under sampling and over sampling 

# using both to understand and see difference

# # under sampling

# In[ ]:


from imblearn.under_sampling import NearMiss


# In[ ]:


nm= NearMiss()
X_res,y_res=nm.fit_sample(X,y)


# In[ ]:


X_res.shape,y_res.shape


# 

# now they seems to be balanced

# we see datatypes of allcolumns using ".dtypes" command

# In[ ]:


data.dtypes


# # Data Splitting and Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.33, random_state=42)


# feature scaling 

# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


logisticRegr = LogisticRegression()


# In[ ]:


logisticRegr.fit(X_train, y_train)


# In[ ]:


predictions = logisticRegr.predict(X_test)


# In[ ]:


score = logisticRegr.score(X_test, y_test)
print(score)


# We are getting accuracy of 72%.
# It can be improved by cross validaation.

# In[ ]:


confusion_matrix = metrics.confusion_matrix(y_test,predictions)
print(confusion_matrix)

