#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,r2_score
sns.set_style("darkgrid")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


sns.countplot(df['Gender'])


# In[ ]:


sns.countplot(df['OverTime'])


# In[ ]:


sns.countplot(df['MaritalStatus'])


# In[ ]:


plt.figure(figsize=(15,7))
sns.countplot(df['JobRole'])


# In[ ]:


plt.figure(figsize=(20,7))
sns.countplot(df['EducationField'])


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(df['TotalWorkingYears'])


# In[ ]:


pd.crosstab(df['Attrition'],df['DistanceFromHome']).plot.bar(figsize=(18,5))
# sns.countplot(df.DistanceFromHome)


# In[ ]:


sns.countplot(df['Attrition'])


# In[ ]:


en = LabelEncoder()
df['Age'] = en.fit_transform(df['Age'])
df['BusinessTravel'] = en.fit_transform(df['BusinessTravel'])
df['DailyRate'] = en.fit_transform(df['DailyRate'])
df['Department'] = en.fit_transform(df['Department'])
df['EducationField'] = en.fit_transform(df['EducationField'])
df['JobRole'] = en.fit_transform(df['JobRole'])
df['MaritalStatus'] = en.fit_transform(df['MaritalStatus'])
df['Attrition'] = en.fit_transform(df['Attrition'])
df['OverTime'] = en.fit_transform(df['OverTime'])
df['Over18'] = en.fit_transform(df['Over18'])
df['Gender'] = en.fit_transform(df['Gender'])


# In[ ]:


df.head()


# In[ ]:


X = df.drop('Attrition',axis=1)
y = df['Attrition']
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = []


# In[ ]:


log = LogisticRegression()
log.fit(X_train,y_train)
y_predict = log.predict(X_test)
model.append(["LogisticRegression",accuracy_score(y_predict,y_test)])


# In[ ]:


log = DecisionTreeClassifier()
log.fit(X_train,y_train)
y_predict = log.predict(X_test)
model.append(["DecisionTreeClassifier",accuracy_score(y_predict,y_test)])


# In[ ]:


log = ExtraTreesClassifier()
log.fit(X_train,y_train)
y_predict = log.predict(X_test)
model.append(["ExtraTreesClassifier",accuracy_score(y_predict,y_test)])


# In[ ]:


log = KNeighborsClassifier()
log.fit(X_train,y_train)
y_predict = log.predict(X_test)
model.append(["KNeighborsClassifier",accuracy_score(y_predict,y_test)])


# In[ ]:


log = GaussianNB()
log.fit(X_train,y_train)
y_predict = log.predict(X_test)
model.append(["GaussianNB",accuracy_score(y_predict,y_test)])


# In[ ]:


model = pd.DataFrame(model)
model.columns = ["Name","Score"]


# In[ ]:



model.sort_values(by="Score",ascending=False)


# In[ ]:




