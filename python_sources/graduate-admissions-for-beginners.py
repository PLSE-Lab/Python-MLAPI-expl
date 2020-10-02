#!/usr/bin/env python
# coding: utf-8

# Purpose of this is to help students in shortlisting universities with their profiles
# 
# If you like my kernel, kindly Upvote it.

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


# ### Importing Libraries and Reading the Dataset

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# Observations :
# 1. Dataset contains 500 records of students
# 2. Avg GRE Score,TOFEL score and CGPA is 316.47, 107.19 and 8.58 respectively 
# 3. Highest GRE Score,TOFEL score and CGPA is 340, 120 and 9.92 respectively 
# 

# In[ ]:


fig, axes = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="magma")


# From above heatmap, we can see that CGPA, GRE score and TOFEL score features has highest correlation with Chance of Admit feature

# DATA VISUALIZATION TO UNDERSTAND THE DATASET

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
sns.distplot(df['CGPA'])
plt.title('Distribution plot for CGPA')


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
sns.distplot(df['GRE Score'])
plt.title('Distribution plot for GRE')


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
sns.distplot(df['TOEFL Score'])
plt.title('Distribution plot for TOFEL score')


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
plt.scatter(x= df['TOEFL Score'], y=df['GRE Score'])
plt.title('TOFEL Score V/s GRE Score')


# In[ ]:


sns.countplot(x='Research', data=df)


# In[ ]:


for i in df[df['Chance of Admit ']>0.75]:
    print(i)


# Let's make this a classification problem by saying students having records where probability of 'Chance of Admit' is more than 80% are admitted and not admitted otherwise

# In[ ]:


from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

y = df['Chance of Admit ']
x = df.drop('Chance of Admit ', axis=1)

x_train, x_test,y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
'''y_train_01 = [1 if each > 0.8 else 0 for each in y_train]
y_test_01 = [1 if each > 0.8 else 0 for each in y_train]'''

y_train[y_train>=0.8] = 1
y_train[y_train<0.8] = 0

y_test[y_test>=0.8] = 1
y_test[y_test<0.8] = 0


# In[ ]:


x_train.drop('Serial No.', axis=1, inplace=True)
x_test.drop('Serial No.', axis=1, inplace=True)
print('Serial No. dropped')


# In[ ]:


columns = x.columns
print(columns.values)


# In[ ]:


x['GRE Score'].unique()


# In[ ]:


plt.hist('GRE Score', data= x)


# In[ ]:


plt.hist('TOEFL Score', data= x)


# In[ ]:


plt.hist('University Rating', data= x)


# In[ ]:


x['SOP'].unique()


# In[ ]:


plt.hist('SOP', data= x)


# In[ ]:


x['LOR '].unique()


# In[ ]:


plt.hist('LOR ', data= x)


# In[ ]:


plt.hist('CGPA', data= x)


# In[ ]:


plt.hist('Research', data= x)


# In[ ]:


x['Research'].unique()


# Model creation, Paramter tuning and Model Validation

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer,f1_score, accuracy_score, precision_score


# In[ ]:


print(x_test.shape)
print(y_test.shape)


# In[ ]:


x_test.head()


# In[ ]:


y_test.head()


# 1. Logistics Regression

# In[ ]:


import warnings
warnings.simplefilter(action='ignore')

acc_scorer = make_scorer(accuracy_score)

accuracy_scoring  = {}
f1_scoring = {}
params = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}
model = GridSearchCV(LogisticRegression(), param_grid = params, scoring = acc_scorer, cv=5)

model.fit(x_train, y_train)
pred = model.predict(x_test)
print('Logistics Regression')
print('Best Parameter', model.best_params_)
print('Training Accuracy Score', model.best_score_)

v1 = accuracy_score(y_test, pred)
v2 = f1_score(y_test, pred)
print('Testing Accuracy Score', v1)
print('Testing f1 Score', v2)
accuracy_scoring['LogisticRegression Testing Accuracy Score'] = v1
f1_scoring['LogisticRegression Testing f1 Score'] = v2


# 2. RandomForestRegressor

# In[ ]:


params = {"n_estimators": [i for i in range(1, 200, 10)]}
print('RandomForest Classifier')
model = GridSearchCV(RandomForestClassifier(), param_grid = params, scoring = acc_scorer, cv=5)

model.fit(x_train, y_train)
pred = model.predict(x_test)
print('Best Parameter', model.best_params_)
print('Training Accuracy Score', model.best_score_)

v3 = accuracy_score(y_test, pred)
v4 = f1_score(y_test, pred)
print('Testing Accuracy Score', v3)
print('Testing f1 Score', v4)
accuracy_scoring['RandomForestClassifier Testing Accuracy Score'] = v3
f1_scoring['RandomForestClassifier Testing f1 Score'] = v4


# 3. Support Vector Machine

# In[ ]:


print('Support Vector Machine')
param_grid = {
    'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf']
}
model = GridSearchCV(SVC(), param_grid = param_grid, cv=5, scoring = acc_scorer)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print('Training Accuracy ', model.best_score_)
v5 = accuracy_score(y_test, pred)
v6 = f1_score(y_test, pred)
print('Best Parameter', model.best_params_)
print('Testing Accuracy Score', v5)
print('Testing F1 Score', v6)
accuracy_scoring['SVC Testing Accuracy Score'] = v5
f1_scoring['SVC Testing f1 Score'] = v6


# 4. Gaussian NB

# In[ ]:


print('Gaussian NB')
param_grid = {}

model = GridSearchCV(GaussianNB(), param_grid = param_grid, cv=5, scoring = acc_scorer)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print('Training Accuracy ', model.best_score_)
v7 = accuracy_score(y_test, pred)
v8 = f1_score(y_test, pred)

print('Testing Accuracy Score', v7)
print('Testing F1 Score', v8)
accuracy_scoring['GaussianNB Testing Accuracy Score'] = v7
f1_scoring['GaussianNB Testing f1 Score'] = v8


# In[ ]:


for i, j in enumerate(accuracy_scoring):
    print(j, accuracy_scoring[j])


# In[ ]:


for i, j in enumerate(f1_scoring):
    print(j, round(f1_scoring[j], 2))


# From above statistics it is clear that, SVM with Best Parameter as {'C': 10, 'kernel': 'linear'}

# In[ ]:


model = SVC(C= 10, kernel= 'linear')
model.fit(x_train, y_train)
pred = model.predict(x_test)

print('Training Accuracy Score ', accuracy_score(y_test, pred))
print('Testing F1 Score ', round(f1_score(y_test, pred), 2))
print('Testing Precision Score ', round(precision_score(y_test, pred), 2))

