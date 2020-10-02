#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) time import time
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score


get_ipython().run_line_magic('matplotlib', '')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#loading winequality datasets
winedata=pd.read_csv("/kaggle/input/winequality-white.csv",sep=';')
display(winedata.head(n=10))
winedata.head()


# In[ ]:


#Cecking if data is distributed evenly
winedata.info


# In[ ]:


#checking if there is any missing data
data.isnull().any


# In[ ]:


#Performing perliminary analysis on the data sets
n_wines = data.shape[0]

# Number of wines with quality rating above 6
quality_above_6 = data.loc[(data['quality'] > 6)]
n_above_6 = quality_above_6.shape[0]

# Number of wines with quality rating below 5
quality_below_5 = data.loc[(data['quality'] < 5)]
n_below_5 = quality_below_5.shape[0]

# Number of wines with quality rating between 5 to 6
quality_between_5 = data.loc[(data['quality'] >= 5) & (data['quality'] <= 6)]
n_between_5 = quality_between_5.shape[0]
# Percentage of wines with quality rating above 6
greater_percent = n_above_6*100/n_wines

# Print the results
print("Total number of wine data: {}".format(n_wines))
print("Wines with rating 7 and above: {}".format(n_above_6))
print("Wines with rating less than 5: {}".format(n_below_5))
print("Wines with rating 5 and 6: {}".format(n_between_5))
print("Percentage of wines with quality 7 and above: {:.2f}%".format(greater_percent))

# Some more additional data analysis
display(np.round(data.describe()))


# In[ ]:


#Comparing Citric acid factor with fixed acidity
fig = plt.figure(figsize=(10,5))
sns.barplot (x = 'citric acid',y = 'fixed acidity',data= winedata)


# In[ ]:


#Comparing Quality with fixed acidity
fig = plt.figure(figsize=(10,5))
sns.barplot (x = 'quality',y = 'fixed acidity',data= winedata)


# In[ ]:


#Comparing Volatile acid with fixed acidity
fig = plt.figure(figsize=(10,5))
sns.barplot (x = 'alcohol',y = 'citric acid',data= winedata) 


# In[ ]:


#Comparing Volatile acid with fixed acidity
fig = plt.figure(figsize=(10,5))
sns.barplot (x = 'chlorides',y = 'fixed acidity',data= winedata) 


# In[ ]:


#Comparing residual sugars with density
fig = plt.figure(figsize=(10,5))
sns.barplot (x = 'residual sugar',y = 'density',data= winedata)


# In[ ]:


#Comparing citric acids factors with density factor
fig = plt.figure(figsize=(10,5))
sns.barplot (x = 'citric acid',y = 'density',data= winedata)


# In[ ]:


#Comparing quality factor with density
fig = plt.figure(figsize=(10,5))
sns.barplot (x = 'quality',y = 'density',data= winedata) 


# In[ ]:


#Comparing volatile acidity factor with residual sugars factor 
fig = plt.figure(figsize =(10,5))
sns.barplot(x= 'volatile acidity',y ='residual sugar',data=winedata)


# In[ ]:


#Comparing quality factor with citric acid factor
fig = plt.figure(figsize =(10,5))
sns.barplot(x='quality',y='citric acid',data=winedata)


# In[ ]:


#Comparing Volatile acid with fixed acidity
fig = plt.figure(figsize=(10,5))
sns.barplot (x = 'alcohol',y = 'quality',data= winedata) 


# In[ ]:


bins = (2,6.5,8)
group_names = ['bad','good']

winedata['quality']= pd.cut(winedata['quality'],bins=bins, labels = group_names)


# In[ ]:


label_quality= LabelEncoder()


# In[ ]:


winedata['quality'] =label_quality.fit_transform(winedata['quality'])


# In[ ]:


winedata['quality'].value_counts()


# In[ ]:


sns.countplot(winedata['quality'])


# In[ ]:


X=winedata.drop('quality',axis=1)
Y=winedata['quality']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state =42)


# In[ ]:


sc = StandardScaler

X_train = sc.fit_tansform(X-train)
X_test = sc.fit_transform(X-test)
# <h1>Random Forest Classifier</h1>
# 

# In[ ]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test,pred_rfc))


# In[ ]:


print(confusion_matrix(y_test,pred_rfc))


# <h1>STOCHASTIC GRADIENT DESCENT CLASSIFIER<H1>

# In[ ]:


sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)
print(classification_report(y_test,pred_sgd))


# <H1>SUPPORT VECTOR CLASSIFIER<H1>

# In[ ]:


svc = SVC()
svc.fit(X_train , y_train)
pred_svc = svc.predict(X_test)
print(classification_report(y_test,pred_svc))


# # GRID SEARCH CV

# In[ ]:


param = { 'C':[0.1,0.2,0.4,0.8,1.6],
            'kernel':['liner', 'rbf ']
            'gamma' : [0.1,0.3,0.5,0.7]}
grid_svc= GridSearchCV(svc,param_grid = param ,scoring ='accuracy',cv=10)


# In[ ]:


grid_svc.fit(X_train,y_train)
grid_svc.best_params_


# In[ ]:


svc2 = SVC(C = 1.2 , gamma = 0.9 , kernel = 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)


# In[ ]:


print(classification_report(y_test, pred_svc2))

