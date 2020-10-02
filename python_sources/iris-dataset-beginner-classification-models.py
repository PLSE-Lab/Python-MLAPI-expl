#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


iris = pd.read_csv('../input/Iris.csv')


# In[ ]:


iris.head(2)


# In[ ]:


iris.tail(2)


# In[ ]:


iris.info()


# In[ ]:


iris1 = iris.drop("Id",axis=1)


# In[ ]:


#Renaming columns
iris1.columns=['sepal_length','sepal_width','petal_length','petal_width','species']


# In[ ]:


iris1.head(1)


# In[ ]:


iris1.dtypes


# In[ ]:


#changing the data type of species to category
iris1['species']=iris1['species'].astype('category')
iris1.dtypes


# In[ ]:


iris1.species.unique()


# In[ ]:


iris2 = pd.DataFrame(iris1.species.value_counts())
iris2


# In[ ]:


iris1.shape


# In[ ]:


iris2.shape


# In[ ]:


iris1.describe()


# In[ ]:


iris1.size


# In[ ]:


iris1.isnull().sum()


# In[ ]:


iris1.species =iris1.species.map({'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica':2})


# In[ ]:


iris1.head(2)


# In[ ]:


#Realtion between sepal length and width
#iris1.plot(kind='scatter',x='species', y='sepal_length')

fig = iris1[iris1.species==0].plot(kind='scatter', x='sepal_length', y='sepal_width', color='orange', label='Setosa')
iris1[iris1.species==1].plot(kind='scatter', x='sepal_length', y='sepal_width', color='blue',label='versicolor',ax=fig)
iris1[iris1.species==2].plot(kind='scatter', x='sepal_length', y='sepal_width', color='green',label='veriginica',ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length Vs Width")


# In[ ]:


#relation between petal length and width
fig = iris1[iris1.species==0].plot(kind='scatter', x='petal_length', y='petal_width', color='orange', label='Setosa')
iris1[iris1.species==1].plot(kind='scatter', x='petal_length', y='petal_width', color='blue',label='versicolor',ax=fig)
iris1[iris1.species==2].plot(kind='scatter', x='petal_length', y='petal_width', color='green',label='versicolor',ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title("Petal Length Vs Width")


# In[ ]:


iris1.hist(edgecolor='black')


# In[ ]:


iris1_data = iris1.iloc[:,:4]
iris1_data.head(2)


# In[ ]:


iris1_target = iris1.loc[:,'species']
type(iris1_target)


# In[ ]:


X,y = iris1_data, iris1_target


# In[ ]:


#using train_test_split 
X_train, X_test, y_train,y_test = train_test_split(X, y,test_size=.2, random_state=0)


# In[ ]:


#using logistic regression
log = LogisticRegression().fit(X_train, y_train)


# In[ ]:


y_pred = log.predict(X_test)


# In[ ]:


logreg_score = accuracy_score(y_test,y_pred)
logreg_score


# In[ ]:


#using KNeighborclassifer
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)


# In[ ]:


y_pred_k = model.predict(X_test)


# In[ ]:


knn_score = accuracy_score(y_test,y_pred_k)
knn_score


# In[ ]:


k_range=range(1,31)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))


# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(k_range,scores)
plt.xlabel('value of k')
plt.ylabel('scores')


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
y_pred


# Cross validation

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 5)
scores= cross_val_score(knn, X,y,cv=10, scoring='accuracy')
scores


# In[ ]:


scores.mean()


# In[ ]:


k_range=range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X,y,cv=10,scoring='accuracy')
    k_scores.append(scores.mean())
k_scores


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(k_range,k_scores)
plt.xlabel('value of k')
plt.ylabel('scores')


# In[ ]:


#applying the best params from teh kfold 
knn = KNeighborsClassifier(n_neighbors = 20)
scores= cross_val_score(knn, X,y,cv=10, scoring='accuracy')
scores.mean()


# 

# GridSearchCV

# In[ ]:


k_range = range(1,31)
param_grid = dict(n_neighbors=k_range)
param_grid


# In[ ]:


grid= GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')


# In[ ]:


grid.fit(X,y)


# In[ ]:


grid.cv_results_.keys()


# In[ ]:


grid.best_estimator_


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


#applying the best params from the grid search 
knn = KNeighborsClassifier(n_neighbors = 13)
knn.fit(X,y)
X.head(2)
X.shape
#predict the y for input [3,5,4,2]
knn.predict([[3,5,4,2]])


# Decision Tree classifier

# In[ ]:


k_range = range(1,31)
weight_options = ['uniform','distance']


# In[ ]:


param_grid = dict(n_neighbors=k_range,weights=weight_options)
param_grid

grid = GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')
# In[ ]:


grid.fit(X,y)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


grid.predict([[3,5,4,13]])


# Randomised search cv

# In[ ]:


param_dist = dict(n_neighbors=k_range, weights=weight_options)


# In[ ]:


rand = RandomizedSearchCV(knn,param_dist, cv=10,scoring= 'accuracy',n_iter = 10, random_state = 5)
rand.fit(X,y)
rand.cv_results_.keys()


# In[ ]:


rand.best_params_


# In[ ]:


rand.best_score_


# In[ ]:


rand.best_index_
rand.best_estimator_


# 

# In[ ]:


best_scores = []
for _ in range(20):
    rand = RandomizedSearchCV(knn,param_dist,cv =10,scoring='accuracy',n_iter=10)
    rand.fit(X,y)
    best_scores.append(rand.best_score_)
best_scores


# Observation
# Though randomizedSearchCV executed only 10 combinations, it was able to find the best combination (score= 0.98) which was given by the grid search CV.

# 

# 

# 

# 

# 

# 
