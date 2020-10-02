#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("/kaggle/input/iris/Iris.csv")


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


data.drop("Id",axis = 1 , inplace = True)


# In[ ]:


data.columns


# In[ ]:


data.Species.unique()


# In[ ]:


data.Species=[0 if i == "Iris-setosa" else 1 if  i ==  "Iris-versicolor" else 2 for i in data.Species]


# In[ ]:


data.Species.unique()


# In[ ]:


y = data.Species.values
X = data.drop("Species",axis = 1).values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


# In[ ]:


from sklearn import metrics


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[ ]:


k_range = list(range(1, 26))
scores = []
for i in k_range:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
plt.plot(k_range, scores)
plt.show()


# In[ ]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=False).split(range(25))


# In[ ]:


print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
for iteration, data in enumerate(kf, start=1):
    print('{:^9} {} {:^25}'.format(iteration, data[0], str(data[1])))


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)


# In[ ]:


print(scores.mean())


# In[ ]:


k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=20)
print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())


# In[ ]:


logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


k_range = list(range(1, 31))
print(k_range)


# In[ ]:


param_grid = dict(n_neighbors=k_range)
print(param_grid)


# In[ ]:


grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)
grid.fit(X, y)


# In[ ]:


grid_mean_scores = grid.cv_results_['mean_test_score']
print(grid_mean_scores)


# In[ ]:


plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# In[ ]:


print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)


# In[ ]:


k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']
param_grid = dict(n_neighbors=k_range, weights=weight_options)
print(param_grid)


# In[ ]:


grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)
grid.fit(X, y)


# In[ ]:


pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]


# In[ ]:


print(grid.best_score_)
print(grid.best_params_)

