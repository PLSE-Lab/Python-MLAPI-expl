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


import sklearn.datasets as dataset


# In[ ]:


wine_data = dataset.load_wine()


# In[ ]:


wine_data['feature_names']


# In[ ]:


X = wine_data['data']


# In[ ]:


X.shape


# In[ ]:


y=wine_data['target']


# In[ ]:


y.shape


# In[ ]:


X.ndim


# In[ ]:


X.shape


# In[ ]:


y.ndim


# In[ ]:


y.shape


# In[ ]:


y[0:5]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[ ]:


X_train[0:5]


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)


# In[ ]:


y_pred = knn.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


from sklearn.metrics import confusion_matrix
con_mat = confusion_matrix(y_test,y_pred)
con_mat


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# **Implementing the same using GridSearchCV**

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid={"weights":["uniform","distance"],
           "algorithm":["ball_tree","kd_tree","brute"],
           "n_neighbors":np.arange(2,10)}


# In[ ]:


gridcv = GridSearchCV(knn,param_grid=param_grid,cv=3)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'KNeighborsClassifier')


# In[ ]:


knngrid = gridcv.fit(X,y)


# In[ ]:


knngrid.best_score_


# In[ ]:


best_knn = knngrid.best_estimator_


# In[ ]:


best_knn.fit(X_train,y_train)


# In[ ]:


y_pred = best_knn.predict(X_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# * **Implementing the same using RandomSearchCV**

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


get_ipython().run_line_magic('pinfo', 'RandomizedSearchCV')


# In[ ]:


randomCV = RandomizedSearchCV(knn,param_grid,cv=3)


# In[ ]:


clf = randomCV.fit(X,y)


# In[ ]:


clf.best_score_


# In[ ]:


knnrandom = clf.best_estimator_


# In[ ]:


knnrandom.fit(X_train,y_train)


# In[ ]:


y_pred = knnrandom.predict(X_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:




