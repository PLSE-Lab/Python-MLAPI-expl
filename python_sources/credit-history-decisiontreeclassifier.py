#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd


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

# Any results you write to the current directory are saved as output


# In[ ]:


dat=pd.read_csv("/kaggle/input/credit-history/credit_history.csv")
dat.head()


# In[ ]:


dat.isnull().sum()


# In[ ]:


dat['years'].describe()


# In[ ]:


dat['years'].fillna(4,inplace=True)


# In[ ]:


X=dat.drop("default",axis=1)


# In[ ]:


X.head()


# In[ ]:


X=pd.get_dummies(X)


# In[ ]:


X.head()


# In[ ]:


y=dat['default']


# In[ ]:


import sklearn.model_selection as model_selection
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2,random_state=200)


# In[ ]:


import sklearn.tree as tree
clf=tree.DecisionTreeClassifier(max_depth=3,random_state=200)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# In[ ]:


import sklearn.metrics as metrics
metrics.roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])


# In[ ]:


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


# In[ ]:


dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=X.columns,  
                         class_names=["0","1"],  
                         filled=True, rounded=True,  
                         special_characters=True,proportion=True)


# In[ ]:


import graphviz
graph = graphviz.Source(dot_data)


# In[ ]:


graph


# ### Grid Search-Cross Validation

# In[ ]:


clf=tree.DecisionTreeClassifier(max_depth=3,random_state=200)


# In[ ]:


mod=model_selection.GridSearchCV(clf,param_grid={'max_depth':[2,3,4,5,6]})
mod.fit(X_train,y_train)


# In[ ]:


mod.best_estimator_


# In[ ]:


mod.best_score_


# In[ ]:


help DecisionTreeClassifier()


# In[ ]:





# In[ ]:




