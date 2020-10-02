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


df=pd.read_csv("/kaggle/input/credit_history.csv")


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df["years"].fillna(df["years"].mean(),inplace=True)


# In[ ]:


df.describe()


# In[ ]:


x=df.drop("default",axis=1)


# In[ ]:


x.head()


# In[ ]:


x=pd.get_dummies(x)
x.head()


# In[ ]:


y=df['default']


# In[ ]:


import sklearn.model_selection as ms
import sklearn.tree as Tree


# In[ ]:


x_train,x_test,y_train,y_test=ms.train_test_split(x,y,test_size=0.3,random_state=200)
clf=Tree.DecisionTreeClassifier(max_depth=2,random_state=200)
clf.fit(x_train,y_train)
clf.score(x_test,y_test)


# In[ ]:


rclf=Tree.DecisionTreeRegressor(max_depth=3,random_state=200)
rclf.fit(x_train,y_train)
rclf.score(x_test,y_test)


# In[ ]:


dot_data=Tree.export_graphviz(clf,out_file=None,feature_names=x.columns,class_names=["0","1"],filled=True,rounded=True,
                             special_characters=True,proportion=True)


# In[ ]:


import graphviz
graph=graphviz.Source(dot_data)
graph


# In[ ]:


grp=Tree.export_graphviz(rclf,out_file=None,feature_names=x.columns,class_names=["0","1"],special_characters=True,proportion=True,rounded=True,filled=True)
graph2=graphviz.Source(grp)
graph2


# In[ ]:


##cross validation grid search


# In[ ]:


clf=Tree.DecisionTreeClassifier(max_depth=3,random_state=200)
mod=ms.GridSearchCV(clf,param_grid={'max_depth':[2,3,4,5]})
mod.fit(x_train,y_train)


# In[ ]:


mod.best_estimator_


# In[ ]:


mod.best_score_


# In[ ]:




