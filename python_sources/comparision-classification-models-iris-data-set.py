#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.datasets import load_iris
iris=load_iris()


# In[ ]:


#features
X=iris.data
X


# In[ ]:


iris.feature_names


# In[ ]:


#label
y=iris.target
y


# In[ ]:


#flower species
iris.target_names


# Goal is to classify the flower species as per their sepal and petal length
# 

# Lets split the data for training and testing so that we can apply different classifiers to train and predict

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2) # train/test ratio is 80/20


# we are not using features scaling here,although its a good way to standardize the data

# KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier()
kn.fit(X_train,y_train)
kn.predict(X_test)
kn_score=kn.score(X_test,y_test)
kn_score


# Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=1)
lr.fit(X_train,y_train)
lr.predict(X_test)
lr_score=lr.score(X_test,y_test)
lr_score


# Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=1)
dt.fit(X_train,y_train)
dt.predict(X_test)
dt_score=dt.score(X_test,y_test)
dt_score


# Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(random_state=1)
rf.fit(X_train,y_train)
rf.predict(X_test)
rf_score=dt.score(X_test,y_test)
rf_score


# Gradient boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
gb.fit(X_train,y_train)
gb.predict(X_test)
gb_score=dt.score(X_test,y_test)
gb_score


# Lets see the scores all togeather
# 

# In[ ]:


print("KNN_score: ",kn_score)
print("logistic regression_score: ",lr_score)
print("decision tree_score: ",dt_score)
print("random forest_score: ",rf_score)
print("gradient boosting_score: ",gb_score)

