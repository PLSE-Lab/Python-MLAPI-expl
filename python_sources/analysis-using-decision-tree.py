#!/usr/bin/env python
# coding: utf-8

# This model is completely based on the decision tree classifier. So will only be using the decision tree.

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[ ]:


data=load_iris() #we can directly load the data from Sklearn,as it already have it.  
print('Classes to predict: ', data.target_names)


# We can see that there are three target classes. The prediction of the petal types will be based on these three.  

# In[ ]:


data


# In[ ]:


X=data.data
y=data.target
print('Number of examples in the data:', X.shape[0])


# In[ ]:


X[:5] # upper 5 rows


# In[ ]:


X_train, X_test, y_train,y_test=train_test_split(X,y,random_state=47,test_size=0.30)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf= DecisionTreeClassifier(criterion='entropy')


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


y_pred=clf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
print('Accuracy score on the train data: ',accuracy_score(y_true=y_train,y_pred=clf.predict(X_train)))
print('Accuracy score on the test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))


# In[ ]:


clf=DecisionTreeClassifier(criterion='entropy',min_samples_split=40)
clf.fit(X_train, y_train)
print('Accuracy score on the train data: ', accuracy_score(y_true=y_train,y_pred=clf.predict(X_train)))
print('Accuracy score on the test data: ', accuracy_score(y_true=y_test,y_pred=clf.predict(X_test)))


# I tried experimenting with the minimum sample split value. The best result we get is at 40.
# 
# Fine tuning of other components can give better result.

# In[ ]:




