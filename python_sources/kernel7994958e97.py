#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


iris = pd.read_csv('../input/Iris.csv')
iris.head()


# In[ ]:


print(iris.shape)


# In[ ]:


iris.set_index('Id',inplace=True)


# In[ ]:


iris.columns


# In[ ]:


X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']


# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=100)


# In[90]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

k_range = range(1,26)
score = {}

for k in k_range:
    knn = KNeighborsClassifier(k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    score[k] = metrics.accuracy_score(y_test,y_pred)


# In[91]:


score


# > **Since, k=7 is giving me 100% results, we can go for k=7**

# 
