#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split #Cross validation
from sklearn.model_selection import cross_val_score


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png' #set 'png' here when working on notebook")
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv("../input/adult.csv")


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train['age'].hist(bins = 100)


# In[ ]:


capitals = pd.DataFrame({"gains":train['capital.gain'], "losses":train['capital.loss']})
capitals.hist(bins = 12)


# In[ ]:


train['income'] = train['income'].replace({'<=50K': 0, '>50K':1}, regex=True)


# In[ ]:


train.head()


# In[ ]:


train = pd.get_dummies(train)


# In[ ]:


print(train.shape)


# In[ ]:


train.head()


# In[ ]:


data = train.drop('income', axis = 1)
target = train.income
#X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)


# In[ ]:


knn = neighbors.KNeighborsClassifier(20)


# In[ ]:


#knn.fit(X_train, y_train)


# In[ ]:


scores = cross_val_score(knn, data, target, cv=5)


# In[ ]:


scores.mean()


# In[ ]:


scores.std()


# In[ ]:


scores = []
for k in range(1, 100):
    knn = neighbors.KNeighborsClassifier(k)
    scores.append(cross_val_score(knn, data, target, cv=2).mean())


# In[ ]:


scores = pd.Series(scores)
scores.plot()


# In[ ]:


scores.describe()


# In[ ]:




