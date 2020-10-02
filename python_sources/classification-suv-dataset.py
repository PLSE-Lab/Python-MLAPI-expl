#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[ ]:


data = pd.read_csv('../input/classification-suv-dataset/Social_Network_Ads.csv')


# In[ ]:


data_nb = data


# In[ ]:


data_nb.head()


# In[ ]:


X = data_nb.iloc[:, [2,3]].values
y = data_nb.iloc[:, 4].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[ ]:


sc_X = StandardScaler()


# In[ ]:


X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[ ]:


classifier=GaussianNB()
classifier.fit(X_train,y_train)


# In[ ]:


y_pred=classifier.predict(X_test)


# In[ ]:


acc=accuracy_score(y_test, y_pred)


# In[ ]:


print(acc)

