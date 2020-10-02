#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/social-network-ads/Social_Network_Ads.csv')


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


X = df.iloc[:,2:4].values


# In[ ]:


y = df.iloc[:,-1].values


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:


X_train=scaler.fit_transform(X_train)


# In[ ]:


X_test=scaler.transform(X_test)


# ## Value Of K :
# we will follow two methods
# - 1st method np.sqrt(X_train.shape)
# - hit and ttrail

# In[ ]:


## method_1
np.sqrt(X_train.shape)[0]


# In[ ]:


k=17


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=k)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


y_pred=knn.predict(X_test)


# In[ ]:


y_pred.shape


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[ ]:


## method 2


# In[ ]:


accuracy=[]
for i in range(1,26):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    accuracy.append(accuracy_score(y_test,knn.predict(X_test)))


# In[ ]:


accuracy


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.plot(range(1,26),accuracy)


# In[ ]:



knn=KNeighborsClassifier(n_neighbors=7)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


y_pred=knn.predict(X_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:




