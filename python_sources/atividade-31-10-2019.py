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
        
import sklearn.datasets as dt

# Any results you write to the current directory are saved as output.


# In[ ]:


dic = dt.load_digits()
dic.keys()


# In[ ]:


dic.data


# In[ ]:


dic.data.shape


# In[ ]:


dic.images.shape


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(dic.images[200])


# In[ ]:


x=dic.data
y=dic.target


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# In[ ]:


X_train.shape


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7)
modelo = knn.fit(X_train, y_train)
y_pred = modelo.predict(X_test)


# In[ ]:


y_score = modelo.score(X_test,y_test)
y_score


# In[ ]:


compara = pd.DataFrame(y_test)
compara['pred']= y_pred
compara.head(40)


# In[ ]:




