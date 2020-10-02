#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **KneighborsClassifier and Matplotlib for sns**

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv("../input/IRIS.csv")


# In[ ]:


df.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x = df.iloc[ :, :-1]
y = df.iloc[:, 4]
print(x.shape)
print (y.shape)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[ ]:


print(x_train.shape)
print(y_train.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=6)


# In[ ]:


knn.fit(x_train,y_train)


# In[ ]:


y_pred=knn.predict(x_test)


# In[ ]:


knn.score(x_test,y_test)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.figure()


# In[ ]:


import seaborn as sns


# In[ ]:


sns.pairplot(data= df,hue='species',palette='RdBu')


# In[ ]:


plt.xticks([0,1],['features','species'])


# **Confusion Matrix**

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


from sklearn.metrics import roc_curve


# **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logreg=LogisticRegression()


# In[ ]:


print(logreg)


# In[ ]:




