#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from seaborn import pairplot


# In[ ]:


data = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")


# In[ ]:


pairplot(data,hue="species")


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(data.drop("species",axis=1),data["species"],test_size=0.3)


# In[ ]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()


# In[ ]:


clf = clf.fit(x_train, y_train) #si mesin learningnya disini
clf


# In[ ]:


hasil_prediksi = clf.predict(x_test) #untuk Prediksi
hasil_prediksi


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test,hasil_prediksi)


# In[ ]:


data.head()

