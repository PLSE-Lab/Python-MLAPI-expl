#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
import pandas as pd


# In[ ]:


data = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")


# In[ ]:


data.tail()


# In[ ]:


data.info()


# In[ ]:


data


# In[ ]:


from seaborn import pairplot


# In[ ]:


pairplot(data, hue="species")


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(data.drop("species", axis=1),
                                                    data["species"],
                                                        test_size=0.3)


# In[ ]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# **Machine Learning in Phython - Cara Mudah Banget**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier  # MACHINE AKAN LEARNING
clf = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


clf = clf.fit(x_train, y_train)  #MACHINE AKAN LEARNING DISINI
clf


# In[ ]:


hasil_prediksi = clf.predict(x_test)
hasil_prediksi


# In[ ]:


y_test.ravel()


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test, hasil_prediksi)

