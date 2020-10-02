#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# ### Bagi dataset

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


# ## Machine Learning in Python - Cara mudah banget
# 
# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier   # MACHINE YANG AKAN LEARNING
clf = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


clf = clf.fit(x_train, y_train)   # SI MACHINE LEARNING'NYA DI SINI
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


# In[ ]:




