#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# We are using the iris dataset, the goal is to predict catogory from padal width and length data.

# In[44]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

iris = pd.read_csv("../input/Iris.csv")


# In[7]:


# Here is a peak of the data
iris.head()


# In[8]:


X = iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = iris["Species"]


# ## Seperate test and trail data

# In[11]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)


# In[24]:


X_train.shape,X_test.shape


# ## Fit Model

# In[27]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train) 


# In[28]:


model.score(X_test, y_test)


# ## Overfitting and underfitting

# In[36]:


neighbors = np.arange(1,9) # array([1, 2, 3, 4, 5, 6, 7, 8])
train_accuracy = np.empty(len(neighbors)) # array([......])
test_accuracy = np.empty(len(neighbors)) # array([......])


# In[40]:


for i, n in enumerate(neighbors):
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_train, y_train) 
    train_accuracy[i] = model.score(X_train, y_train)
    test_accuracy[i] = model.score(X_test, y_test)


# In[42]:


train_accuracy, test_accuracy


# In[45]:


plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# ## Visualize Boudaries

# In[ ]:




