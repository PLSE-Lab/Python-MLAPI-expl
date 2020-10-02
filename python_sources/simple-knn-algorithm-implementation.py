#!/usr/bin/env python
# coding: utf-8

# # KNN ALGORITHM

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


data = pd.read_csv("../input/breastCancer.csv")


# In[4]:


data.tail()


# In[5]:


data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)


# In[6]:


data.head()


# In[7]:


M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]


# In[8]:


plt.scatter(M.radius_mean, M.texture_mean, color="red", label="kotu",alpha=.29)
plt.scatter(B.radius_mean, B.texture_mean, color="green", label = "iyi", alpha=.29)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()


# In[9]:


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)


# In[10]:


y


# In[11]:


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[12]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.3, random_state=1)


# In[13]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)


# In[14]:


prediction


# In[15]:


print("{} - nn score: {}".format(3, knn.score(x_test,y_test)))


# In[17]:


score_list = []
for each in range(1,15):
    knns = KNeighborsClassifier(n_neighbors = each)
    knns.fit(x_train,y_train)
    score_list.append(knns.score(x_test,y_test))

plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()


# In[ ]:




