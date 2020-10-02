#!/usr/bin/env python
# coding: utf-8

# *K-NN Algorithm Simple Example*

# KNN is a classification algorithm. In other hand it called K Nearest Neighbors. It is supervised algorithm and useful technique.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/breastCancer.csv")


# In[ ]:


df.head()


# In[ ]:


df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)


# In[ ]:


df.head()


# In[ ]:


M = df[df.diagnosis =="M"]
B = df[df.diagnosis =="B"]


# In[ ]:


plt.scatter(M.radius_mean, M.area_mean, color ="red")


# In[ ]:


plt.scatter(B.radius_mean, B.area_mean, color="green")


# In[ ]:


plt.scatter(M.radius_mean, M.area_mean, color ="red", alpha=.15, label="kotu")
plt.scatter(B.radius_mean, B.area_mean, color="green", alpha=.15, label="iyi")
plt.legend()
plt.show()


# In[ ]:


plt.scatter(M.radius_mean, M.texture_mean, color = "red", label= "kotu", alpha=.3)
plt.scatter(B.radius_mean, B.texture_mean, color = "green", label = "iyi", alpha=.3)
plt.xlabel("radius mean")
plt.ylabel("texture mean")
plt.legend()
plt.show()


# In[ ]:


df.diagnosis = [1 if each =="M" else 0 for each in df.diagnosis]


# In[ ]:


y = df.diagnosis.values
x_data = df.drop(["diagnosis"], axis=1)


# In[ ]:


x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3, random_state=1)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


model = KNeighborsClassifier(n_neighbors = 3)


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


prediction = model.predict(x_test)


# In[ ]:


prediction


# In[ ]:


print("{} nn score : {}".format(3, model.score(x_test,y_test)))


# In[ ]:


score_list = []
for each in range(1,50):
    model2 = KNeighborsClassifier(n_neighbors=each)
    model2.fit(x_train,y_train)
    score_list.append(model2.score(x_test, y_test))
    print("{} NN score : {}".format(each, model2.score(x_test,y_test)))


# In[ ]:


plt.plot(range(1,50), score_list)
plt.xlabel("k values")
plt.ylabel("score metrics")
plt.legend()
plt.show()

