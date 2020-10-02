#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import os


# Import data

# In[ ]:


data = pd.read_csv("../input/voice.csv")


# Review the data

# In[ ]:


data.info()


# In[ ]:


data.head()


# Convert gender information to binary

# In[ ]:


data.label = [1 if each == "male" else 0 for each in data.label]

data.head() # check if binary conversion worked


# Split the gender information

# In[ ]:


gender = data.label
data.drop(["label"], axis = 1, inplace = True)


# Data normalization

# In[ ]:


data = (data-np.min(data))/(np.max(data)-np.min(data))


# In[ ]:


features = data


# Split gender and features data for training and test purposes

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, gender, test_size = 0.2, random_state = 42)


# Import Knn from sklearn library

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


K = 1 # K nearest neighbors
knn = KNeighborsClassifier(n_neighbors = K)


# In[ ]:


knn.fit(x_train, y_train)
knn.score(x_test,y_test)


# Now let's find the optimum value of K (n_neighbors) and visualize it

# In[ ]:


K_optimized = 0
max_score = 0
score = []
for K in range(1,20):
    knn = KNeighborsClassifier(n_neighbors = K)
    knn.fit(x_train, y_train)
    score.append(100*knn.score(x_test, y_test))
    if max_score < 100*knn.score(x_test, y_test):
        max_score = 100*knn.score(x_test, y_test)
        K_optimized = K
print("K_optimized is {} and the corresponding accuracy is %{}".format(K_optimized,max_score))
    


# Visualization

# In[ ]:


import matplotlib.pyplot as plt
plt.plot(range(1,20), score)
plt.xlabel("K values")
plt.ylabel("ACCURACY (%)")
plt.legend()
plt.show()


# In[ ]:


print("Accuracy of gender recognition from voice data by Knn learning is %{}".format(round(max_score,3)))

