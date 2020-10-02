#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # KNN Algorithm
# K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions). KNN has been used in statistical estimation and pattern recognition already in the beginning of 1970's as a non-parametric technique.

# In[ ]:


from IPython.display import Image
Image(url="https://www.kdnuggets.com/wp-content/uploads/rapidminer-knn-image1.jpg")


# In[ ]:


df = pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
df


# In[ ]:


df.info()


# In[ ]:


df.describe().T


# First of all, we need to represent the class line with 1-0. For this, I assigned 0 to normal data and 1 to abnormal data.

# In[ ]:


df["class"] = [ 1 if each == "Abnormal" else 0 for each in df["class"]]
df


# In[ ]:


abnormal = df[df["class"] == 1]
normal = df[df["class"] == 0]

# scatter plot
plt.scatter(abnormal.sacral_slope,abnormal.pelvic_radius,color="red",label="Abnormal",alpha=0.5)
plt.scatter(normal.sacral_slope,normal.pelvic_radius,color="blue",label="Normal",alpha=0.5)
plt.xlabel("sacral_slope")
plt.ylabel("pelvic_radius")
plt.show()


# 1- Choose K value <br>
# 2- Find the nearest data points in K <br>
# 3- Calculate how many of the class nearest neighbors in K <br>
# 4- Determine which class of point or data we tested belongs to

# In[ ]:


y = df["class"].values
y


# In[ ]:


x_data = df.drop(["class"],axis=1) # axis=1 for columns
x_data


# # Normalization

# In[ ]:


# normalization: I represent the data to a value between 0 and 1 for more accurate processing.
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))
x


# # Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)


# # KNN Model

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) # n_neighbors => key value
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)


# In[ ]:


prediction


# In[ ]:


print("{} nn score: {} ".format(3,knn.score(x_test,y_test)*100))


# I said an estimated 3. Here, a method can be used to find the n_neighbors value.

# In[ ]:


# find k value
score_list = []

for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()


# As can be seen in the graph, the most accurate value is the number 13. So we write the value 13 in the n_neighbors section.

# In[ ]:


print("{} KNN score: {} ".format(13,knn.score(x_test,y_test)*100))

