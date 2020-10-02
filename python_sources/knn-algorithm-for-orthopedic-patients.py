#!/usr/bin/env python
# coding: utf-8

# # **KNN Algorithm for Biomechanical Features of Orthopedic Patients**
# KNN means "K Nearest Neighbour" and the algorithm is used to classify points according to class of their K nearist neighbour point.
# The algorithm selects K closest points according to Euclidean Distance formula and assigns class name according to its neighbours class names. 
# Assume we assign K as 5. Algorithm finds 5 nearest neighbours according to Euclidean Distance. If class name of one of them is A and the class name of rest of four is B, then the class name of our point is classified as B.
# 
# Let's start with importing libraries and loading our data set.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualizations

data = pd.read_csv("../input/column_2C_weka.csv")


# Explore the data set.

# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.describe()


# I changed name of class column because it gives an error, I guess kernel confuses it with the real class.

# In[ ]:


data.rename(columns={
    'class': 'symptom_class'
}, inplace=True)


# Split our data set into Abnormal and Normal and visiualize it to see the distribution.

# In[ ]:


abnormal = data[data.symptom_class == "Abnormal"]
normal = data[data.symptom_class == "Normal"]
plt.scatter(abnormal.lumbar_lordosis_angle, abnormal.degree_spondylolisthesis, color = "red",label = "Abnormal")
plt.scatter(normal.lumbar_lordosis_angle, normal.degree_spondylolisthesis, color = "green",label = "Normal")
plt.legend()
plt.xlabel("Lumbar Lordosis")
plt.ylabel("Degree Spondylolisthesis")
plt.show()


# We have 6 numerical and 1 object typed columns. All columns are full of data, no missing or null values exist. 
# sypmtom_class column has two values as Abnormal and Normal. This is the column that we want to use for classification. But we need this column as integer so I want to change its names as 0 and 1. 

# In[ ]:


data.symptom_class = [1 if each == "Abnormal" else 0 for each in data.symptom_class]


# Split our data set as x and y values. y would be syptom_class column because this column is the column that we want to use for classification. Rest of the columns would be our x.

# In[ ]:


y = data.symptom_class.values
x_ = data.drop(["symptom_class"],axis=1)


# Before applying the algorithm, I want to normalize x values. Normalization means scaling all values between 0 and 1. This is important not to ignore small values.

# In[ ]:


x = (x_ - np.min(x_))/(np.max(x_)-np.min(x_)).values


# In order to calculate the accuracy of our algorithm, we need to split it into train and test data. We will use train data to train our algorithm, and then apply it to our test data to measure the accuracy. I prefer to split the data as %80 train and %20 test, so I give 0.2 to the test_size parameter.

# In[ ]:


#Split data into Train and Test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state =42)


# Now it is time to create KNN model, train our data and test it.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) #set K neighbor as 3
knn.fit(x_train,y_train)
predicted_y = knn.predict(x_test)
print("KNN accuracy according to K=3 is :",knn.score(x_test,y_test))


# We assume K = 3 for first iteration, but actually we don't know what is the optimal K value that gives maximum accuracy. So we can write a for loop that iterates for example 25 times and gives the accuracy at each iteartion. So that we can find the optimal K value.

# In[ ]:


score_array = []
for each in range(1,25):
    knn_loop = KNeighborsClassifier(n_neighbors = each) #set K neighbor as 3
    knn_loop.fit(x_train,y_train)
    score_array.append(knn_loop.score(x_test,y_test))
    
plt.plot(range(1,25),score_array)
plt.xlabel("Range")
plt.ylabel("Score")
plt.show()


# As you can see above, if we use K = 15, then we get maximum score of %80.

# In[ ]:


knn_final = KNeighborsClassifier(n_neighbors = 15) #set K neighbor as 15
knn_final.fit(x_train,y_train)
predicted_y = knn_final.predict(x_test)
print("KNN accuracy according to K=15 is :",knn_final.score(x_test,y_test))

