#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# **Importing the necessary libraries**

# In[ ]:


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# *Knowing the data*

# In[ ]:


data = pd.read_csv('../input/zoo-animal-classification/zoo.csv')
data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()


# Splitting the data into two dataframes X-containing all the features and y-containing the target

# In[ ]:


y=data['class_type'].values
X=data.drop(['class_type','animal_name'],axis=1).values


# Splitting the data into train and test
# Training the model.

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)


# In[ ]:


knn.score(X_train,y_train)


# In[ ]:


knn.score(X_test,y_test)


# We obtained an accuracy of 80% using KNN classification
