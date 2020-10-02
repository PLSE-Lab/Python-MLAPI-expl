#!/usr/bin/env python
# coding: utf-8

# # Assignment 4
# 
# ## Churn Prediction Analysis

# In this assignment, we are going to predict customer churn for a telecom company using the K-Nearest Neighbor algorithm.

# # Setup
# 
# Import python libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')


# # Exercise 1
# 
# Load the csv file called 'Churn.csv' using pandas and assign it to a variable called df. (hint: pd.read_csv?) Look at the first 10 rows of the data set. (hint: df.head?)

# In[ ]:



df=pd.read_csv('../input/Churn_6SIJGngxq2.csv')
df.head(10)


# # Exercise 2: 
# 
# Which column indicates whether the user has churned? How many users are there in your dataset? (hint: len?)

# In[ ]:


df.iloc[:,-1]
len(df)


# # Exercise 3:
# 
# Use df.describe() to explore each column. Why is the count different for each column and not equal to 5000?

# In[ ]:


df.describe()
#Count is different as it excludes all the data entries which are empty.


# # Exercise 4: 
# 
# Fill the missing numbers with the median of the whole dataset! (hint: df.fillna?) Check to see if the counts for all columns is now 5000.

# In[ ]:


df.fillna(value=df.median(), inplace=True)
df.describe()


# # Exercise 5:
# 
# Separate the data into the features and labels. Assign the features to variable X and labels to variable y.

# In[ ]:


X=df.copy()
y=df.churned
del X['churned']
print(y)


# # Exercise 6: 
# 
# Split the data into 70% training set and 30% test set and assign them to variables named X_train, X_test, y_train, y_test (hint: train_test_split?)

# In[ ]:



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3)


# # Exercise 7: 
# 
# Create a N-Nearest Neighbors classifier of 5 neighbors. (hint: KNeighborsClassifier?)

# In[ ]:



df.fillna(value=df.median(), inplace=True)
k=KNeighborsClassifier(n_neighbors=5)
print(k)


# # Exercise 8:
# 
# Fit the model to the training set. (hint: knn.fit?)

# In[ ]:


k.fit(X, y)


# # Exercise 9:
# 
# Use the model to make a prediction on the test set. Assign the prediction to a variable named y_pred

# In[ ]:


y_pred=k.predict(X)
print(y_pred)


# # Exercise 10:
# 
# Determine how accurate your model is at making predictions. (hint: accuracy_score?)

# In[ ]:


acc=accuracy_score(y,y_pred)
print(acc)


# # Exercise 11 (Optional)
# 
# Try different number of k neighbors between 1 and 10 and see if you find a better result

# In[ ]:


for i in [1,2,3,4,5,6,7,8,9,10]:
    k=KNeighborsClassifier(n_neighbors=i)
    print(k)
    k.fit(X, y)
    y_pred=k.predict(X)
    acc=accuracy_score(y,y_pred)
    print(acc)


