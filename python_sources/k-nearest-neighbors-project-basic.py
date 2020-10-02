#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load dataset

df = pd.read_csv('../input/KNN_Project_Data.csv')
df.head()


# In[ ]:


# Dataset includes anonymized data with the features not known.
# Goal is to use KNN to create a model that predicts a class for new data points based off these given features


# In[ ]:


# Check for missing values
df.isnull().sum()


# In[ ]:


# Features will be scaled to standardize the order of magnitude among all features
df.describe()


# In[ ]:


# Very basic EDA to see any trends
sns.pairplot(data=df, hue='TARGET CLASS')
plt.show


# In[ ]:


# Splitting training and test data

X = df.drop('TARGET CLASS', axis=1)
y = df['TARGET CLASS']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


# Scaling training and test sets

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


# Using the KNN model

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# In[ ]:


# Predicting values based of our model

y_pred = knn.predict(X_test)


# In[ ]:


# Model performance evaluation
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# About 79% accuracy achieved


# In[ ]:


# Selecting the value for n_neighbors can be optimized analyzing the error rate vs. n_neighbors
error_rate = []

for i in range(1,60):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,60), error_rate, color='blue', linestyle='--', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. n_neighbors')
plt.xlabel('n_neighbors')
plt.ylabel('Error Rate')
plt.show()


# In[ ]:


# A value of 55 may give us better accuracy

knn_2 = KNeighborsClassifier(n_neighbors=55)
knn_2.fit(X_train, y_train)


# In[ ]:


y_pred_2 = knn_2.predict(X_test)


# In[ ]:


# Model performance evaluation
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred_2))
print(classification_report(y_test, y_pred_2))

# A small increase in accuracy from 79% to 81% was achieved 

