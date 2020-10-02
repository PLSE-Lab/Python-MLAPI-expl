#!/usr/bin/env python
# coding: utf-8

# **K-Nearest Neighbors** 
# 
# 1. KNN is based on the principle of how people judge
# you by observing your peers (neighbors)
# 2. Makes predictions by:
# a. Averaging - For Regression Tasks
# b. Majority Voting - For Classification Tasks
# 
# **K-Nearest Neighbors - Two Important Steps**
# 1. Choosing the right distance metric
# (e.g. for something like words, you may want to use cosine similarity because you're more interested
# in the direction of the word rather than the actual size of the values)
# 2. Choosing the value of K
# 
# **K-Nearest Neighbors - How It Works**
# 
# The algorithm uses feature "similarity" to predict new values of any new data
# point
# 1. Determine K (no. of nearest neighbors)
# 2. Calculate distance (Euclidean,Manhattan,etc.)
# 3. Determine K-minimum distance neighbors
# 4. Make Prediction
# a. For classification, the new data point is assigned to the majority class
# of the K neighbors.
# b. For regression, the new data point is assigned by the average of the K
# neighbors.

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))
get_ipython().run_line_magic('matplotlib', 'inline')

#importing train_test_split
from sklearn.model_selection import train_test_split


# In[ ]:


# Import the campaign dataset from Excel (Sheet 0 = Non Responders, Sheet 1 = Responders)
diabetes_df = pd.read_csv("../input/diabetes.csv")
diabetes_df.head()


# In[ ]:


#Examine Shape of Dataset
diabetes_df.shape


# In[ ]:


#Examine Class Distribution
diabetes_df.Outcome.value_counts() / len(diabetes_df)


# **Create Seperate Arrays for IVs and DV**

# In[ ]:


# Create array to store our features and target variable
X = diabetes_df.drop('Outcome',axis=1).values
y = diabetes_df['Outcome'].values


# **Scale the Data**
# 
# Apply Standard Scaling

# In[ ]:


# Apply Standard Scaler to our X dataset
import sklearn.preprocessing as preproc
X_scaled = preproc.StandardScaler().fit_transform(X)
X_scaled


# **Train/Test Split**

# In[ ]:


#Split our data into a train and test set
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42, stratify=y)


# **Import K-NN Classifier**
# 
# Let's apply different values of K to evaluate which value should give us the best prediction performance
# 
# We will be using 50 different values of K (1-50)

# In[ ]:


# Import KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

# Create K values (1-10) & Create Arrays to store train/test performance accuracy
k = np.arange(1,50)
train_accuracy = np.empty(len(k))
test_accuracy = np.empty(len(k))

for i,k in enumerate(k):
    # Instantiate NN Classifier with K Neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Fit KNN model
    knn.fit(X_train, y_train)
    
    # Evaluate train performance 
    train_accuracy[i] = knn.score(X_train, y_train)
    
    # Evaluate test performance
    test_accuracy[i] = knn.score(X_test, y_test)


# **Visualize the Train/Test Report**

# In[ ]:


# Visualize Train/Test Performance
k = np.arange(1,50)
plt.title('k-NN Varying number of neighbors')
plt.plot(k, test_accuracy, label='Testing Accuracy')
plt.plot(k, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('# K-Neighbors')
plt.ylabel('Accuracy')
plt.show()


# **Apply GridSearchCV**
# 
# It's hard to visually see which value of K is best for our prediction accuracy.
# 
# We'll apply GridSearchCV where:
# 
# For each value of K, we will apply 5-Fold Cross Validation to it
# Specifically:
# 
# Try different values of K
# 
# Train/Fit them all seperately
# 
# Evaluate each of their performance
# 
# Select the best score

# In[ ]:


#import GridSearchCV
from sklearn.model_selection import GridSearchCV

#In case of classifier like knn the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(1,50)}

knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X_scaled,y)


# **Examine the Best Score**

# In[ ]:


knn_cv.best_score_


# **Examinen the Best K Value**

# In[ ]:


knn_cv.best_params_

