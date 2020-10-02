#!/usr/bin/env python
# coding: utf-8

# Importing required dependencies

# In[55]:


import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Reading in the data

# In[56]:


heart = pd.read_csv("../input/heart.csv")


# First five rows of the data

# In[57]:


heart.head()


# Dataset stucture

# In[58]:


heart.shape


# Renaming the columns to more meaningful names

# In[59]:


heart.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar',
                 'rest_ecg', 'max_heart_rate_achieved','exercise_induced_angina', 'st_depression', 'slope',
                 'num_major_vessels', 'thalassemia', 'target']


# Checking for missing values

# In[60]:


heart.isnull().sum()


# The dataset is clean and has no missing values

# Checking for data types of the dataset columns

# In[61]:


heart.dtypes


# Observing the target samples

# In[62]:


heart['target'].value_counts().plot(kind = 'bar', color = ['b','r'])


# The target is quite balanced.

# Checking for some categorical columns so as to make sure that they are converted into the correct data type before preprocessing of the model

# In[64]:


print("sex : {}".format(heart['sex'].nunique()))

print("chest_pain_type : {}".format(heart['chest_pain_type'].nunique()))

print("fasting_blood_sugar : {}".format(heart['fasting_blood_sugar'].nunique()))

print("rest_ecg : {}".format(heart['rest_ecg'].nunique()))

print("exercise_induced_angina : {}".format(heart['exercise_induced_angina'].nunique()))

print("slope : {}".format(heart['slope'].nunique()))

print("num_major_vessels : {}".format(heart['num_major_vessels'].nunique()))

print("thalassemia : {}".format(heart['thalassemia'].nunique()))


# All of the above are categorical columns and thus we need to change their data type from integer format to categorical

# In[65]:


for col in ['sex','chest_pain_type','rest_ecg','exercise_induced_angina','fasting_blood_sugar','slope',
            'num_major_vessels','thalassemia']:
    heart[col] = heart[col].astype('category')


# Checking if the columns are converted to categorical type

# In[66]:


heart.dtypes


# Converting labels in the categorical columns to numerical form using Label Encoder so as to ready the data to use in machine learning model

# In[67]:


cols = ['sex','chest_pain_type','rest_ecg','exercise_induced_angina','fasting_blood_sugar','slope',
            'num_major_vessels','thalassemia', 'target']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for c in cols:
    le.fit(list(heart[c].values))
    heart[c] = le.transform(list(heart[c].values)) 
    


# Getting dummy variables for all the columns in the entire data set

# In[68]:


heart = pd.get_dummies(heart, drop_first = True)


# Splitting the data into train and test sets

# In[70]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(heart.drop('target', 1), heart['target'],
                                                    test_size = .3, random_state=100) 


# **Random Forest Classifier(RFC)**

# Fitting the RFC on the training set 

# In[71]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=5)
model.fit(X_train, y_train)


# Predicting the target values on the test set using RFC

# In[72]:


y_pred = model.predict(X_test)


# Checking accuracy of our predictions using the RFC 

# In[73]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy


# Confusion matrix for the classified variables on the test set using RFC

# In[74]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
confusion_matrix


# 37 out of 46 positives classified correctly yielding Sensitivity of 0.804 or true positive rate of 80.4%
# 
# 39 out of 45 negatives classified correctly yielding Specificity of 0.866 or true negative rate of 86.6%

# **Decision Tree Classifier**

# Fitting the Decision Tree Model on the training set

# In[75]:


from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier(max_depth = 5, max_features = 4)
dec_tree.fit(X_train, y_train)


# Predicting the target values on the test using Decision Tree Model

# In[76]:


dec_tree_pred = dec_tree.predict(X_test)


# Checking for accuracy

# In[77]:


dec_tree_accuracy = accuracy_score(y_test,dec_tree_pred)
dec_tree_accuracy


# The tree Classifier and the ensemble classifier yielded **>80%** test accuracy. 
# 
# Therefore, let us make predictions using a comparatively weaker classifier KNN

# **K-Nearest Neighbor (KNN)**

# Fitting the KNN model to training data

# In[78]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights = 'distance', leaf_size = 20)
knn.fit(X_train, y_train)


# Classifying target variables using KNN

# In[79]:


knn_pred = knn.predict(X_test)


# Checking for accuracy 

# In[80]:


knn_accuracy = accuracy_score(y_test,knn_pred)
knn_accuracy


# Confusion Matrix for KNN

# In[81]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,knn_pred)
confusion_matrix


# 25 out of 46 positives classified correctly yielding Sensitivity of 0.543 or true positive rate of 54.35%
# 
# 35 out of 45 negatives classified correctly yielding Specificity of 0.777 or true negative rate of 77.7%
