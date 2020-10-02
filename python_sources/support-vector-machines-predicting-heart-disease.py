#!/usr/bin/env python
# coding: utf-8

# **Problem Description**
# * Predict the Heart Disease based on given attributes
# * * 0 - NO HEART DISEASE
# * * 1 - HEART DISEASE

# **Attribute Information: **
# > 1. age 
# > 2. sex (1 = male; 0 = female) 
# > 3. chest pain type (4 values) 
# Value 1: typical angina 
# Value 2: atypical angina 
# Value 3: non-anginal pain 
# Value 4: asymptomatic 
# > 4. resting blood pressure 
# > 5. serum cholestoral in mg/dl 
# > 6. fasting blood sugar > 120 mg/dl
# > 7. resting electrocardiographic results (values 0,1,2)
# -- Value 0: normal 
# -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) 
# -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria 
# > 8. maximum heart rate achieved 
# > 9. exercise induced angina 
# > 10. oldpeak = ST depression induced by exercise relative to rest 
# > 11. the slope of the peak exercise ST segment 
# > 12. number of major vessels (0-3) colored by flourosopy 
# > 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

# In[43]:


#Import python packages
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import svm #Import svm model
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix


# In[44]:


#Import the heart data
data = pd.read_csv("../input/heart.csv")


# In[45]:


#Display first 5 lines of heart data
data.head()


# In[46]:


#Display basic statistics of data
data.describe()


# In[47]:


#Display basic info about the data
data.info()


# In[48]:


#Separate Feature and Target Matrix
x = data.drop('target',axis = 1) 
y = data.target


# In[49]:


# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109) # 70% training and 30% test


# **Training and Testing the Machine Learning Model - SVM (Support Vector Machines)**

# In[50]:


#Create a svm Classifier
ml = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
ml.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = ml.predict(x_test)


# In[51]:


# Model Accuracy: how often is the classifier correct?
ml.score(x_test,y_test)


# Find the results [TP   FP
# 
#                   FN  TN ]

# In[52]:


confusion_matrix(y_test,y_pred)


# **COMMENTS:** 
# * 35 patients were predicted that they **will** have Heart Disease,the Prediction was CORRECT (True-Positive)
# * 47 patients were predicted that they **will NOT** have Heart Disease,the Prediction was CORRECT (True-Negative)
# * 5 patients were predicted that they **will** have Heart Disease but the Prediction was WRONG (False-Positive)
# * 4 patients were predicted that they **will NOT** have Heart Disease but the Prediction was WRONG (False-Negative)

# 
