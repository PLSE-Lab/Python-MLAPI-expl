#!/usr/bin/env python
# coding: utf-8

# **LEARNING TO APPLY GRID SEARCH CROSS VALIDATION FOR LOGISTIC REGRESSION CLASSIFICATION MODELS**
# 
# In this kernel I am using breast cancer dataset to create a logistic regression machine learning model.
# 
# To improve my model I will use grid search cross validation.
# 
# Grid search cross validation method will give me the best parameters, so I will use these parameters to improve my logistic regression model.
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# **Analysis of the Dataset:**

# In[ ]:


data = pd.read_csv('../input/Breast_cancer_data.csv')
data.head()


# In[ ]:


data.info()


# I can see there is no NaN value in this dataset. So I don't need to clean it before use.

# In[ ]:


data.describe()


# From dataset statistics I can see that later I will need to normalize values. Because the features' values range in a big scale.

# In[ ]:


data.corr()


# From the dataset correlation statistics I can easly see that 'radius', 'perimeter' and 'area' features are strongly related.

# Let's see how many of the dataset inputs are diagnosed as malignant (1) or belign (0):

# In[ ]:


data['diagnosis'].value_counts()


# 
# 
# 
# 
# 

# 
# 

# **1. First of all I will separete diagnosis feature from the dataset. Diagnosis values will be my target (y).**

# In[ ]:


y = data.diagnosis.values
x = data.drop('diagnosis', axis=1)
x.head(3)


# 
# 
# 
# 
# 
# 
# 

# **2. Now I will implement normalization process to my x values.**

# In[ ]:


x = (x-np.min(x))/(np.max(x)-np.min(x))
x.describe()


# 
# 

# **3. Divide the dataset into train and test:**

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# In[ ]:


print('x_train.shape:', x_train.shape)
print('y_train.shape:', y_train.shape)
print('x_test.shape :', x_test.shape)
print('y_test.shape :', x_test.shape)


# 
# 

# 
# 

# In this kernel my aim is to investigate two different scenerios;
# *     Scenario_1: Apply Logistic Regression Classification and examine the accuracy of the model,
# *     Scenario_2: Apply Grid Search Cross Validation and use these parameters in Logistic Regression Class. in order to improve accuracy.
# 
# 
# 

# **4. Scenerio_1 Applying Logistic Regression Classification Algorithm Directly:**

# In[ ]:


from sklearn.linear_model import LogisticRegression

# Creating the model:
lr = LogisticRegression() 

# Training the model with the training datas:
lr.fit(x_train, y_train)

print('Scenario_1 score of the logistic regression: ', lr.score(x_test, y_test))


# 
# 

# 
# 

# **5. Scenario_2 Apply Grid Search Cross Validation for Logistic Regression:**

# In[ ]:


from sklearn.model_selection import GridSearchCV

grid = {'C': np.logspace(-3,3,7), 'penalty': ['l1', 'l2']}
# C and penalty are logistic regression regularization parameters
# If C is too small model is underfitted, if C is too big model is overfitted.
# l1 and l2 are regularization loss functions (l1=lasso, l2=ridge)

# Creating the model:
lr = LogisticRegression() 

# Creating GridSearchCV model:
lr_cv = GridSearchCV(lr, grid, cv=10) # Using lr model, grid parameters and cross validation of 10 (10 times of accuracy calculation will be applied) 

# Training the model:
lr_cv.fit(x_train, y_train)

print('best paremeters for logistic regression: ', lr_cv.best_params_)
print('best score for logistic regression after grid search cv:', lr_cv.best_score_)


# After the grid search cross validation for logistic regression I found that logistic regression regularization parameters should be;
# * C = 100
# * penalty = l1
# 
# for the best scored logistic regression model.

# In[ ]:


lr_tuned = LogisticRegression(C=100.0, penalty='l1')

lr_tuned.fit(x_train, y_train)

print('Scenario_2 (tuned) logistic regression score: ', lr_tuned.score(x_test, y_test))


# 
# 

# 
# 

# **CONCLUSION:**
# 
# In order to improve our models accuracy we should apply grid search cross validation before to find the best parameters. 
# 
# Then we can use these regularization parameters to improve our logistic regression classification model. 
