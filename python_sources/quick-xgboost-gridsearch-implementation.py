#!/usr/bin/env python
# coding: utf-8

# **Quick XGBoost GridSearch Implementation**
# 
# This is a minimal and very simplistic implementation of XGBoost and GridSearchCV for someone to test on the dataset very quickly.
# Hence I have kept the text and explaination to minimum
# 
# 
# 
# Importing required packages

# In[ ]:


# Ref : https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as mtrcs
import os
import matplotlib as plt
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')


# Lets import the data and go through it

# In[ ]:


data = pd.read_csv('../input/HeartDisease.csv')


# Let us see the distribution and structure of data

# In[ ]:


#Lets see the structure of the data

print(data.info())

print(data.head())


# Num is our target variable which tells about the severity of the heart disease from a scale of 0 to 4. Let us treat this scale as classes and implement an XGBoost classifier, tuning the hyper parameters using GridSearch CV

# In[ ]:


target = data['num'].as_matrix().reshape([457,1])


# Almost all the predictors are numerical, except one: **place**
# 
# Lets convert this varable to dummy variables (one hot encoding) and add back to the dataset

# In[ ]:



dummy_cat =pd.get_dummies(data['Place'], drop_first = True) #dropping the one column of the dummy varables as it is redundant

#selecting the varables
#well, in this case all

predictor_df =data[['Age','Sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']]

predictor_df = pd.concat([predictor_df,dummy_cat],axis = 1)

print(predictor_df.head())

predictors = predictor_df.as_matrix()


# The martix looks good. Ready to be fed to the model.
# 
# But before that we need to split the data into train and test data.
# 
# Lets take 2/3rd data as training and 1/3 as test. (Incase of more data we can just take 10-15% data for test)

# In[ ]:


seed = 1
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(predictors,target, test_size=test_size, random_state=seed)


# In[ ]:


model = XGBClassifier() # declaring the XGBoost classifier


# To implement the grid search we have used the GridSearchCV o sklearn. 
# 
# The grid search does an "Exhaustive search over specified parameter values for an estimator"
# 
# Here we are specifying various learning rate and  tree depths. We can also iiterate over various models using GridSearchCV but lets limit ourselves to XGBoost.
# 
# 

# In[ ]:


grid = GridSearchCV(estimator=model,scoring = 'accuracy',param_grid={'learning_rate': np.array([0.1,0.2,0.3,0.6,0.7]),'max_depth':np.array([2,3,4,5,6])})
grid.fit(X_train, y_train)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)


# In[ ]:


print(grid.best_estimator_)


# When we do grid search over the previously specified paramets, we get the above classifier as the best one!!
# This one has been optimizd over the accuracy metric.
# 
# Now, lets predict using this classifier on our test data and store values

# In[ ]:


y_pred = grid.best_estimator_.predict(X_test)
predictions = [round(value) for value in y_pred]


# Quickly checking the test accuracy, precision and recall

# In[ ]:


accuracy = mtrcs.accuracy_score(y_test, predictions)
recall = mtrcs.recall_score(y_test,predictions)
precision = mtrcs.precision_score(y_test,predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Recall: %.2f%%" % (recall * 100.0))
print("Precision: %.2f%%" % (precision * 100.0))


# Hmm, Not Bad
