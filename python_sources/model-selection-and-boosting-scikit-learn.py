#!/usr/bin/env python
# coding: utf-8

# Created by: Sangwook Cheon
# 
# Date: Dec 31, 2018
# 
# This is step-by-step guide to Model Selection and Boosting, which I created for reference. I added some useful notes along the way to clarify things. This notebook's content is from A-Z Datascience course, and I hope this will be useful to those who want to review materials covered, or anyone who wants to learn the basics of Model Selection and Boosting.
# 
# ## Content:
# ### 1. K-Fold Cross Validation
# ### 2. Grid Search
# ### 3. XGBoost
# --------
# These techniques above are used to improve model performance. 
# 
# # 1. K-Fold Cross Validation
# This technique is useful to evaluate bias and variance more accurately. It splits the training set into k groups, and in each iteration, the algorithm chooses different test fold (individual section) for testing. This allows every part of the training set to be used for testing.
# 
# ### Dataset overview

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../input/social-network-ads/Social_Network_Ads.csv')
dataset.head()


# Using this fictional dataset, the company wants to know whether a customer will buy its product based on age and estimated salary.
# 
# I will add the Cross Validation part to a standard Kernel SVM model.

# In[ ]:


x = dataset.iloc[:, [2,3]]
y = dataset.iloc[:, 4]

#split test and train set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_train)

#building the model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)

#predicting based on the model
y_pred = classifier.predict(X_test)


# **Now Let's apply K-Fold Cross Validation**

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
#cv --> number of parts (folds): 10 is the most common practice
#estimator --> put the trained model.
print(accuracies)
print(accuracies.mean()) #this is a better evaluation of the model.
print(accuracies.std()) #to get a sense of variance and bias


# Using the mean of all the accuracies is a better estimation of the overall performance of the mode. 
# 
# # 2. Grid Search
# This is used to effectively figure out good combination of hyperparameters. We need to examine each model's hyperparameters, and include them inside a dictionary of parameters we want to test. Using this, let's figure out what model is best for this specific dataset, and what parameters should be used!
# 
# ### Note: We can simply put the Grid Search section after K-Fold CV section.

# In[ ]:


#Applying grid search
from sklearn.model_selection import GridSearchCV
parameters = [{"C": [1, 10, 100, 1000], "kernel": ['linear']}, 
              {"C": [1, 10, 100, 1000], "kernel": ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001]}]
#we use this parameters list of dictionaries to test out many combinations of parameters and models.
#This dictionary is tailored to SVM model. For different models, different dictionary keys and values should be used.
#C: penalty parameter that reduces overfitting. Gamma: for optimal kernel

#Use this list to train
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
#scoring: how models are evaluated | cv: apply K-fold cross validation
#n_jobs: how much CPU to use. -1 --> all CPU.

grid_search = grid_search.fit(X_train, Y_train)

#Use attributes of grid_search to get the results
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print(best_accuracy)
print(best_parameters)


# # 3. XGBoost
# The most famous algorithm to boost execution speed on large datasets. 

# ### Install XGBoost
# To install XGBoost, [please refer to this official guide](https://xgboost.readthedocs.io/en/latest/build.html). But on kaggle kernel, xgboost is already installed!
# 
# ### Prepare data and Build the XGBoost model
# #### Data overview
# 

# In[ ]:


dataset2 = pd.read_csv('../input/bank-customer-churn-modeling/Churn_Modelling.csv')
dataset2.head()


# A bank wants to predict if current customers will exit or stay in the bank based on many characteristics of each customer. We need to create a machine learning model to predict this as accurately as possible.
# 
# ### Build the model

# In[ ]:


#prepare the dataset
x = dataset2.iloc[:, 3:13]
y = dataset2.iloc[:, 13]

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x.iloc[:, 1] = labelencoder_x.fit_transform(x.iloc[:, 1]) #applying on Geography

#apply encoder on Gender as well
labelencoder_x_2 = LabelEncoder()
x.iloc[:, 2] = labelencoder_x_2.fit_transform(x.iloc[:, 2]) #applying on Gender

from keras.utils import to_categorical
encoded = pd.DataFrame(to_categorical(x.iloc[:, 1]))
#no need to encode Gender, as there are only two categories

x = pd.concat([encoded, x], axis = 1)

#Dropping the existing "geography" category, and one of the onehotcoded columns.

x = x.drop(['Geography', 0], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# ************* Feature selection is not necessary in xgboost, as it is decision-tree algorithm.


# In[ ]:


#build the model
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, Y_train)


# In[ ]:


#predict the results
y_pred = classifier.predict(X_test)
y_pred[:5]


# In[ ]:


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
cm


# In[ ]:


#K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())


# I want to note that this only scratched the surface of XGBoost. To find out more about this powerful tool, such as learning about parameters, please refer to the official document (The link is above. Just navigate to the homepage!) 
# 
# ----------------
# Thank you for reading this kernel. If you found this kernel useful, I would really appreciate if you upvote it or leave a short comment below.
