#!/usr/bin/env python
# coding: utf-8

# # Introduction to the Notebook.
# Iris dataset - It is the "Hello World" of Machine Learning Classification problems. Every aspiring coder once in his life have had an date with this dataset. Its manageble, its simple to comprehend and its clean. So you can directly jump into it - So we will.
# 
# Target of this notebook is to use this dataset and lay down ground work for all useful classifiers (atleast that i know of, will keep adding as and when i find a new one, or you guys comment it down for me).
# Then we will calculate accuracy of these models against a "never seen before test data". We learn the concept of cross validation while doing it. So grab those headphones and sit tight!!!

# ## The Dataset - Iris.

# In[ ]:


# So let us start by loading the dataset. Now you can use the one from kaggle or you can use the one by sklearn.
# Let us use the sklearn iris dataset.

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import pandas as pd


# In[ ]:


# Load the iris dataset into a dataframe.
df = load_iris()

# Load the independent features and the target variable. (y is the target variable, we will predict this one)
x = df.data
y = df.target


# In[ ]:


# Let us see some info about the Iris dataset.
print("Shape of Independent features : ", x.shape)
print("Shape of Target feature : ", y.shape)

print("The Independent features in Iris dataset are :", df.feature_names)
print("The Target in Iris dataset is one of these :", df.target_names)


# ## Breaking down the Story of Classification.
# 
# A classification problem in machine learning is one whose target variable is basically observations belonging to a certain "class", such as - 
# * Analysis of the customer data to predict whether he will buy computer accessories (Target class: Yes or No)
# * Classifying fruits from features like color, taste, size, weight (Target classes: Apple, Orange, Cherry, Banana)
# etc.
# 
# The targets could be binary (like yes or no) or multiclass (like iris targets). 
# Here is a list of Classification algos that we will see in this notebook (to be regularly updated):
# 1. Logistic Regression
# 2. Support Vector Machines.
# 3. Naive.
# 4. K - Nearest.
# 5. Random Forest.
# 6. XGboost

# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

model_log = LogisticRegression(max_iter=1000)
print(cross_val_score(model_log, x, y, cv=4, scoring='accuracy').mean())


# ## Support Vector Machine

# In[ ]:


from sklearn.svm import SVC

model_svm = SVC()
print(cross_val_score(model_svm, x, y, cv=4, scoring='accuracy').mean())


# ## Naive

# In[ ]:


from sklearn.naive_bayes import GaussianNB

model_naive = GaussianNB()
print(cross_val_score(model_naive, x, y, cv=4, scoring='accuracy').mean())


# ## K - Nearest Neighbours

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier()
print(cross_val_score(model_knn, x, y, cv=4, scoring='accuracy').mean())


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model_random = RandomForestClassifier()
print(cross_val_score(model_random, x, y, cv=4, scoring='accuracy').mean())


# ## XG Boost

# In[ ]:


from xgboost import XGBClassifier

model_xg = XGBClassifier()
print(cross_val_score(model_xg, x, y, cv=4, scoring='accuracy').mean())


# Source - [https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/?](http://)
