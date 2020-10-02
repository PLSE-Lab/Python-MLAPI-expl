#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data = pd.read_csv('../input/data.csv')
import warnings
warnings.filterwarnings('ignore')
# Any results you write to the current directory are saved as output.


# In[ ]:


data.head()


# Column **id **and **Unnamed: 32** aren't needed for machine learning so, we have to drop it first

# In[ ]:


df = data.drop(columns=['id','Unnamed: 32'])
print("Dataset size : ",df.shape)


# ## Explore the values
# 

# In[ ]:


df.describe()


# In[ ]:


df.hist(figsize=(20,30),bins=50,xlabelsize=8,ylabelsize=8);


# # Training Dataset Preparation
# 
# Since most of the Algorithm machine learning only accept array like as input, so we need to create an array from dataframe set to X and y
# array before running machine learning algorithm.<br>
# the dataset is splitted by X the parameter and y for classification labels.

# In[ ]:


X=np.array(df.drop(columns=['diagnosis']))
y=df['diagnosis'].values
print ("X dataset shape : ",X.shape)
print ("y dataset shape : ",y.shape)


# # Machine Learning Model
# 
# ## Import Machine Learning Library from Scikit-Learn
# 
# Machine learning model used is Classification model, since the purpose of this Study case is to classify diagnosis between "Malignant"
# (M) Breast Cancer and "Benign" (B) Breast Cancer, we will be using 5 model of classification algorithm. <br>
# - Model 1 : Using Simple Logistic Regression
# - Model 2 : Using Support Vector Classifier
# - Model 3 : Using Decision Tree Classifier
# - Model 4 : Using Random Forest Classifier
# - Model 5 : Using Gradient Boosting Classifie

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


model_1 = LogisticRegression()
model_2 = SVC()
model_3 = DecisionTreeClassifier()
model_4 = RandomForestClassifier()
model_5 = GradientBoostingClassifier()


# # Model Fitting
# 
# Since we need to fit the dataset into algorithm, so proper spliting dataset into training set and test set are required
# 
# ## Method 1. Train test split
# 
# Using Scikit learn built in tools to split data into training set and test set to check the result score of the model
# train_test_split configuration using 20% data to test and 80& data to train the model, random_state generator is 45

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
print ("Train size : ",X_train.shape)
print ("Test size : ",X_test.shape)


# ## Fitting Training set into models

# In[ ]:


model_1.fit(X_train,y_train)
model_2.fit(X_train,y_train)
model_3.fit(X_train,y_train)
model_4.fit(X_train,y_train)
model_5.fit(X_train,y_train)


# ## Predict and show Score and F1 Score prediction using test data

# In[ ]:


# Predict data
y_pred1=model_1.predict(X_test)
y_pred2=model_2.predict(X_test)
y_pred3=model_3.predict(X_test)
y_pred4=model_4.predict(X_test)
y_pred5=model_5.predict(X_test)
#Show F1 Score
from sklearn.metrics import f1_score
f1_model1=f1_score(y_test,y_pred1,average='weighted',labels=np.unique(y_pred1))
f1_model2=f1_score(y_test,y_pred2,average='weighted',labels=np.unique(y_pred2))
f1_model3=f1_score(y_test,y_pred3,average='weighted',labels=np.unique(y_pred3))
f1_model4=f1_score(y_test,y_pred4,average='weighted',labels=np.unique(y_pred4))
f1_model5=f1_score(y_test,y_pred5,average='weighted',labels=np.unique(y_pred5))
print("F1 score Model 1 : ",f1_model1)
print("F1 score Model 2 : ",f1_model2)
print("F1 score Model 3 : ",f1_model3)
print("F1 score Model 4 : ",f1_model4)
print("F1 score Model 5 : ",f1_model5)


# ## Method 2. Cross validation method
# Using Cross validation will resulted in more reliability of the model. <br>
# in this case using StratifiedKFold from Scikit Learn, with n_split = 10 times and Shuffle = True.

# In[ ]:


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True)
skf.get_n_splits(X,y)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'StratifiedKFold')


# In[ ]:


# Set Container to gather the cross validation result of the model
score_list_model1,score_list_model2,score_list_model3,score_list_model4,score_list_model5 = [],[],[],[],[]


# In[ ]:


for train_index, test_index in skf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model_1.fit(X_train, y_train)
    model_2.fit(X_train, y_train)
    model_3.fit(X_train, y_train)
    model_4.fit(X_train, y_train)
    model_5.fit(X_train, y_train)
    y_pred1=model_1.predict(X_test)
    y_pred2=model_2.predict(X_test)
    y_pred3=model_3.predict(X_test)
    y_pred4=model_4.predict(X_test)
    y_pred5=model_5.predict(X_test)
    score_list_model1.append(f1_score(y_test,y_pred1,average='weighted',labels=np.unique(y_pred1)))
    score_list_model2.append(f1_score(y_test,y_pred2,average='weighted',labels=np.unique(y_pred2)))
    score_list_model3.append(f1_score(y_test,y_pred3,average='weighted',labels=np.unique(y_pred3)))
    score_list_model4.append(f1_score(y_test,y_pred4,average='weighted',labels=np.unique(y_pred4)))
    score_list_model5.append(f1_score(y_test,y_pred5,average='weighted',labels=np.unique(y_pred5)))


# In[ ]:


score_table = pd.DataFrame({"F1 Score model 1" :score_list_model1,"F1 Score model 2" :score_list_model2,"F1 Score model 3" :score_list_model3,"F1 Score model 4" :score_list_model4,"F1 Score model 5" :score_list_model5})

score_table


# In[ ]:


final_1=np.mean(score_list_model1)
final_2=np.mean(score_list_model2)
final_3=np.mean(score_list_model3)
final_4=np.mean(score_list_model4)
final_5=np.mean(score_list_model5)
print("F1 Score Average Model_1 :",final_1)
print("F1 Score Average Model_2 :",final_2)
print("F1 Score Average Model_3 :",final_3)
print("F1 Score Average Model_4 :",final_4)
print("F1 Score Average Model_5 :",final_5)


# # Conclusion
# 
# After Testing 5 Model of machine learning classifier and testing both using train test split and cross validation method, conclude that
# Model 5 which is Gradient Boosting winc with crossvalidation F1 Score 0.96

# In[ ]:




