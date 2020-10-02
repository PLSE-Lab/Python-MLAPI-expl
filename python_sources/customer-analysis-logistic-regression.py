#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Telecommunication industry is getting bigger and having a huge impact on everyone's daily life. The industry is also getting very competitive. In this project we will analyze an extensive consumer data set for a telecommunication company and create a Machine Learning Algorithm by using Logistic Regression. The Business is concerned about many customers leaving the land-line business for other cable competitors. The problem in question that we are trying to solve is "Who are these customers leaving and why?" Business also thinks it is easier and less costly to keep the existing customers rather than acquiring new ones. 

# # Contents
# 
# 1- About the Dataset
# 
# 2- Data Collection and Understanding
# 
# 3- Data Wrangling and Exploration
# 
# 4- Model Selecting and Set Up
# 
# 5- Model Development
# 
# 6- Evaluation
# 
# 7- Conclusion
# 

# # About the Dataset
# 
# The data provided by the business is the historical dataset with all the customers. Each row represents one customer and we will use this dataset to predict the customer churn.
# 
# We can analyze this dataset to predict what behaviors will retain the customers and further develop customer retention focused programs. 
# 
# Data set variables that requires explanation are as follow;
# 
# * Churn: The Customers who left within the last month
# 
# * Phone, Multiple Lines, Internet, Online Security, Online backup, device protection, tech support and streaming TV and movies: These variables are the services that each customer signed up for
# 
# * Customer Account information: Shows how long the customer have been a member, contract, payment method, paperless billing, monthly charges, and total charges.
# 
# * Demographic information about the customers: Gender, Age, If they have partners or dependents.
# 
# 

# # Data Collection and Understanding

# In[ ]:


# import libraries 
import pandas as pd
import numpy as np


# In[ ]:


df=pd.read_csv('../input/ChurnData.csv')


# In[ ]:


df.head(5)


# In[ ]:


df.info()


# In[ ]:


# check to see if there are any missing data
missing_data=df.isnull()
missing_data.head(5)


# In[ ]:


for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print('')


# There are no missing data at our dataset.

# In[ ]:


df.shape


# There are 200 rows and 28 columns. We will not need all 28 columns to analyze and create our model. We can look at the correlation to see the relationship between these variables.

# # Data Wrangling and Exploration

# In[ ]:


df.corr()


# In[ ]:


# selecting the features that we can use for our model development
df = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]

# churn type is float. We need to change this to integer for our Logictic Regression
df['churn'] = df['churn'].astype('int')
df.head(5)


# In[ ]:


#lets see our column and row number now
df.shape


# # Model Selecting and Set Up
# 
# The reason we are picking logical regression as a machine learning method is that, linear regression is more for continues numbers such as predicting future house prices. Logical Regression is better to estimate class of a data point. We are trying to figure out what is the most probable class for a particular data point. 

# In[ ]:


# preprocessing and definning the X and y (Features and target)
X=np.asarray(df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]


# In[ ]:


y=np.array(df['churn'])
y[0:5]


# In[ ]:


# normalize and preprocess the dataset
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[ ]:


# split the data set for train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# # Model Development

# In[ ]:


# import required libraries and create the model with Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[ ]:


# predict using the test set
yhat = LR.predict(X_test)
yhat


# In[ ]:


# predict probability (estimates) for all classes 
# first column will be probability of class 1 and second column is probability of class 0
yhat_prob = LR.predict_proba(X_test)
yhat_prob


# # Evaluation
# 
# We are going to use the jaccard index for the accuracy of our model. 

# In[ ]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)


# # Conclusion
# 
# Based on our analysis, tenure, age, education, address, income , employment and equipment are the features that can impact of staying with the services provided or moving to another telecommunication provider. We based on model considering these variables, we split the data set to 20 / 80 , train and test data, trained the data set and used the test data for prediction. Our evaluation based on jaccard index is 75% accurate. 

# In[ ]:




