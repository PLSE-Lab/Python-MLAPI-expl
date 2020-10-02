#!/usr/bin/env python
# coding: utf-8

#  # FINDING DONORS

# In[ ]:


import os
os.listdir('../input/adult-census-income')


# In[ ]:


#import the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[ ]:


# Read the census data from github 
# github.com/sumathi16/ML_FDP_SVEC
path = '../input/adult-census-income/adult.csv'
data = pd.read_csv(path)


# In[ ]:


data.head()


# In[ ]:


# seperate the features and target
# features
X_data = data.drop('income',axis=1)
# target
y_data = data.income
# check the shape of X_data and y_data
print(X_data.shape,y_data.shape)


# #### Preprocessing the Data
# ##### Numerical data

# In[ ]:


numerical_cols=X_data.columns[X_data.dtypes!=object]
numerical_cols


# In[ ]:


# Check the distribution of data for all the columns
X_data.hist()


# In[ ]:


X_data[['capital.gain','capital.loss']].describe()


# In[ ]:


# Applying log1p for skewed columns
X_data[['capital.gain','capital.loss']] =  np.log1p(X_data[['capital.gain','capital.loss']])
X_data[['capital.gain','capital.loss']].describe()


# In[ ]:


# Observe the min,max vlaues for all columns
X_data.describe()


# **Don't run the below cell more than once**

# In[ ]:


# Applying MinMAXScaler for numerical cols
#import the scaler
from sklearn.preprocessing import MinMaxScaler
# create an instance
sc = MinMaxScaler()
# Fit the model with the data to which we need to 
# apply scaling
sc.fit(X_data[numerical_cols])
#update the values with scaled values
X_data[numerical_cols] =         sc.transform(X_data[numerical_cols])


# In[ ]:


# Observe the min,max values for all columns are 0,1
X_data.describe()


# ##### Categorical Columns

# In[ ]:


cat_cols=X_data.columns[X_data.dtypes==object]
cat_cols


# In[ ]:


# check the no of columns before applying OHE
print(X_data.shape)
# Applying one hot encoding for all categorical
# columns
X_data = pd.get_dummies(X_data)
# check the no of columns after applying OHE
print(X_data.shape)


# In[ ]:


# Observe the column names
X_data.columns


# In[ ]:


# Check the top five rows
X_data.head()


# Convert the y_data 

# In[ ]:


# check the output you shoud get 0,1's
y_data.apply(lambda x: 1 if x=='>50K' else 0)


# In[ ]:


# y_data = data.income
# Store the result in y_data
y_data=y_data.apply(lambda x: 1 if x=='>50K' else 0)
# check the no of zeros and ones in y_data
y_data.value_counts()


# ##### Split the data

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =     train_test_split(X_data,y_data,random_state=42,test_size=0.3)
print(X_train.shape)
print(y_train.shape)
X_train.head()


# In[ ]:


y_data.value_counts(normalize=True)


# In[ ]:


y_train.value_counts(normalize=True)


# In[ ]:


y_test.value_counts(normalize=True)


# ##### Applying KNN CLASSIFIER

# In[ ]:


#import the classifier
from sklearn.neighbors import KNeighborsClassifier
# create an instance
knn = KNeighborsClassifier()
# train the model
knn.fit(X_train,y_train)


# In[ ]:


# predict the outcome for test data
y_pred = knn.predict(X_test)
# import the metric
from sklearn.metrics import accuracy_score
#caluculate the test data accuracy
print('test accuracy score',      accuracy_score(y_test,y_pred))


# In[ ]:


# predict the outcome for test data
y_train_pred = knn.predict(X_train)
#caluculate the test data accuracy
print('train  accuracy score',      accuracy_score(y_train,y_train_pred))


# ##### Applying Logistic Regression Classifier

# In[ ]:


from sklearn.linear_model import LogisticRegression
Lr =  LogisticRegression()
Lr.fit(X_train,y_train)


# In[ ]:


accuracy_score(y_test,Lr.predict(X_test))


# In[ ]:


accuracy_score(y_train,Lr.predict(X_train))


# In[ ]:





# In[ ]:




