#!/usr/bin/env python
# coding: utf-8

# **Multi-Label Classification**
# 
# Multi-label classification originated from the investigation of text categorisation problem, where each document may belong to several predefined topics simultaneously.
# 
# Multi-label classification of textual data is an important problem. Examples range from news articles to emails. For instance, this can be employed to find the genres that a movie belongs to, based on the summary of its plot.
# 
# In multi-label classification, the training set is composed of instances each associated with a set of labels, and the task is to predict the label sets of unseen instances through analyzing training instances with known label sets.
# 
# Difference between multi-class classification & multi-label classification is that in multi-class problems the classes are mutually exclusive, whereas for multi-label problems each label represents a different classification task, but the tasks are somehow related.
# 
# For example, multi-class classification makes the assumption that each sample is assigned to one and only one label: a fruit can be either an apple or a pear but not both at the same time. Whereas, an instance of multi-label classification can be that a text might be about any of religion, politics, finance or education at the same time or none of these.
# 
# For more information on multi label classification:
# https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff

# In[ ]:


import os
print(os.listdir("../input"))


# Adding required packages 

# In[ ]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Reading csv file

# In[ ]:


train= pd.read_csv("../input/women-health-care-requirements/train Data.csv")


# viewing data 
# column starts with
# n_ is numeric
# c_ is categorical
# o_ is ordinal

# In[ ]:


train.head()


# imputation function for integers (continues)

# In[ ]:



def impute_int(df,column):
    df[column] = df[column].fillna(df[column].mean()) 


# imputation function for categorical

# In[ ]:



def impute_categ(df, column):
    df[column] = df[column].fillna(df[column].mode().loc[0]) 


# imputation function for ordinal

# In[ ]:


def impute_ordinal(df, column):
    df[column] = df[column].fillna(df[column].mode().loc[0])


# In[ ]:


# view max of 50 record at a time
pd.set_option('display.max_rows', 50)


# missing count(percentage)

# In[ ]:


def missing_Colums_Percenatage(df):
    missing_values = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending=False) != 0]
    percentage = round((df.isnull().sum().sort_values(ascending = False)*100)/len(df),2)[round((df.isnull().sum().sort_values(ascending = False)*100)/len(df),2) != 0]
    missing_values_df = pd.DataFrame(missing_values)
    percentage_df = pd.DataFrame(percentage)
    missing_values_df.reset_index(level=0, inplace=True)
    percentage_df.reset_index(level=0, inplace=True)
    
    return pd.merge(left=missing_values_df,right= percentage_df, left_on='index', right_on='index')
    
    
missing_Colums_Percenatage(train)


# droping 13000 rows which are having max columns with null 
# impute integer
# impute category
# impute ordinal 

# In[ ]:


train = train.dropna(axis=1,thresh=13000)


# In[ ]:


impute_int(train,[col for col in train if col.startswith('n_')])


# In[ ]:


train[[col for col in train if col.startswith('c_')]]


# In[ ]:


impute_categ(train,[col for col in train if col.startswith('c_')])


# In[ ]:


impute_ordinal(train,[col for col in train if col.startswith('o_')])


# Converting C_(categorical) and o_(ordinal) data types to str (bez few columns are with int data tye)

# In[ ]:


train[[col for col in train if col.startswith('c_')]].astype(str)
train[[col for col in train if col.startswith('o_')]].astype(str)


# creating dummies
# reading lables data set which is having target variable
# merging train and label data set together by common column ID (independent variables + dependent variable)
# 

# In[ ]:


train.shape


# In[ ]:


traindf = pd.get_dummies(train)


# In[ ]:


traindf.head()


# In[ ]:


trainlabeldf= pd.read_csv("../input/women-health-care-requirements/train labels.csv")


# In[ ]:


trainlabeldf['service_a']


# In[ ]:


completedf = pd.merge(left=traindf,right=pd.DataFrame(trainlabeldf), left_on='id', right_on='id')


# In[ ]:


completedf.head()


# Building a model 
# if target variable is more than one then this type of problem comes under multi lable classification (more than one taget with yes or no)
# Based on user feedback we may choose more than 1 service or none
# 
# Spliting data into test train split
# Used a RandomForestClassifier Algorithm to predict 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


# In[ ]:


[col for col in completedf if col.startswith('service')]


# In[ ]:


X = completedf.drop([col for col in completedf if col.startswith('service')], axis=1)
X = X.drop(['id'], axis =1 )
y = completedf[[col for col in completedf if col.startswith('service')]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


regressor = RandomForestClassifier(n_estimators = 100, random_state = 0) 
  
# fit the regressor with x and y data 
clf = regressor.fit(X, y) 


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


y_pred=clf.predict(X_test)


# Finding results by using classification_report 
# Accuracy 27%
# still we can improve

# In[ ]:


from sklearn import metrics
from sklearn.metrics import classification_report
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#metrics.multilabel_confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

