#!/usr/bin/env python
# coding: utf-8

# # Mobility Analytics
# 
# With the upcoming cab aggregators and demand for mobility solutions, the past decade has seen immense growth in data collected from commercial vehicles with major contributors such as Uber, Lyft and Ola to name a few. 
# 
# There are loads of innovative data science and machine learning solutions being implemented using such data and that has led to tremendous business value for such organizations. 
# 
# **So Let's Hack the crises....**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # DATA

# In[ ]:


train = pd.read_csv("/kaggle/input/train_Wc8LBpr.csv")
test = pd.read_csv("/kaggle/input/test_VsU9xXK.csv")
sub = pd.read_csv("/kaggle/input/sample_submission_NoPBkjr.csv")


# Set Trip_ID as index because we are not using it as feature.

# In[ ]:


train = train.set_index('Trip_ID')


# In[ ]:


train.head()


# In[ ]:


# Target/label analysis
train.Surge_Pricing_Type.value_counts()


# In[ ]:


train.shape


# In[ ]:


# Set Trip_ID as index because we are not using it as feature.
test = test.set_index('Trip_ID')


# In[ ]:


test.head()


# In[ ]:


test.shape


# In[ ]:


sub.head()


# In[ ]:


sub.shape


# Check the information about dataset like data type, null values,etc.

# In[ ]:


train.info()


# In[ ]:


train['Surge_Pricing_Type'].value_counts()


# In[ ]:


test.info()


# In[ ]:


# List of columns that contains null values
cols_with_missing = [col for col in train.columns 
                                 if train[col].isnull().any()]
cols_with_missing


# In[ ]:


# for imputation of missing values
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy='mean')


# In[ ]:


#List of columns that contains category values
cat_column  = [col for col in train.columns if train[col].dtypes == 'object']
cat_column


# In[ ]:


# for imputation of category values
from category_encoders.one_hot import OneHotEncoder
target_enc = OneHotEncoder(cols=cat_column)


# In[ ]:


# Classification Model
from catboost import CatBoostClassifier
catb = CatBoostClassifier()


# In[ ]:


# Pileline where all things like imputation, category encoding and modeling is done
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
my_pipeline = Pipeline([('enc',target_enc),('imputer',my_imputer),('clf',catb)])


# In[ ]:


# Features
X = train.drop('Surge_Pricing_Type', axis=1).copy()
X.head()


# In[ ]:


# Labels
y = train['Surge_Pricing_Type'].copy()
y.head()


# In[ ]:


# Split the data into train, test sets
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,train_size=0.80, test_size=0.20,random_state = 0)


# In[ ]:


# Fit the Pipeline and predcit the validation set with checking accuracy score
from sklearn.metrics import accuracy_score
my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(val_X)
print(f'Accuracy score is {accuracy_score(val_y, predictions)}')


# In[ ]:


# prediction of test data
predict = my_pipeline.predict(test)
predict


# In[ ]:


# convert the predictions to dataframe
d = pd.DataFrame(predict)
d.shape


# In[ ]:


# covert the train id to dataframe
d1 = pd.DataFrame(test.index)
d1.shape


# In[ ]:


# concatinate the prediction corresponding to train_id
df1 = pd.concat([d,d1],axis=1)


# In[ ]:


# Give the names to columns
df1.columns = ['Surge_Pricing_Type', 'Trip_ID']


# In[ ]:


# Set index as Trip_id
df1 = df1.set_index('Trip_ID')


# In[ ]:


df1.head()


# In[ ]:


# Save the final submission file
df1.to_csv('submission.csv')


# In[ ]:




