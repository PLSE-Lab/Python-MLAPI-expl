#!/usr/bin/env python
# coding: utf-8

# ### My Goals for this notebook is to build a simple baseline model for "2019 Data Science Bowl" Kaggle competition:
# 
# Detailed descripton of the problem and data can be found here https://www.kaggle.com/c/data-science-bowl-2019

# ### imports

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report, confusion_matrix
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# #### Load the data

# In[ ]:


train_df = pd.read_csv('../input/data-science-bowl-2019/train.csv')
test_df = pd.read_csv('../input/data-science-bowl-2019/test.csv')
train_labels_df = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
specs_df = pd.read_csv('../input/data-science-bowl-2019/specs.csv')


# In[ ]:


train_df.shape


# In[ ]:


train_df.head()


# In[ ]:


test_df.shape


# In[ ]:


test_df.head()


# In[ ]:


train_labels_df.shape


# In[ ]:


train_labels_df.head()


# In[ ]:


specs_df.shape


# In[ ]:


specs_df.head()


# In[ ]:


train_df.dtypes


# In[ ]:


#Lets check null values
train_df.isnull().sum()


# In[ ]:


#plt.figure(figsize=(20,29)) 
sns.countplot("event_id", data = train_df)


# In[ ]:


train_df['event_id'].value_counts()


# * > Join Training dataset with labels dataset

# In[ ]:


training_data_with_label = pd.merge(train_df, train_labels_df, how='inner')


# In[ ]:


training_data_with_label.head()


# In[ ]:


training_data_with_label.shape


# ![](http://)Lets prepare few aggerated and date features 

# In[ ]:


def prepare_agg_and_date_features(input_df, input_columns):
    for column in input_columns:
        input_df[column+"_total"]= input_df.groupby(column)[column].transform('count')
        input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
        input_df['year'] = input_df['timestamp'].dt.year
        input_df['quarter_of_year'] = input_df['timestamp'].dt.quarter
        input_df['month_of_year'] = input_df['timestamp'].dt.month
        input_df['day_of_month'] = input_df['timestamp'].dt.day
        input_df['hour_of_day'] = input_df['timestamp'].dt.hour
        input_df['minute_of_hour '] = input_df['timestamp'].dt.minute 
    return input_df


# In[ ]:


input_agg_columns = ['event_id', 'game_session', 'installation_id','title','type','world']


# In[ ]:


#Derive features from training dataset
training_features = prepare_agg_and_date_features(training_data_with_label, input_agg_columns)


# In[ ]:


training_features.shape


# In[ ]:


training_features.head()


# In[ ]:


#Derive features from test dataset
test_features = prepare_agg_and_date_features(test_df, input_agg_columns)


# In[ ]:


test_features.head()


# In[ ]:


test_features.shape


# In[ ]:


# remove Object type feaures
required_training_features = training_features.select_dtypes(exclude  = object)


# Prepare dataset for modeling

# In[ ]:


X = required_training_features.drop(["timestamp","accuracy","accuracy_group","num_correct", "num_incorrect"], axis=1)


# In[ ]:


y = required_training_features['accuracy_group']


# In[ ]:


X.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[ ]:


#training the model
lr_clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)


# In[ ]:


#get prediction 
lr_prediction = lr_clf.predict(X_test)


# In[ ]:


confusion_matrix(y_test, lr_prediction)


# In[ ]:


display(classification_report(y_test, lr_prediction))


# In[ ]:


cohen_kappa_score(y_test, lr_prediction)


# Get the prediction on the test dataset

# In[ ]:


exclude_columns = ["event_id","game_session","event_data","title","type","world","timestamp"]


# In[ ]:


required_testing_features = test_df.drop(exclude_columns,axis=1)


# In[ ]:


required_testing_features.head()


# In[ ]:


required_testing_features.shape


# In[ ]:


lr_test_data_prediction = lr_clf.predict(required_testing_features.drop('installation_id',axis=1))


# In[ ]:


required_testing_features['accuracy_group'] = lr_test_data_prediction


# In[ ]:


submission_df = required_testing_features[['installation_id','accuracy_group']]


# In[ ]:


submission_df.head()


# Take mean aggregate prediction of each installation_id

# In[ ]:


submission_df_group_by = pd.DataFrame(submission_df.groupby(['installation_id'])['accuracy_group'].mean())
submission_df_group_by = submission_df_group_by.round().astype(int)
submission_df_group_by.head(10)


# In[ ]:


submission_df_group_by = submission_df_group_by.reset_index()


# In[ ]:


submission_df_group_by.head()


# > ### Preparing submission file**

# In[ ]:


submission_df_group_by.to_csv("submission.csv",index=False)


# ![](http://)<a href="./ladle_submission.csv"> Download File </a>
