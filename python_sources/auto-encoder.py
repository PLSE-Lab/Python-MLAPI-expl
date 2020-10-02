#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ##  Binary Classification using Perception, MLP and Autoencoder

# #### Loading the required libraries

# In[ ]:


import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential, Model
from keras.layers import Dense, Input


# #### Problem
# 
#     Is to identify products at risk of backorder before the event occurs so the business has time to react. 

# #### Data
# 
# Data file contains the historical data for the 8 weeks prior to the week we are trying to predict. The data was taken as weekly snapshots at the start of each week. Columns are defined as follows:
# 
#     sku - Random ID for the product
# 
#     national_inv - Current inventory level for the part
# 
#     lead_time - Transit time for product (if available)
# 
#     in_transit_qty - Amount of product in transit from source
# 
#     forecast_3_month - Forecast sales for the next 3 months
# 
#     forecast_6_month - Forecast sales for the next 6 months
# 
#     forecast_9_month - Forecast sales for the next 9 months
# 
#     sales_1_month - Sales quantity for the prior 1 month time period
# 
#     sales_3_month - Sales quantity for the prior 3 month time period
# 
#     sales_6_month - Sales quantity for the prior 6 month time period
# 
#     sales_9_month - Sales quantity for the prior 9 month time period
# 
#     min_bank - Minimum recommend amount to stock
# 
#     potential_issue - Source issue for part identified
# 
#     pieces_past_due - Parts overdue from source
# 
#     perf_6_month_avg - Source performance for prior 6 month period
# 
#     perf_12_month_avg - Source performance for prior 12 month period
# 
#     local_bo_qty - Amount of stock orders overdue
# 
#     deck_risk - Part risk flag
# 
#     oe_constraint - Part risk flag
# 
#     ppap_risk - Part risk flag
# 
#     stop_auto_buy - Part risk flag
# 
#     rev_stop - Part risk flag
# 
#     went_on_backorder - Product actually went on backorder. This is the target value.

# #### Identify Right Error Metrics
# 
#     Based on the businees have to identify right error metrics.

# #### Loading the data

# In[ ]:


PATH = os.getcwd()


# In[ ]:


os.chdir(PATH)


# In[ ]:


data = pd.read_csv("../input/BackOrders.csv",header=0)


# #### Understand the Data

# See the No. row and columns

# In[ ]:


data.shape


# Display the columns

# In[ ]:


data.columns


# Display the index

# In[ ]:


data.index


# See the top rows of the data

# In[ ]:


data.head(3)


# Shows a quick statistic summary of your data using describe

# In[ ]:


data.describe(include='all')


# Display data type of each variable

# In[ ]:


data.dtypes


# #### Observations
# 
#     sku is Categorical but is interpreted as int64 
#     potential_issue, deck_risk, oe_constraint, ppap_risk, stop_auto_buy, rev_stop, and went_on_backorder are also categorical but is interpreted as object. 

# #### Convert all the attributes to appropriate type

# Data type conversion
# 
#     Using astype('category') to convert potential_issue, deck_risk, oe_constraint, ppap_risk, stop_auto_buy, rev_stop, and went_on_backorder attributes to categorical attributes.
# 

# In[ ]:


for col in ['sku', 'potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop', 'went_on_backorder']:
    data[col] = data[col].astype('category')


# Display data type of each variable

# In[ ]:


data.dtypes


# #### Delete sku attribute

# In[ ]:


np.size(np.unique(data.sku, return_counts=True)[0])


# In[ ]:


data.drop('sku', axis=1, inplace=True)


# #### Missing Data
# 
#     Missing value analysis and dropping the records with missing values

# In[ ]:


data.isnull().sum()


# Observing the number of records before and after missing value records removal

# In[ ]:


print (data.shape)


# Since the number of missing values is about 5%. For initial analysis we ignore all these records

# In[ ]:


data = data.dropna(axis=0)


# In[ ]:


data.isnull().sum()
print(data.shape)


# #### Converting Categorical to Numeric
# 
# For some of the models all the independent attribute should be of type numeric and Linear Regression model is one among them.
# But this data set has some categorial attributes.
# 
# 'pandas.get_dummies' To convert convert categorical variable into dummy/indicator variables
# 

# In[ ]:


print (data.columns)


# Creating dummy variables.
# 
#     If we have k levels in a category, then we create k-1 dummy variables as the last one would be redundant. So we use the parameter drop_first in pd.get_dummies function that drops the first level in each of the category
# 

# In[ ]:


categorical_Attributes = data.select_dtypes(include=['category']).columns


# In[ ]:


data = pd.get_dummies(columns=categorical_Attributes, data=data, prefix=categorical_Attributes, prefix_sep="_",drop_first=True)


# In[ ]:


print (data.columns, data.shape)


# #### Target attribute distribution

# In[ ]:


pd.value_counts(data['went_on_backorder_Yes'].values)


# #### Split the data in to train and test
# 
# sklearn.model_selection.train_test_split
# 
#     Split arrays or matrices into random train and test subsets

# In[ ]:


#Performing train test split on the data
X, y = data.loc[:,data.columns!='went_on_backorder_Yes'].values, data.loc[:,'went_on_backorder_Yes'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# In[ ]:


#To get the distribution in the target in train and test
print(pd.value_counts(y_train))
print(pd.value_counts(y_test))


# #### Perceptron Model Building

# In[ ]:


perceptron_model = Sequential()

perceptron_model.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid', kernel_initializer='normal'))


# In[ ]:


perceptron_model.compile(loss='binary_crossentropy', optimizer='adam')


# In[ ]:


perceptron_model.fit(X_train, y_train, epochs=100)


# #### Predictions

# In[ ]:


test_pred=perceptron_model.predict_classes(X_test)
train_pred=perceptron_model.predict_classes(X_train)


# #### Getting evaluation metrics and evaluating model performance

# In[ ]:


confusion_matrix_test = confusion_matrix(y_test, test_pred)
confusion_matrix_train = confusion_matrix(y_train, train_pred)

print(confusion_matrix_train)
print(confusion_matrix_test)


# #### Calculate Accuracy, True Positive Rate and True Negative Rates

# In[ ]:


Accuracy_Train = (confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
TNR_Train = confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])
TPR_Train = confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train TNR: ",TNR_Train)
print("Train TPR: ",TPR_Train)
print("Train Accuracy: ",Accuracy_Train)


# In[ ]:


Accuracy_Test = (confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]+confusion_matrix_test[1,0]+confusion_matrix_test[1,1])
TNR_Test = confusion_matrix_test[0,0]/(confusion_matrix_test[0,0] +confusion_matrix_test[0,1])
TPR_Test = confusion_matrix_test[1,1]/(confusion_matrix_test[1,0] +confusion_matrix_test[1,1])

print("Test TNR: ",TNR_Test)
print("Test TPR: ",TPR_Test)
print("Test Accuracy: ",Accuracy_Test)


# #### Derive new non-linear features using autoencoder

# In[ ]:


# The size of encoded and actual representations
encoding_dim = 18
actual_dim = X_train.shape[1]


# In[ ]:


X_train.shape


# In[ ]:


# Input placeholder
input_attrs = Input(shape=(actual_dim,))

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_attrs)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(actual_dim, activation='sigmoid')(encoded)


# In[ ]:


# this model maps an input to its reconstruction
autoencoder = Model(input_attrs, decoded)


# In[ ]:


print(autoencoder.summary())


# In[ ]:


autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# In[ ]:


autoencoder.fit(X_train, X_train, epochs=10)


# Create a separate encoder model

# In[ ]:


# this model maps an input to its encoded representation
encoder = Model(input_attrs, encoded)


# In[ ]:


print(encoder.summary())


# #### derive new non-linear features

# In[ ]:


X_train_nonLinear_features = encoder.predict(X_train)
X_test_nonLinear_features = encoder.predict(X_test)


# #### Combining new non-linear features to X_train and X_test respectively

# In[ ]:


X_train = np.concatenate((X_train, X_train_nonLinear_features), axis=1)
X_test = np.concatenate((X_test, X_test_nonLinear_features), axis=1)


# In[ ]:


X_test.shape


# #### Perceptron Model Building with both actual and non-linear features

# In[ ]:


perceptron_model = Sequential()

perceptron_model.add(Dense(1, input_dim=X_train_nonLinear_features.shape[1], activation='sigmoid'))


# In[ ]:


perceptron_model.compile(loss='binary_crossentropy', optimizer='adam')


# In[ ]:


get_ipython().run_line_magic('time', 'perceptron_model.fit(X_train, y_train, epochs=30)')


# #### Predictions

# In[ ]:


test_pred=perceptron_model.predict_classes(X_test_nonLinear_features)
train_pred=perceptron_model.predict_classes(X_train_nonLinear_features)


# #### Getting evaluation metrics and evaluating model performance

# In[ ]:


confusion_matrix_test = confusion_matrix(y_test, test_pred)
confusion_matrix_train = confusion_matrix(y_train, train_pred)

print(confusion_matrix_train)
print(confusion_matrix_test)


# #### Calculate Accuracy, True Positive Rate and True Negative Rates

# In[ ]:


Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])
TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train TNR: ",TNR_Train)
print("Train TPR: ",TPR_Train)
print("Train Accuracy: ",Accuracy_Train)


# In[ ]:


Accuracy_Test=(confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]+confusion_matrix_test[1,0]+confusion_matrix_test[1,1])
TNR_Test= confusion_matrix_test[0,0]/(confusion_matrix_test[0,0] +confusion_matrix_test[0,1])
TPR_Test= confusion_matrix_test[1,1]/(confusion_matrix_test[1,0] +confusion_matrix_test[1,1])

print("Test TNR: ",TNR_Test)
print("Test TPR: ",TPR_Test)
print("Test Accuracy: ",Accuracy_Test)


# Test TNR:  0.9836749116607774
# Test TPR:  0.2958257713248639
# Test Accuracy:  0.8534028414298809

# Test TNR:  0.9496819787985866
# Test TPR:  0.5187537810042347
# Test Accuracy:  0.8680682859761687

# 14 dimensions Auto Encoder
# 
# 
# Test TNR:  0.9351943462897526
# Test TPR:  0.6657592256503327
# Test Accuracy:  0.8841659028414299

# 18 dimensions Auto Encoder
# 
# Test TNR:  0.9385159010600707
# Test TPR:  0.6454930429522081
# Test Accuracy:  0.8830201649862511
