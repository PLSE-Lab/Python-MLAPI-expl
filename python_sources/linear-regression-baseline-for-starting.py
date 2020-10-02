#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Importing required libraries

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt


# For training purpose, we have three datasets available in this compitition: train.csv, building_meta.csv and weather_train.csv. "train.csv" and "building_meta.csv" are related with each other through "building_id", "building_meta.csv" and "weather_train.csv" are related with each other through "site_id".
# 
# As I don't want to make the complex model in this baseline, let's focus on just "train.csv" and "building_meta.csv", so we don't need to consider all features related to weather.

# In[ ]:


#let's load datasets

train_df = pd.read_csv("/kaggle/input/ashrae-energy-prediction/train.csv")
bilding_df = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")
test_df = pd.read_csv("/kaggle/input/ashrae-energy-prediction/test.csv")


# In[ ]:


# Joining required dataset in single dataframe

df_train_building_left = pd.merge(train_df, bilding_df, on = "building_id", how = "left")
df_test_building_left = pd.merge(test_df, bilding_df, on = "building_id", how = "left")


# The next step is to preprocess the dataframe, that includes, getting overall overview of datase, checking whether any column contain NaN (missing value), converting categorical text column to categorical int column.

# In[ ]:


df_train_building_left.head()


# By looking at first five columns, it seems that only "floor_count" column contains missing value, but it's good to check whether any other column contains missing value or not.

# In[ ]:


for column in df_train_building_left:
    print(column + "\t" + str(df_train_building_left[column].isnull().any()))


# So, now it's clear that only two columns: "year_built" and "floor_count" have missing values. In this baseline, I don't want to deal with missing values, so we will not consider these two columns in model building phase.

# In[ ]:


# converting data into training and testing part

X_train, X_test, y_train, y_test = train_test_split(df_train_building_left[["building_id", "meter", "site_id", "primary_use", "square_feet"]], df_train_building_left["meter_reading"], test_size = 0.25)


# In[ ]:


# Applying label encoding technique on "primary_use" column as it contains categorical text data

label_encoder = preprocessing.LabelEncoder()

X_train["primary_use"] = label_encoder.fit_transform(X_train["primary_use"])
X_test["primary_use"] = label_encoder.transform(X_test["primary_use"])

# Normalizing the dataset because "square_feet" is in different scale than other columns 

standard_scaler = preprocessing.StandardScaler().fit(X_train)

X_train = standard_scaler.transform(X_train)
X_test = standard_scaler.transform(X_test)


# In[ ]:


# Building a simple linear regression model on preprocessed data

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train.values)
y_pred_lr = lin_reg.predict(X_test)


# As human, we know that billing price can never be negative, but machine doesn't know this. There are high chances that linear regression predicts negative value due to Extrapolation problem. So, we need to replace all negative values with 0.

# In[ ]:


y_pred_lr[y_pred_lr < 0] = 0


# In[ ]:


# Calculating accuracy

print(np.sqrt(mean_squared_log_error( y_test, y_pred_lr )))


# We got 4.06 accuracy in here, so we might get the accuracy in the same range in actual test data. 

# In[ ]:


# Plot importance of all features based on linear regression coefficients

get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(["building_id", "meter", "site_id", "primary_use", "square_feet"], lin_reg.coef_)


# It looks like building_id and site_id are important features than other three.

# Now, we will apply the same process in actual training and testing data.

# In[ ]:


label_encoder = preprocessing.LabelEncoder()

df_train_building_left["primary_use"] = label_encoder.fit_transform(df_train_building_left["primary_use"])
df_test_building_left["primary_use"] = label_encoder.transform(df_test_building_left["primary_use"])

final_X_train = df_train_building_left[["building_id", "meter", "site_id", "primary_use", "square_feet"]]
final_X_test = df_test_building_left[["building_id", "meter", "site_id", "primary_use", "square_feet"]]

final_y_train = df_train_building_left["meter_reading"]

standard_scaler = preprocessing.StandardScaler().fit(final_X_train)
final_X_train = standard_scaler.transform(final_X_train)
final_X_test = standard_scaler.transform(final_X_test)

lin_reg = LinearRegression()
lin_reg.fit(final_X_train, final_y_train.values)
y_pred_lr = lin_reg.predict(final_X_test)


# In[ ]:


submission = pd.DataFrame({'row_id':df_test_building_left['row_id'], 'meter_reading':y_pred_lr})

submission.to_csv("ashrae_prediction.csv",index=False)

