#!/usr/bin/env python
# coding: utf-8

# ## Predicting Airfoil Self-Noise using Regression
# 
# **Overview**
# 
# The NASA data set comprises different size NACA 0012 airfoils at various wind tunnel speeds and angles of attack. The span of the airfoil and the observer position were the same in all of the experiments.
# 
# We need to predict the scaled sound pressure level in decibels. 
# 
# **Data source**
# 
# The dataset has been taken from the UCI open source repository.  [UCI dataset repos](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise) 
# 
# ** Predictors **
# *  Frequency, in Hertzs. 
# *  Angle of attack, in degrees. 
# *  Chord length, in meters. 
# *  Free-stream velocity, in meters per second. 
# *  Suction side displacement thickness, in meters. 
# 
# ** Objective **
# *  Scaled sound pressure level, in decibels
# 
# 
# 
# 
# ### 1. Importing libraries and loading data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Importing sklearn libaraies for model creation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Extracting data from the csv file
airfoil_data = pd.read_csv("../input/airfoil_self-noise.csv")


# ### 2. Exploring the data 
# 
# Examining the structure of the data and variable datatype information

# In[ ]:


airfoil_data.info()
print("="*40)
airfoil_data.head()


# In[ ]:


# statistical view on the data
airfoil_data.describe().T


# In[ ]:


# Correlation between variables and dependent variable
airfoil_data.corr()


# In[ ]:


# Checking for missing values in the dataset
airfoil_data.isnull().sum()


# There are no missing values in any of the variables.

# ### 3. Model creation

# In[ ]:


# Extract y vector from the dataframe
y = airfoil_data['Scaled sound pressure level']
X = airfoil_data.drop('Scaled sound pressure level', axis=1)

# spliting train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)


# In[ ]:


print("Size of train data : ", X_train.shape)
print("Size of test data : ", X_test.shape)


# In[ ]:


# Linear model object instantiation
lr = LinearRegression().fit(X_train, y_train)

predictions = lr.predict(X_test)


# ### Evaluating the model
#  
#  Calculating the `Mean Absoluate Error` value to evaluate the prediction

# In[ ]:


mae = mean_absolute_error(predictions, y_test)

print("Mean Absolute Error :", round(mae, 2))


# In[ ]:


pred_series = pd.Series(predictions, name="Predicted")

submission = pd.concat([airfoil_data, pred_series], axis=1)

submission.head()


# In[ ]:




