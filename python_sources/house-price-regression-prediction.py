#!/usr/bin/env python
# coding: utf-8

# # House Pricing
# 
# ![](https://images.pexels.com/photos/186077/pexels-photo-186077.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940)
# 
# This notebook intends to predict the value for a house according to determined values for its features. The implemented model is multiple linear regression

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


# ## Extracting and checking the data

# In[ ]:


house_df = pd.read_csv('/kaggle/input/housepricing/HousePrices_HalfMil.csv')
house_df.head()


# In[ ]:


house_df.dtypes


# All columns are numerical, no need for categorical treatment
# 
# Checking whether the dataset has null values

# In[ ]:


house_df.isna().any()


# No null values found on any colum. Operating normally

# ## Linear regression prediction
# 

# In[ ]:


import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X = house_df.drop(['Prices'], axis = 1)
y = house_df[['Prices']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

linear_regression = sm.OLS(y_train, sm.add_constant(X_train)).fit()

y_predict = linear_regression.predict(sm.add_constant(X_test))

print('R2: ', r2_score(y_test, y_predict))


# Correlation coefficient looks promising. Checking the regression coefficients

# In[ ]:


linear_regression.params


# Attempting to predict what the cost of a house should be if it has the following values for its features:
# 
# Area: 150
# Garage: 1
# FirePlace: 0
# Baths: 3
# White Marble: 1
# Black Marble: 0
# Indian Marble: 0
# Floors: 2
# City: 3
# Solar: 0
# Electric: 1
# Fiber: 0
# Glass Doors: 0
# Swiming Pool: 0
# Garden: 1

# In[ ]:


def calculate_prediction(area, garage, fireplace, baths, white_mar, black_mar, indian_mar, floors, city, solar, electric, fiber, glass_doors, pool, garden):
    X_test = [area, garage, fireplace, baths, white_mar, black_mar, indian_mar, floors, city, solar, electric, fiber, glass_doors, pool, garden]
    
    result = linear_regression.params[0]
    
    for i, x in enumerate(X_test):
        result += linear_regression.params[i+1] * x
    
    return result


prediction = calculate_prediction(150, 1, 0, 3, 1, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1)

print(f'The expected price for the above described house is of ${prediction:.2f}')

