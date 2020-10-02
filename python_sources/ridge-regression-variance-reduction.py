#!/usr/bin/env python
# coding: utf-8

# ### Problem
# We want to reduce the variance of linear regression model
# 
# ### Solution
# Use a linear algorithm that includes a shrinkage penalty (also called regularization) like **Ridge regression**.
# 
# References to understand Ridge Regression: 
# * [Ridge Regression from Microsoft](https://www.youtube.com/watch?v=JZ0BGj6Vv2g)
# 
# * [Ridge Regression for Better Usage](https://towardsdatascience.com/ridge-regression-for-better-usage-2f19b3a202db)
# 
# * [Ridge Regression Explained in Hindi](https://www.youtube.com/watch?v=Yj7sIK0VMg0&list=PLYwpaL_SFmcBhOEPwf5cFwqo5B-cP9G4P&index=17)
# 
# 
# 

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


# In[ ]:


# Load data from Boston Dataset
from sklearn.datasets import load_boston 
boston = load_boston()
dir(boston)


# In[ ]:


data = pd.DataFrame(boston.data)
data.head()


# In[ ]:


# Load target variable
target = pd.DataFrame(boston.target)
target.head()


# In[ ]:


# Feature Scalling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_standardized = scaler.fit_transform(data)
features_standardized = pd.DataFrame(features_standardized)
features_standardized.head()


# In[ ]:


# before we use Ridge Regression, we need to find an optimal parameter
from sklearn.linear_model import RidgeCV

# Create ridge regression with three alpha variable
regr_cv = RidgeCV(alphas=[0.1,1.0, 10.0])

# fit the linear regression
model_cv = regr_cv.fit(features_standardized, target)
model_cv


# In[ ]:


# view coefficients
model_cv.coef_


# In[ ]:


# View the best value for alpha
model_cv.alpha_


# In[ ]:


# Create a ridge regression with an alpha value
from sklearn.linear_model import Ridge
regression = Ridge(alpha=model_cv.alpha_)
model = regression.fit(features_standardized, target)


# In[ ]:




