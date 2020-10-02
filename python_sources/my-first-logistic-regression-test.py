#!/usr/bin/env python
# coding: utf-8

# In this notebook I'm trying to "cross-compile" my Octave script to Python (where I am still a newbie).
# I will start with logistic regression as first attempt.
# 
# I will follow the prof. Ng notation for feature vector and data manipulation as teached in the ML courses from Coursera.
# 
# Some deep explanation of Linear regression can be found here:
# 
# [You should definitely take a look here..][1]
# 
# 
#   [1]: https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import math as mat

from scipy import stats
from scipy.stats import norm
#from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
#import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.

import sklearn.linear_model as LinReg
import sklearn.linear_model as LogReg
import sklearn.metrics as metrics

#from scipy.optimize import fmin_bfgs

#loading the data 
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')


# I want to build an "X" matrix with m*n dimensions with the below spec:
# 
#  1. n is the number of features (columns )
#  2. m is the number of examples (raws)
#  3. the first column is always composed by all ones 
# 
# Due to the fact that this dataset is composed by 81 columns, I will now choose as significative features only some of this. 
# I will follow the analysis performed here to justify which feature to use, please take a look here:
# [https://www.kaggle.com/pmarcelino/house-prices-advanced-regression-techniques/comprehensive-data-exploration-with-python][1]
# 
# 
#   [1]: https://www.kaggle.com/pmarcelino/house-prices-advanced-regression-techniques/comprehensive-data-exploration-with-python

# The below features will be the priviledged ones for my first analysis attempt:
# 
#  1. OverallQual 
#  2. GrLivArea
#  3. GarageCars
#  4. TotalBsmtSF
#  5. 1stFlrSF 
#  6. FullBath
#  7. YearBuilt
# 
# This due to the big correlation that they have with SalePrice.

# In[ ]:


data_train.shape


# In[ ]:


# visualize the relationship between some features and the Sale price using scatterplots
#sns.pairplot(data_train, x_vars=['GrLivArea','TotalBsmtSF'], y_vars='SalePrice', size=7, aspect=0.7)


# In[ ]:


vars = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath','YearBuilt']
Y = data_train[['SalePrice']] #dim (1460, 1)
ID_train = data_train[['Id']] #dim (1460, 1)
ID_test = data_test[['Id']]   #dim (1459, 1)
#extract only the relevant feature with cross correlation >0.5 respect to SalePrice
X_matrix = data_train[vars]
X_matrix.shape  #dim (1460,6)

X_test = data_test[vars]
X_test.shape   #dim (1459,6)


# In[ ]:


#check for missing data:
#missing data
total = X_matrix.isnull().sum().sort_values(ascending=False)
#check = houses.isnull().count() #this gives the number of elements for each column (feature)

percent = (X_matrix.isnull().sum()/X_matrix.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#no missing data in this training set


# In[ ]:



#substitute NA data with the mean value of that feature:
X_test['TotalBsmtSF']=X_test['TotalBsmtSF'].fillna(X_test['TotalBsmtSF'].mean())
X_test['GarageCars']=X_test['GarageCars'].fillna(mat.ceil(X_test['GarageCars'].mean()))

#let's drop NA value from the matrix:
#X_test = X_test.dropna()


total = X_test.isnull().sum().sort_values(ascending=False)
#check = houses.isnull().count() #this gives the number of elements for each column (feature)

percent = (X_test.isnull().sum()/X_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


X_test.shape #now the dimensions are (1457, 6) 


# Feature scaling and mean normalization with the preprocessing module.
# It further provides a utility class **StandardScaler** that implements the *transformer* method to compute the mean and standard deviation on a training set so as to be able to later reapply the same transformation on the testing set.

# In[ ]:


max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_matrix)
print(X_train_maxabs)


# In[ ]:


X_test_maxabs = max_abs_scaler.fit_transform(X_test)
print(X_test_maxabs)


# In[ ]:


lr=LinReg.LinearRegression().fit(X_train_maxabs,Y)

Y_pred_train = lr.predict(X_train_maxabs)
print("Lin Reg performance evaluation on Y_pred_train")
print("R-squared =", metrics.r2_score(Y, Y_pred_train))
#print("Coefficients =", lr.coef_)

Y_pred_test = lr.predict(X_test_maxabs)
print("Lin Reg performance evaluation on X_test")
#print("R-squared =", metrics.r2_score(Y, Y_pred_test))
print("Coefficients =", lr.coef_)

