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


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
#df_train.columns
#df_train.info
#df_train.describe
#df_train['MSSubClass'].dtype
#df_train['KitchenQual'].unique


# **DataTypes:**
# 
# To understand our data, we print out all the datatypes and review them. Most probably, all the object type variables are categorical variables and all the numeric variables are either int64 or float64. NB: We must consider that there could be =categorical variables which or ordinal but stored as int64. For these sort of scenatios, we will take into account the given descriptions of the variables in the 'data_description.txt' file.

# In[ ]:


df_train.dtypes.tail(80)


# **Probable categorical variables**

# In[ ]:


#Probable categrical variables
df_train.loc[:, df_train.dtypes == np.object].dtypes


# **Probable Numeric Variables**

# In[ ]:


#Probable categrical variables
df_train.loc[:, df_train.dtypes != np.object].dtypes


# **Conclusion on datatypes**

# In[ ]:


#somthing about datatypes


# **Missing Values:
# **
# We want to find out if there are missing values in our dataset.
# Note: According to the 'data_description.txt', most Nan values account for scenarios where the house does not have the feature. 
# 
# For example -
# Alley: Type of alley access to property
# 
#        Grvl	Gravel
#        Pave	Paved
#        NA 	No alley access
#        
# We will  replace these types of null entries with an appropriate string i.e. 'NoAlley'

# In[ ]:


#missing data
total_missing = df_train.isnull().sum()
total_missing = total_missing[total_missing > 0]
total_missing.sort_values(inplace=True)
total_missing.plot.bar()


# In[ ]:


#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(19)


# In[ ]:


df_train["PoolQC"].fillna("NoPool", inplace = True)
df_train["MiscFeature"].fillna("NoMiscFeature", inplace = True)
df_train["Alley"].fillna("NoAlley", inplace = True)
df_train["Fence"].fillna("NoFence", inplace = True)
df_train["FireplaceQu"].fillna("NoFirePlace", inplace = True)
df_train["LotFrontage"].fillna(0 , inplace = True)
df_train["GarageCond"].fillna("NoGarage", inplace = True)
df_train["GarageType"].fillna("NoGarage", inplace = True)
df_train["GarageFinish"].fillna("NoGarage", inplace = True)
df_train["BsmtExposure"].fillna("NoGarage", inplace = True)
df_train["BsmtFinType2"].fillna("NoBsmt", inplace = True)
df_train["BsmtFinType1"].fillna("NoBsmt", inplace = True)
df_train["BsmtCond"].fillna("NoBsmt", inplace = True)
df_train["BsmtQual"].fillna("NoBsmt", inplace = True)
df_train["MasVnrArea"].fillna("NoMasVnr", inplace = True)
df_train["MasVnrType"].fillna("NoMasVnr", inplace = True)
df_train["Electrical"].fillna("NoElectrical", inplace = True)

#GarageYrBlt still needs to be resolved

df_train[missing_data.index.values].head(5)


# In[ ]:


#last check for missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(5)


# In[ ]:


#Commparing GarageCond and Garage
df_train[df_train['GarageCond'] != df_train['GarageQual']][['GarageCond', 'GarageQual']].head(5)


# In[ ]:


#Check to see the entries in MsSubclass are categorical or numeric
#Does sklearn read integer values as categorical or numerical?
df_train['MSSubClass'].head(5)


# **Analysing sale price**

# In[ ]:


import seaborn as sns


# In[ ]:


#1 histogram
sns.distplot(df_train['SalePrice'])


# In[ ]:


# correlations between predictor variables and SalePrice
df_train.corr()['SalePrice'].sort_values(ascending=False)


# #Correlation with SalePrice
# 
# OverallQual      0.790982
# 
# GrLivArea        0.708624
# 
# GarageCars       0.640409
# 
# GarageArea       0.623431
# 
# TotalBsmtSF      0.613581
# 
# 1stFlrSF         0.605852

# In[ ]:


#correlation matrix
corrx = df_train.corr()
corrx[corrx > 0.80] 


# Find cells with values greater than 0.8 (no extra etremely strong correlation)
# 
# find correlated pairs

# In[ ]:


# smaller correlation matrix using variables from the pairs and to the sale price correlation calculation
df_train[['YearBuilt',
   'GarageYrBlt',
   'OverallQual', 'GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF']].corr()


# unnecessary or Double data?:
# * GarageCars and GarageArea (number of cars that fit into the garage is due to the garage area)
# * TotalBsmtSF and 1stFloor (Total square feet of basement area and First Floor square feet)
# * TotRmsAbvGrd' and 'GrLivArea(Total rooms above grade (does not include bathrooms) and Above grade (ground) living area square feet)
# 
# 
# * * GarageYrBlt has 0.486362 correlation to Sale price

# kinda pressured to start something lol. so 

# From what I thought were duplicates, I choose the column with higher correlation and made sense

# In[ ]:


df = df_train[['SalePrice','OverallQual', 'GrLivArea','GarageCars','TotalBsmtSF']]
df.head()


# In[ ]:


df.dtypes


# In[ ]:


X_train = df.drop('SalePrice', axis=1)
y_train = df['SalePrice']


# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
from sklearn.model_selection import train_test_split


# In[ ]:


lm.fit(X_train, y_train)


# In[ ]:


b = float(lm.intercept_)


# In[55]:


coeff = pd.DataFrame(lm.coef_, X_train.columns, columns=['Coefficient'])


# In[56]:


coeff


# This might be too over simplified

# 

# In[ ]:


from sklearn import metrics


# In[ ]:


train_lm = lm.predict(X_train)

print('MSE (train)')
print('Linear:', metrics.mean_squared_error(y_train, train_lm))


# In[ ]:


X_test = df_test[['OverallQual', 'GrLivArea','GarageCars','TotalBsmtSF']]


# In[ ]:


missing_val_count_by_column = (X_test.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# In[ ]:


df_test["GarageCars"].fillna(0 , inplace = True)
df_test["TotalBsmtSF"].fillna(0 , inplace = True)


# In[ ]:


X_test = df_test[['OverallQual', 'GrLivArea','GarageCars','TotalBsmtSF']]
y_test= lm.predict(X_test)


# In[ ]:


X_test = df_test[['OverallQual', 'GrLivArea','GarageCars','TotalBsmtSF']]
y_test= lm.predict(X_test)

test_lm = lm.predict(X_test)

print('MSE (test)')
print('Linear:', metrics.mean_squared_error(y_test, test_lm))


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train_plot = y_train.append(pd.Series(y_test[0], index=['SalePrice']))
plt.plot(np.arange(len(train_plot)), train_plot, label='Training')
plt.plot(np.arange(len(y_test))+len(y_train), y_test, label='Testing')
plt.legend()

plt.show()


# In[ ]:


output = pd.concat([ df_test['Id'], pd.Series(y_test)], axis=1, keys=['Id', 'SalePrice'])


# In[50]:


from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train, y_train)
b = float(ridge.intercept_)


# In[53]:


coeff = pd.DataFrame(ridge.coef_, X_train.columns, columns=['Coefficient'])


# In[54]:


coeff


# In[57]:


lm = LinearRegression()

lm.fit(X_train, y_train)
train_lm = lm.predict(X_train)
train_ridge = ridge.predict(X_train)

print('Training MSE')
print('Linear:', metrics.mean_squared_error(y_train, train_lm))
print('Ridge :', metrics.mean_squared_error(y_train, train_ridge))


# In[58]:


test_lm = lm.predict(X_test)
test_ridge = ridge.predict(X_test)

print('Testing MSE')
print('Linear:', metrics.mean_squared_error(y_test, test_lm))
print('Ridge :', metrics.mean_squared_error(y_test, test_ridge))


# In[60]:


train_plot = y_train.append(pd.Series(y_test[0], index=['SalePrice']))
plt.plot(np.arange(len(train_plot)), train_plot, label='Training')
plt.plot(np.arange(len(y_test))+len(y_train), y_test, label='Testing')
plt.legend()

plt.show()


# In[ ]:




