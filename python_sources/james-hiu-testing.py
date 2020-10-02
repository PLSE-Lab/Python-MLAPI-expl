#!/usr/bin/env python
# coding: utf-8

# # **My Housing Price Competition Notebook**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Reading the Data**

# In[ ]:


home = pd.read_csv('../input/home-data-for-ml-course/train.csv',index_col='Id')
test = pd.read_csv('../input/home-data-for-ml-course/test.csv',index_col='Id')

#home.head()
#test.head()


# # Explore the data - Correlations
# Identify property features that have "very Low" correlation with Sale Price and property features highly correlated with one another (multicollinearity)
# 
# **1. Low Correlation Features (-0.1<x<0.1)** - We wish to drop features that appear to have low value in predicting a home's sale price
# <br>
#       <br>  a)<font color=blue> **MSSubClass**:</font> Identifies the type of dwelling involved in the sale (-0.08)
#       <br>  b)<font color=blue> **OverallCond**:</font> Rates the overall condition of the house (-0.08)
#       <br>  c)<font color=blue> **LowQualFinSF**:</font> Low quality finished square feet (all floors) (-0.03)
#       <br>  d)<font color=blue> **MiscVal**:</font> Dollar value of miscellaneous feature (-0.02)
#       <br>  e)<font color=blue> **PoolArea**:</font> Month Sold (0.09)
#       <br>  f)<font color=blue> **MoSold**:</font> Month Sold (0.05)
#       <br>  g)<font color=blue> **YrSold**:</font> Year Sold (-0.03)
# 
# **2. Highly Correlated Features |x|>0.8** - We wish to drop features highly similar features to predict sale price when one feature will do
# <br>
#       <br>  a)<font color=blue> **GarageYrBlt**:</font> GarageYrBlt & YearBuilt (0.83), drop GarageYrBlt because property without garage more likely than garage without property
#       <br>  b)<font color=blue> **TotRmsAbvGrd**:</font> GrLivArea & TotRmsAbvGrd (0.83), drop TotRmsAbvGrd because this variable does not count bathrooms and squarefeet differences is more detailed
#       <br>  c)<font color=blue> **GarageCars**:</font> GarageArea and GarageCars (0.88), drop GarageCars because squarefeet differences is more detailed
# 
# **3. Other Features**
# <br>
#       <br> As for BsmtFinSF2 (-0.01) and BsmtHalfBath (-0.02) having weak correlation with Sales Price.
#       <br> Also, the strong correlation between TotalBsmtSF vs 1stFlrSF (0.82).
#       <br>
#       <br> I am retaining these features for feature engineering.

# In[ ]:


#Generate feature correlation visualization

plt.figure(figsize=(18,18))
sns.heatmap(home.corr(), annot=True, fmt=".2f", vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', cbar_kws= {'orientation': 'horizontal'} )

home.drop(['MSSubClass','OverallCond','LowQualFinSF','MiscVal','PoolArea','MoSold','YrSold','GarageYrBlt','TotRmsAbvGrd','GarageCars'], axis=1, inplace=True)
test.drop(['MSSubClass','OverallCond','LowQualFinSF','MiscVal','PoolArea','MoSold','YrSold','GarageYrBlt','TotRmsAbvGrd','GarageCars'], axis=1, inplace=True)


# # Explore the data - Missing Values
# Identify columns with missing values and decide how to best treat these variables
# 
# **1. Dropped Columns** - We wish to drop variables with a very high number of null values.
#       <br>  a)<font color=blue> **PoolQC**:</font> Pool Quality 1453 null values out of 1460
#       <br>  b)<font color=blue> **MiscFeature**:</font> Miscellaneous features (e.g. Elevators/Shed) 1406 null values out of 1460
#       <br>  c)<font color=blue> **Alley**:</font> Type of alley access to property 1369 null values out of 1460
#       <br>  d)<font color=blue> **Fence**:</font> Fence Quality on the property 1179 null values out of 1460
# 
# **2. Categorical Features:** - For categorical columns with missing values we will encode missing values with the most common type reported in that neighbourhood.
# 
# **3. Numerical Features:** - For numerical columns with missing values we will encode with averages value reported in that neighbourhood.
# 

# In[ ]:


#Identify and report count - columns with missing values
missing_data = home.isnull().sum()
col_with_missing = missing_data[missing_data>0]
col_with_missing.sort_values(inplace=True)
print(col_with_missing)

missing_data2 = test.isnull().sum()
col_with_missing2 = missing_data2[missing_data2>0]
col_with_missing2.sort_values(inplace=True)
print(col_with_missing2)

home.drop(['PoolQC','MiscFeature','Alley','Fence'], axis=1, inplace=True)
test.drop(['PoolQC','MiscFeature','Alley','Fence'], axis=1, inplace=True)

for df in [home, test]:
#Encode missing categorical features with most common type/quality, grouping by neighborhood
    for col in ("Electrical","MasVnrType", "GarageType","BsmtQual","BsmtCond","BsmtFinType1","BsmtFinType2","BsmtExposure","GarageFinish","GarageQual","GarageCond","FireplaceQu","KitchenQual","SaleType"
               ,"Exterior1st","Exterior2nd","Utilities","Functional","MSZoning"):
         df[col] = df.groupby("Neighborhood")[col].transform(lambda x: x.fillna(x.mode()[0]))
#Encode missing numerical features with average values, grouping by neighborhood
    for col2 in ("MasVnrArea","LotFrontage","TotalBsmtSF","GarageArea","BsmtUnfSF","BsmtFinSF2","BsmtFinSF1","BsmtFullBath","BsmtHalfBath"):
        df[col2] = df.groupby('Neighborhood')[col2].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


#Identify Numerical Data
numerical_data = home.select_dtypes(exclude=['object']).drop('SalePrice', axis=1)

#Explore Numerical Data Distribution
fig = plt.figure(figsize=(20,18))
for i in range(len(numerical_data.columns)):
    fig.add_subplot (9,4,i+1)
    sns.distplot(a=numerical_data.iloc[:,i].dropna(), kde=False)
    plt.xlabel(numerical_data.columns[i])
plt.tight_layout()
plt.show()


# In[ ]:


#Identify Categorical Data
categorical_data = home.select_dtypes(['object'])

#Explore Categorical Data Distribution
fig = plt.figure(figsize=(20,18))
for i in range(len(categorical_data.columns)):
    fig.add_subplot(12,4,i+1)
    sns.countplot(x=categorical_data.iloc[:,i])
plt.tight_layout()
plt.show()


# # Modelling the Data

# In[ ]:


#Fit the Model on Training Data
y = home.SalePrice
X = home.drop(['SalePrice'],axis=1)
# Get list of categorical variables
a = (home.dtypes == 'object')
object_cols = list(a[a].index)
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(home[object_cols]))
# One-hot encoding removed index; put it back
OH_cols_train.index = home.index
# Remove categorical columns (will replace with one-hot encoding)
X = X.drop(object_cols, axis=1)

my_model = XGBRegressor()
my_model.fit(X, y)

predictions = my_model.predict(X)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y)))


#Apply the model to Test Data
X_test = test
# Get list of categorical variables
b = (test.dtypes == 'object')
object_cols2 = list(b[b].index)
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_test = pd.DataFrame(OH_encoder.fit_transform(test[object_cols2]))
# One-hot encoding removed index; put it back
OH_cols_test.index = test.index
# Remove categorical columns (will replace with one-hot encoding)
X_test = test.drop(object_cols, axis=1)


predictions2 = my_model.predict(X_test)

print(predictions2)


# # Submit for Evaluation
# 

# In[ ]:


# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': predictions2})
output.to_csv('submission.csv', index=False)

