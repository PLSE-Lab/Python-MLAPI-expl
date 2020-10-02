#!/usr/bin/env python
# coding: utf-8

# # Describes EDA techinques that i used to reach top 3%

# # Exploratory data analysis
# 
# **Exploratory Data Analysis (EDA)** is an approach for summarizing, visualizing, and becoming intimately familiar with the important characteristics of a data set.In this notebook we are analysing the dataset for the kaggle competion *House Prices: Advanced Regression Techniques* 
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png)
# 

# In[ ]:


#importing libaries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as st
from sklearn import ensemble, tree, linear_model


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


# In[ ]:


test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")


# We can use the describe() function to get various summary statistics that exclude NaN values.

# In[ ]:


test.describe()


# In[ ]:


train.describe()


# In[ ]:


train.head()


# Use shape function to find the number of rows and coloumns

# In[ ]:


print(train.shape)
print(test.shape)


# ## seperating data in numerical and catogrical & analysing

# There are 1460 instances of training data and 1460 of test data. Total number of attributes equals 81, of which 38 is numeric (including salesprice and id), 41 is categorical .
# 
# Numeric: 1stFlrSF, 2ndFlrSF, 3SsnPorch, BedroomAbvGr, BsmtFinSF1, BsmtFinSF2, BsmtFullBath, BsmtHalfBath, BsmtUnfSF, EnclosedPorch, Fireplaces, FullBath, GarageArea, GarageCars, GarageYrBlt, GrLivArea, HalfBath, KitchenAbvGr, LotArea, LotFrontage, LowQualFinSF, MSSubClass, MasVnrArea, MiscVal, MoSold, OpenPorchSF, OverallCond, OverallQual, PoolArea, ScreenPorch, TotRmsAbvGrd, TotalBsmtSF, WoodDeckSF, YearBuilt, YearRemodAdd, YrSold
# 
# Categorical: Alley, BldgType, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, BsmtQual, CentralAir, Condition1, Condition2, Electrical, ExterCond, ExterQual, Exterior1st, Exterior2nd, Fence, FireplaceQu, Foundation, Functional, GarageCond, GarageFinish, GarageQual, GarageType, Heating, HeatingQC, HouseStyle, KitchenQual, LandContour, LandSlope, LotConfig, LotShape, MSZoning, MasVnrType, MiscFeature, Neighborhood, PavedDrive, PoolQC, RoofMatl, RoofStyle, SaleCondition, SaleType, Street, Utilities,

# In[ ]:


numeric=[f for f in train.columns if train.dtypes[f] != 'object']
numeric


# In[ ]:


catagorical = [i for i in train.columns if train.dtypes[i] == 'object']
catagorical


# In[ ]:


num_features = train.select_dtypes(include=[np.number])
num_features.columns


# In[ ]:


cat_features = train.select_dtypes(include=[object])
cat_features.columns


# Numerical variables Types
# 1. Discrete Values
# 2. Countinues values
# 3. Date related values
# 

# In[ ]:


numeric_features=num_features.drop(['YrSold','YearBuilt', 'YearRemodAdd', 'GarageYrBlt'],axis=1)


# In[ ]:


numeric_features


# In[ ]:


discrete_feature=[feature for feature in numeric_features if len(train[feature].unique())<25]
discrete_feature


# relationship between these discrete features and Sale Price

# In[ ]:


for feature in discrete_feature:
    data=train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# relationship between these countinues features and Sale Price

# In[ ]:


continuous_feature=[feature for feature in numeric_features if feature not in discrete_feature]
continuous_feature.pop(0)


# In[ ]:


for feature in continuous_feature:
    data=train.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# In[ ]:


train['HouseStyle'].unique() 


# plotting catogrical values with repect to their count

# In[ ]:


for feature in cat_features:
    
    sns.countplot(x=feature, data=cat_features);
    plt.title(feature)
    plt.show()


# ## Finding Missing Values
# 

# In[ ]:


train.info()


# 19 attributes have missing values, 5 over 50% of all data. Most of times NA means lack of subject described by attribute, like missing pool, fence, no garage and basement.
# 
# 

# Colomns like PoolQC,MiscFeature,Alley,Fence	having missing  values more than 80% (FireplaceQu,LotFrontage also significant amount of missing values

# In[ ]:


total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1,join='outer', keys=['Total Missing Count', ' % of Total Observations'])
missing_data.index.name ='Feature'
missing_data.head(20)


# In[ ]:


missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# ## outliers
# identifying outliers in dataset

# In[ ]:


for feature in continuous_feature:
    data=train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# ## Skewness and kurtosis
# 1. *Skewness* is a measure of symmetry
# 2. *Kurtosis* is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution

# In[ ]:


train.skew()


# In[ ]:


train.kurt()


# In[ ]:


import scipy.stats as stats

y = train['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=stats.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=stats.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=stats.lognorm)


# ## Correlation
# Correlation is a statistical technique that can show whether and how strongly pairs of variables are related will help us to find the most coorelated feature.we are using corelation heat map and Zoomed heatmap

# Corelation heatmap

# In[ ]:


corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

from observations it is found that OverallQual, GrLivArea, GarageCars, GarageArea',TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt,YearRemodAdd etc are the most correlated features. Inorder to observe correlation closer we use Zoomed Heat Map
# Zoomed Heat Map

# In[ ]:


k= 11
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(train[cols].values.T)
f , ax = plt.subplots(figsize = (14,12))
sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',
            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)


# In[ ]:




