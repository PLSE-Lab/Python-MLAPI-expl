#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Hi
# 
# This is my first attempt at DL/ML of something other than images. Therefore, I wanted to start with well explored dataset that has a lot of good kernels, relatively small amount of data and this competiotion seems to suit it. This kernel is mostly about getting farmiliar with the data through visualization, common feature tricks as log transform, missing data handling and generally how to approach various features of the data. Hopefully someone will find this useful.
# <br> Upvotes, notes and remarks are very appriciated!
# <brr>
# I heavily used those kernels: 
# <br>
# Source1 : https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# <br>
# Source2 : https://www.kaggle.com/bsivavenu/house-price-calculation-methods-for-beginners
# <br> 
# Source3 : https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# 
# 
# ### Table of interest:
# > ### 1. Importing libraries
# > ### 2. Importing and inquiring data
# > > #### 2.1 Importing data
# > > #### 2.2 Quiring data
# > ### 3. The predicted variable - Sales price Skew & kurtosis analysis
# > > #### 3.1 Observing histogram
# > > #### 3.2 Tansforming log or box cox: 
# > ### 4. Missing data
# > > #### 4.1 Presenting and locating missing data
# > > #### 4.2 Replacing the missing data
# > ### 5. Numerical and Categorial features
# > > #### 5.1 Splitting the data into categorial and numerical features
# > > #### 5.2 Box cox transform for skewd numerical data
# > ### 6.Plotting the data
# > > #### 6.1 Visually comparing data to sale prices
# > > #### 6.2 Comparing data to sale price through correlation matrix
# > > #### 6.3 Pairplot for the most intresting parameters
# > ### 7 Feature Engineering 
# > > #### 7.1 Creating new features
# > > #### 7.2 Evaluation created features
# 
# 
# 
# 

# ## 1. Importing libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2. Import And quiring data

# ### 2.1 Importing data

# Importing and dropping ID.

# In[ ]:


# Read files
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# For future projects 
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)

print (train.columns)
print(test.columns)
print(train.shape,test.shape)


# ### 2.2 Quiring the data
# Just watching what's out there

# In[ ]:


train.describe()


# In[ ]:


train.head(7)


# In[ ]:


test.head(7)


# ## 3. The predicted variable - Sales price Skew & kurtosis analysis
# The predicted variable is probably the most important variable, therefore it should be inspected throughly. 
# <br> It turns out models work better with symmetric gaussian distributions, therefore we want to get rid of the skewness by using log transformation. More on log transformation later
# <br> <br> Skew: 
# \begin{equation} 
# skew \left( X  \right) = E[ \frac{X-\mu}{\sigma} ]^3
# \end{equation} https://en.wikipedia.org/wiki/Skewness
# ![](https://www.managedfuturesinvesting.com/images/default-source/default-album/measure-of-skewness.jpg?sfvrsn=0)
#  
# <br> Kurtosis: $$kurtosis(X) = E[ (\frac{X-\mu}{\sigma})^4  ]$$
# https://en.wikipedia.org/wiki/Kurtosis
# ![](https://siemensplm.i.lithium.com/t5/image/serverpage/image-id/38460iB0F0D63C4F9B568A/image-size/large?v=1.0&px=999)

# ### 3.1 Observing Sale price histogram
# 

# In[ ]:


train['SalePrice'].describe()
sns.distplot(train['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


# ### 3.2 Tansforming: \begin{equation*} Y = log(1 + X)) \end{equation*}
# Should correct for skew.
# <br> A random example of a different log transformation
# ![](http://www.biostathandbook.com/pix/transformfig1.gif)

# In[ ]:


from scipy import stats
from scipy.stats import norm, skew #for some statistics

# Plot histogram and probability
fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.subplot(1,2,2)
res = stats.probplot(train['SalePrice'], plot=plt)
plt.suptitle('Before transformation')

# Apply transformation
train.SalePrice = np.log1p(train.SalePrice )
# New prediction
y_train = train.SalePrice.values


# Plot histogram and probability after transformation
fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.subplot(1,2,2)
res = stats.probplot(train['SalePrice'], plot=plt)
plt.suptitle('After transformation')


# ## 4. Missing data

# ### 4.1 Locating missing data
# 

# In[ ]:


# Missing data in train
train_nas = train.isnull().sum()
train_nas = train_nas[train_nas>0]
train_nas.sort_values(ascending=False)


# In[ ]:


# Missing data in test
test_nas = test.isnull().sum()
test_nas = test_nas[test_nas>0]
test_nas.sort_values(ascending = False)


# In[ ]:


#missing data percent plot
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# ### 4.2 Replacing the missing data
# 
# Correcting for the format, mostly filling NaN with "No" or "0"

# In[ ]:



# Handle missing values for features where median/mean or most common value doesn't make sense
# Alley : data description says NA means "no alley access"
train.loc[:, "Alley"] = train.loc[:, "Alley"].fillna("None")
# BedroomAbvGr : NA most likely means 0
train.loc[:, "BedroomAbvGr"] = train.loc[:, "BedroomAbvGr"].fillna(0)
# BsmtQual etc : data description says NA for basement features is "no basement"
train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")
train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")
train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")
train.loc[:, "BsmtFullBath"] = train.loc[:, "BsmtFullBath"].fillna(0)
train.loc[:, "BsmtHalfBath"] = train.loc[:, "BsmtHalfBath"].fillna(0)
train.loc[:, "BsmtUnfSF"] = train.loc[:, "BsmtUnfSF"].fillna(0)
# CentralAir : NA most likely means No
train.loc[:, "CentralAir"] = train.loc[:, "CentralAir"].fillna("N")
# Condition : NA most likely means Normal
train.loc[:, "Condition1"] = train.loc[:, "Condition1"].fillna("Norm")
train.loc[:, "Condition2"] = train.loc[:, "Condition2"].fillna("Norm")
# EnclosedPorch : NA most likely means no enclosed porch
train.loc[:, "EnclosedPorch"] = train.loc[:, "EnclosedPorch"].fillna(0)
# External stuff : NA most likely means average
train.loc[:, "ExterCond"] = train.loc[:, "ExterCond"].fillna("TA")
train.loc[:, "ExterQual"] = train.loc[:, "ExterQual"].fillna("TA")
# Fence : data description says NA means "no fence"
train.loc[:, "Fence"] = train.loc[:, "Fence"].fillna("No")
# FireplaceQu : data description says NA means "no fireplace"
train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")
train.loc[:, "Fireplaces"] = train.loc[:, "Fireplaces"].fillna(0)
# Functional : data description says NA means typical
train.loc[:, "Functional"] = train.loc[:, "Functional"].fillna("Typ")
# GarageType etc : data description says NA for garage features is "no garage"
train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")
train.loc[:, "GarageArea"] = train.loc[:, "GarageArea"].fillna(0)
train.loc[:, "GarageYrBlt"] = train.loc[:, "GarageYrBlt"].fillna(0)

train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)
# HalfBath : NA most likely means no half baths above grade
train.loc[:, "HalfBath"] = train.loc[:, "HalfBath"].fillna(0)
# HeatingQC : NA most likely means typical
train.loc[:, "HeatingQC"] = train.loc[:, "HeatingQC"].fillna("TA")
# KitchenAbvGr : NA most likely means 0
train.loc[:, "KitchenAbvGr"] = train.loc[:, "KitchenAbvGr"].fillna(0)
# KitchenQual : NA most likely means typical
train.loc[:, "KitchenQual"] = train.loc[:, "KitchenQual"].fillna("TA")
# LotFrontage : NA most likely means no lot frontage
train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna(0)
# LotShape : NA most likely means regular
train.loc[:, "LotShape"] = train.loc[:, "LotShape"].fillna("Reg")
# MasVnrType : NA most likely means no veneer
train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")
train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)
# MiscFeature : data description says NA means "no misc feature"
train.loc[:, "MiscFeature"] = train.loc[:, "MiscFeature"].fillna("No")
train.loc[:, "MiscVal"] = train.loc[:, "MiscVal"].fillna(0)
# OpenPorchSF : NA most likely means no open porch
train.loc[:, "OpenPorchSF"] = train.loc[:, "OpenPorchSF"].fillna(0)
# PavedDrive : NA most likely means not paved
train.loc[:, "PavedDrive"] = train.loc[:, "PavedDrive"].fillna("N")
# PoolQC : data description says NA means "no pool"
train.loc[:, "PoolQC"] = train.loc[:, "PoolQC"].fillna("No")
train.loc[:, "PoolArea"] = train.loc[:, "PoolArea"].fillna(0)
# SaleCondition : NA most likely means normal sale
train.loc[:, "SaleCondition"] = train.loc[:, "SaleCondition"].fillna("Normal")
# ScreenPorch : NA most likely means no screen porch
train.loc[:, "ScreenPorch"] = train.loc[:, "ScreenPorch"].fillna(0)
# TotRmsAbvGrd : NA most likely means 0
train.loc[:, "TotRmsAbvGrd"] = train.loc[:, "TotRmsAbvGrd"].fillna(0)
# Utilities : NA most likely means all public utilities
train.loc[:, "Utilities"] = train.loc[:, "Utilities"].fillna("AllPub")
# WoodDeckSF : NA most likely means no wood deck
train.loc[:, "WoodDeckSF"] = train.loc[:, "WoodDeckSF"].fillna(0)


train.head(7)


# Checking that we got rid of all the NaN's

# In[ ]:


#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# In[ ]:


#dealing with the last NAN
# train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max() #just checking that there's no missing data missing...


# ## 5. Numerical and Categorial features

# ### 5.1 Splitting the data into categorial and numerical features

# In[ ]:


# Differentiate numerical features (minus the target) and categorical features
categorical_features = train.select_dtypes(include=['object']).columns
print(categorical_features)
numerical_features = train.select_dtypes(exclude = ["object"]).columns
print(numerical_features)

numerical_features = numerical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
train_num = train[numerical_features]
train_cat = train[categorical_features]


# Check all numerical/categorial

# In[ ]:


train_num.head(5)


# In[ ]:


train_cat.head(5)


# ### 5.2 Box cox transform for skewd numerical data
# Another transformation to reduce skew. 
# <br> Equation:
# ![](https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2015/07/boxcox-formula-1.png)
# <br> Transformation example:
# ![](https://www.itl.nist.gov/div898/handbook/eda/section3/gif/boxcox.gif)

# In[ ]:


# Plot skew value for each numerical value
from scipy.stats import skew 
skewness = train_num.apply(lambda x: skew(x))
skewness.sort_values(ascending=False)


# Encode categorial features: can and should be replaced.

# In[ ]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.0001
for feat in skewed_features:
    train[feat] = boxcox1p(train[feat], lam)
    
    
from scipy.stats import skew 
train_num = train[numerical_features]  
train[numerical_features] = train_num


train_num.head(5)


# Observe the correction. 
# We can see that a lot of parameters remained skewd. I suspect that's for variables that have a lot of 0. 

# In[ ]:


skewness = train_num.apply(lambda x: skew(x))
skewness.sort_values(ascending=False)


# In[ ]:


train.head(10)


# ## 6.Plotting the data

# ### 6.1 Visually comparing data to sale prices
# One can observe the behaviour of the variables, locate outlier and more.

# In[ ]:


vars = train.columns
# vars = numerical_features
figures_per_time = 4
count = 0 
for var in vars:
    x = train[var]
    y = train['SalePrice']
    plt.figure(count//figures_per_time,figsize=(25,5))
    plt.subplot(1,figures_per_time,np.mod(count,4)+1)
    plt.scatter(x, y);
    plt.title('f model: T= {}'.format(var))
    count+=1
        


# ### Optional: Box plot
# 
# Box plot is heavy, one can manualy choose the intresting parameters

# In[ ]:



# vars_box = ['OverallQual','YearBuilt','BedroomAbvGr']
vars_box = categorical_features
for var in vars_box:
    data = pd.concat([train['SalePrice'], train[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)


# ### 6.2 Comparing data to sale price through correlation matrix

# Numerical values correlation matrix, to locate dependencies between different variables. 

# In[ ]:


# Complete numerical correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# #### Largest correlation with Sale Price
# Its important to remmber that this are 2D correlations, between sale price and another variable. When stacking all of the parameters the dependencies the picture gets more complex.

# In[ ]:


# saleprice correlation matrix
corr_num = 15 #number of variables for heatmap
cols_corr = corrmat.nlargest(corr_num, 'SalePrice')['SalePrice'].index
corr_mat_sales = np.corrcoef(train[cols_corr].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)
plt.show()


# ### 6.3 Pairplot for the most intresting parameters

# In[ ]:


# pair plots for variables with largest correlation
var_num = 10
vars = cols_corr[0:var_num]
sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[vars], size = 2.5)
plt.show();


# ## Replacing format

# ## 7 Feature Engineering 

# ### 7.1 Creating new features

# In[ ]:


# 2* Combinations of existing features
# Overall quality of the house
train["OverallGrade"] = train["OverallQual"] * train["OverallCond"]
# Overall fireplace score
train["FireplaceScore"] = train["Fireplaces"] * train["FireplaceQu"]
# Overall garage score
train["GarageScore"] = train["GarageArea"] * train["GarageQual"]


# Total number of bathrooms
train["TotalBath"] = train["BsmtFullBath"] + (0.5 * train["BsmtHalfBath"]) + train["FullBath"] + (0.5 * train["HalfBath"])
# Total SF for house (incl. basement)
train["AllSF"] = train["GrLivArea"] + train["TotalBsmtSF"]
# Total SF for 1st + 2nd floors
train["AllFlrsSF"] = train["1stFlrSF"] + train["2ndFlrSF"]
# Total SF for porch
train["AllPorchSF"] = train["OpenPorchSF"] + train["EnclosedPorch"] + train["3SsnPorch"] + train["ScreenPorch"]
# Has masonry veneer or not



train["OverallQual-s2"] = train["OverallQual"] ** 2
train["OverallQual-s3"] = train["OverallQual"] ** 3
train["OverallQual-Sq"] = np.sqrt(train["OverallQual"])
train["AllSF-2"] = train["AllSF"] ** 2
train["AllSF-3"] = train["AllSF"] ** 3
train["AllSF-Sq"] = np.sqrt(train["AllSF"])
train["AllFlrsSF-2"] = train["AllFlrsSF"] ** 2
train["AllFlrsSF-3"] = train["AllFlrsSF"] ** 3
train["AllFlrsSF-Sq"] = np.sqrt(train["AllFlrsSF"])
train["GrLivArea-2"] = train["GrLivArea"] ** 2
train["GrLivArea-3"] = train["GrLivArea"] ** 3
train["GrLivArea-Sq"] = np.sqrt(train["GrLivArea"])
train["GarageCars-2"] = train["GarageCars"] ** 2
train["GarageCars-3"] = train["GarageCars"] ** 3
train["GarageCars-Sq"] = np.sqrt(train["GarageCars"])
train["TotalBath-2"] = train["TotalBath"] ** 2
train["TotalBath-3"] = train["TotalBath"] ** 3
train["TotalBath-Sq"] = np.sqrt(train["TotalBath"])


# ### 7.2 Evaluation created features

# Corr with new features

# In[ ]:


# Complete numerical correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# Sales price corr with new features

# In[ ]:


# saleprice correlation matrix
corr_num = 15*3 #number of variables for heatmap
cols_corr = corrmat.nlargest(corr_num, 'SalePrice')['SalePrice'].index
corr_mat_sales = np.corrcoef(train[cols_corr].values.T)
sns.set(font_scale=1.25  )
f, ax = plt.subplots(figsize=(12 * 3, 9 * 3))
hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)
plt.show()

