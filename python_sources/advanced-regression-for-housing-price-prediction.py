#!/usr/bin/env python
# coding: utf-8

# In[481]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
from scipy.stats import norm
from sklearn import preprocessing
from scipy.stats import boxcox
# Any results you write to the current directory are saved as output.


# ****Practicing Concepts of Maching Learning Regression Algorithm with Kaggle Housing Price Prediction****
# 
# Introduction:
# 
# I have been learning maching learning out of pure interest for the past couple of months and want to share my experience so far. I started by taking [this](http://www.coursera.org/learn/machine-learning) maching learning course on coursera which is excellently designed to make one understand the fundamental concepts behind most widely used algorithm in Machine Learning. To put learned concepts into practice I turned to Kaggle which has amazing community to make learning fun and easy.  I started with [Kaggle Learn](http://www.kaggle.com/learn/overview) to familirized myself with the platform and different liberies. And then jumped right into compition to get hands-on practice
# 
# Reference and Learning materials used:
# 
# *  [Kaggle learn](http://www.kaggle.com/learn/overview) - Must if u are just starting
# 
# * [House price predictions: simple linear model](http://www.kaggle.com/lextoumbourou/housing-prices-top-14-with-linear-regression/notebook)
# 
# Structure: 
# 
# * Data preparation
# 
# * Feature Engineering
# 
# * Handling missing values
# 
# * Modeling
# 
# * Submission

#  **Load and prepare Datasets**

# In[482]:


# Read the data and store in dataframe called train_data and test_data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[483]:


# Sneak peak into datasets 
train_data.describe().transpose()


# In[484]:


# Store train Id and test Id for later use
train_Id = train_data['Id']
test_Id = test_data['Id']
# drop Id's as we wont use it in the preprocessing steps
train_data.drop('Id', axis=1, inplace=True)
test_data.drop('Id', axis=1, inplace=True)


# In[485]:


#Lets find the corelation of columns againts SalePrice column and arrange it in ascending order.
train_data.corr()['SalePrice'].sort_values(ascending=False)


# In[486]:


#Plotting top six significant columns which are helping in predition Sale price.
fig, ax=plt.subplots(3,2,figsize=(20,20))
ax[0][0].scatter(x='OverallQual',y='SalePrice',data=train_data)
ax[0][0].set_title('OverallQual vs Sale Price')

ax[0][1].scatter(x='GrLivArea',y='SalePrice',data=train_data)
ax[0][1].set_title('GrLivArea vs SalePrice')

ax[1][0].scatter(x='GarageCars',y='SalePrice',data=train_data)
ax[1][0].set_title('GarageCars vs SalePrice')

ax[1][1].scatter(x='GarageArea',y='SalePrice',data=train_data)
ax[1][1].set_title('GarageArea vs SalePrice')

ax[2][0].scatter(x='TotalBsmtSF',y='SalePrice',data=train_data)
ax[2][0].set_title('TotalBsmtSF vs Sale Price')

ax[2][1].scatter(x='1stFlrSF',y='SalePrice',data=train_data)
ax[2][1].set_title('1stFlrSF vs Sale Price')

plt.show()


# Spot the outliers in GrLivArea vs SalePrice plot and drop them as sugested by the author of the data sets in his data [documentstion](http://ww2.amstat.org/publications/jse/v19n3/decock/datadocumentation.txt )

# In[487]:


# outliers
train_data = train_data.drop(train_data[(train_data['GrLivArea']>4000)].index)


# In[488]:


#train_data = train_data.drop(train_data[(train_data['TotalBsmtSF']>5000) & (train_data['SalePrice']<250000)].index)


# In[489]:


#train_data = train_data.drop(train_data[(train_data['1stFlrSF']>4500) & (train_data['SalePrice']<250000)].index)


# In[490]:


# Check the plot with outlier removed
plt.scatter(train_data['GrLivArea'], y=train_data['SalePrice'])
plt.title('SALE PRICE vs GR LIV AREA (without outliers)')
plt.show()


# Our target variable, SalePrice, plot and check the distribution of SalePrice

# In[491]:


sns.distplot(train_data['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)


# We can see that out SalePrice is positively skewed. We will use BoxCox to make it more normally distributed as the linear models expects normal distribution

# In[492]:


# correcting the skew SalePrice variable using BoxCox and save the lmbda for doing inverse boxcox latr
# Note: if we supply lmbda as 0 here then following is same as log(sale price + 1)
train_data['SalePrice'], max_lmbda = boxcox(train_data['SalePrice']+1)
print(max_lmbda)
#train_data['SalePrice'], max_lmbda = preprocessing.scale(boxcox(train_data['SalePrice']+1)[0][1])


# In[493]:


# Check the distribution of SalePrice after applying boxcox
sns.distplot(train_data['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)


# In[494]:


# Save no of training and testing sample for later use
ntrain = train_data.shape[0]
ntest = test_data.shape[0]
# Extract our target variable i.e the value to predict
y_train = train_data.SalePrice.values


# **Feature Engineering**
# 
# In order to make sure that we perform same Feature Engineering steps to both training and test data sets, lets combine them into one datasets

# In[495]:


# Join train and test data together to perform 
data = pd.concat((train_data, test_data)).reset_index(drop=True)
# Drop SalePrice
data.drop(['SalePrice'], axis=1, inplace=True)
print("data size is : {}".format(data.shape))


# **Handeling missing values**
# 
# Replacing missing values requires close study of the datasets [descriptions](http://ww2.amstat.org/publications/jse/v19n3/decock/datadocumentation.txt)

# In[496]:


# Over view of missing values
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending=False)
missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing.head(35)


# In[497]:


# Alley : data description says NA means "no alley access"
data["Alley"] = data["Alley"].fillna("None")
# BsmtQual : data description says NaN means "no basement"
data["BsmtQual"] = data["BsmtQual"].fillna("None")
# BsmtCond : data description says NaN means "no basement"
data["BsmtCond"] = data["BsmtCond"].fillna("None")
# BsmtExposure : data description says NaN means "no basement"
data["BsmtExposure"] = data["BsmtExposure"].fillna("None")
# BsmtFinType1 : data description says NaN means "no basement"
data["BsmtFinType1"] = data["BsmtFinType1"].fillna("None")
# BsmtFinType2 : data description says NaN means "no basement"
data["BsmtFinType2"] = data["BsmtFinType2"].fillna("None")
# FireplaceQu : data description says NA means "no fireplace"
data["FireplaceQu"] = data["FireplaceQu"].fillna("None")
# Garagetype: data description says NA means "no garrage"
data["GarageType"] = data["GarageType"].fillna("None")
# GarageFinish : data description says NA means "no garrage"
data["GarageFinish"] = data["GarageFinish"].fillna("None")
# GarageQual: data description says NA means "no garrage"
data["GarageQual"] = data["GarageQual"].fillna("None")
# GarageCond : data description says NA means "no garrage"
data["GarageCond"] = data["GarageCond"].fillna("None")
# PoolQC : data description says NA means "No Pool"
data["PoolQC"] = data["PoolQC"].fillna("None")
# MiscFeature : data description says NA means "no misc feature"
data["Fence"] = data["Fence"].fillna("None")
# MiscFeature : data description says NA means "no misc feature"
data["MiscFeature"] = data["MiscFeature"].fillna("None")


# In[498]:


total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending=False)
missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing.head(8)


# In[499]:


sns.barplot(data=data,x='Neighborhood',y='LotFrontage', estimator=np.median)
plt.xticks(rotation=90)
plt.show()
plt.gcf().clear()


# In[500]:


# LotFrontage : Since the area of each street connected to the house property mostlikely have a similar area to
# other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.
data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# In[501]:


data["GarageYrBlt"] = data["GarageYrBlt"].fillna(0)


# In[502]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    data[col] = data[col].fillna(0)


# In[503]:


#MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. 
#We can fill 0 for the area and None for the type.
data["MasVnrType"] = data["MasVnrType"].fillna("None")
data["MasVnrArea"] = data["MasVnrArea"].fillna(0)


# In[504]:


# We will fill with most common value in the column
data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])


# In[505]:


# Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . 
#Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. 
#We can then safely remove it
data = data.drop(['Utilities'], axis=1)


# In[506]:


# Functional : data description says NA means typical
data['Functional'] = data['Functional'].fillna("Typ")


# In[507]:


data['GarageCars'] = data['GarageCars'].fillna(0)
data['GarageArea'] = data['GarageArea'].fillna(0)


# In[508]:


# Fill with most common value inthe column
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
# Fill with most common value inthe column
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
# Fill with most common value inthe column
data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
# Fill with most common value inthe column
data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
# Fill with most common value inthe column
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])


# In[509]:


# MSSubClass : Na most likely means No building class. We can replace missing values with None
data['MSSubClass'] = data['MSSubClass'].fillna("None")


# Check if any missing values are left

# In[510]:


total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending=False)
missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing.head(8)


# Some categorical variable are in numerical type,
# Make them categorical type

# In[511]:


data['YrSold'] = data['YrSold'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)
data['OverallCond'] = data['OverallCond'].astype(str)
data['MSSubClass'] = data['MSSubClass'].astype(str)


# Lets add addition feature called OverallSF, which is the sum of 1stFlrSF,  2ndFlrSF and TotalBsmtSF

# In[512]:


data['OverallSF'] = data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF']


# Label Encoding categorical variables

# In[513]:


from sklearn.preprocessing import LabelEncoder
category = ('LotShape', 'LandSlope', 'OverallCond', 'ExterQual', 'ExterCond', 
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
            'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 
            'YrSold', 'MoSold', 'Street', 'Alley', 'CentralAir', 'MSSubClass')
# process columns, apply LabelEncoder to categorical features
for c in category:
    lbl = LabelEncoder()
    lbl.fit(list(data[c].values))
    data[c] = lbl.transform(list(data[c].values))


# In[514]:


data.groupby("LotShape").count()


# **Skewed features**

# In[515]:


numerical = data.dtypes[data.dtypes != "object"].index
# Check the skew of all numerical features
skewed = data[numerical].apply(lambda x: x.skew()).sort_values(ascending=False)
skewed.head(10)


# In[516]:


# Lets plot the most positively skewed feature
sns.distplot(data[data["MiscVal"] != 0]["MiscVal"])


# We will fix the skew variables using log1p

# In[517]:


skewed = skewed[abs(skewed) > 0.75]
for idx in skewed.index:
    data[idx] = np.log1p(data[idx])    
sns.distplot(data[data["MiscVal"] != 0]["MiscVal"])    


# Get dummies for the categorical features

# In[518]:


data = pd.get_dummies(data)
data.shape


# Get back train and test data 

# In[519]:


train = data[:ntrain]
test = data[ntrain:]


# **Modelling**

# In[520]:


# load libraries
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


# In[521]:


# Validation function
n_folds = 10
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=1).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# LASSO Regression :

# In[522]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0004, random_state=1))
lasso


# In[523]:


score = rmsle_cv(lasso)
print("Lasso score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# Elastic Net Regression :

# In[524]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0004, l1_ratio=.9, random_state=1))


# In[525]:


score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# In[526]:


ENet.fit(train, y_train)
prediction = ENet.predict(test)
print(prediction)


# Submission

# In[527]:


from scipy.special import inv_boxcox1p
true_prediction = inv_boxcox1p(prediction, max_lmbda)
print(true_prediction)


# In[528]:


my_submission = pd.DataFrame({'Id': test_Id, 'SalePrice': true_prediction})
my_submission.to_csv('new_submission', index=False)

