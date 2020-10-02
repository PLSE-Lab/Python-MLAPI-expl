#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#reading file
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


#shows top 5 rows
train.head()


# In[ ]:


test.head()


# ### statistics

# In[ ]:


#shows info
train.info()


# It says that it has 1460 entries and has 81 columns. 43 has **object** datatype, 35 has **int** datatype,3 has **float** datatype of total memory 924KB

# In[ ]:


#info for test data
test.info()


# This has 1459 rows and 80columns

# In[ ]:


#this shows stats about dataset
train.describe()


# This shows the **count,Mean,standarad deviation, min, max and Interquartile range** values of every column.

# In[ ]:


test.describe()


# In[ ]:


#this shows the shape(rows*columns) of dataset
train.shape


# this shows that it has **1460 rows and 81 columns**.

# In[ ]:


test.shape


# This shows that it has **1459 rows and 80 columns**

# In[ ]:


#This shows all the columns
train.columns


# ### Data Preprocessing

# In[ ]:


#lets save and drop the ID column
#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# #### lets look for outliers and missing values

# some times removing outliers and missing values may effect ouur model badly. the solution is to use Models that are robust to Outliers and Impute missing values using Mean, Median and mode. mostly median plays safe.

# In[ ]:


fig,ax  = plt.subplots(figsize= (8,5))

ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# This shows that we have outliers at right bottom, lets remove them

# In[ ]:


train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)


# as our target variable is Sales price, lets expolore more about it

# In[ ]:


train['SalePrice'].describe()


# you can see that the minimum price is 34900 and max price is 755000. lets see howo the data is distributed

# In[ ]:


plt.hist(train['SalePrice'])


# In[ ]:


sns.distplot(train['SalePrice'])


# you can see here that the Data is 
# - **not normally distributed**
# - **postively skewed**
# - **has peakedness**

# In[ ]:


#we can even get mu and sigma values
from scipy.stats import norm, skew
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')


# apply log transformation, so that the values gets normally distributed

# In[ ]:


#applying log transformation
train['SalePrice'] = np.log(train['SalePrice'])

sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')


# Now you can see that the data of SalePrice is normally distributed. lets see the correlation matrix or heatmap.

# In[ ]:


#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# At first sight, there are two red colored squares that get my attention. The first one refers to the 'TotalBsmtSF' and '1stFlrSF' variables, and the second one refers to the 'GarageX' variables. Both cases show how significant the correlation is between these variables. Actually, this correlation is so strong that it can indicate a situation of multicollinearity. If we think about these variables, we can conclude that they give almost the same information so multicollinearity really occurs.

# ### Feature engineering

# lets combine both test and train

# In[ ]:


all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# In[ ]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na ==0].index).sort_values(ascending = False)
all_data_na


# lets visualize them

# In[ ]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# lets fill/Impute the missing values

# PoolQC : data description says NA means "No Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.

# In[ ]:


all_data["PoolQC"] = all_data["PoolQC"].fillna("None")


# MiscFeature : data description says NA means "no misc feature"
# 

# In[ ]:


all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")


# Alley : data description says NA means "no alley access"

# In[ ]:


all_data["Alley"] = all_data["Alley"].fillna("None")


# Fence : data description says NA means "no fence"

# In[ ]:


all_data["Fence"] = all_data["Fence"].fillna("None")


# FireplaceQu : data description says NA means "no fireplace"

# In[ ]:


all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")


# LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.

# In[ ]:


#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None

# In[ ]:


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')


# GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)

# In[ ]:


for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)


# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement

# In[ ]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)


# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.

# In[ ]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')


# MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.

# In[ ]:


all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)


# MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'

# In[ ]:


all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])


# Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.

# In[ ]:


all_data = all_data.drop(['Utilities'], axis=1)


# Functional : data description says NA means typical

# In[ ]:


all_data["Functional"] = all_data["Functional"].fillna("Typ")


# Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.

# In[ ]:


all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])


# KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
# 

# In[ ]:


all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])


# Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
# 

# In[ ]:


all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])


# SaleType : Fill in again with most frequent which is "WD"

# In[ ]:


all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


# MSSubClass : Na most likely means No building class. We can replace missing values with None

# In[ ]:


all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


# ### More feature engineering

# lets turn numerical variables to categorical variables

# In[ ]:


#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# label encoding the categorical values

# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# #### Adding one more important feature

# Since area related features are very important to determine house prices, we add one more feature which is the total area of basement, first and second floor areas of each house

# In[ ]:


# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# #### Getting dummy categorical features

# In[ ]:


all_data = pd.get_dummies(all_data)
print(all_data.shape)


# #### getting train and test sets

# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
train = all_data[:ntrain]
test = all_data[ntrain:]


# ### from here its time to build the model.

# ### Thank you for visiting my kernel. would love to here your suggestions.

# In[ ]:




