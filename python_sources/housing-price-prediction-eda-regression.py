#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Data
# * Housing price data with information from basement conditions, nieghborhood, to shape of the the lot
# 1. * the descripiton of the columns is in data_description.txt

# # Steps
# 1. open the input files
# 2. inspect the data and look at different features
#     * look at missing data
#     * look at numerical data
#     * look at cateogorical data
# 3. select which features to use

# # Step 1 : open the input files

# In[ ]:


test_path='../input/house-prices-advanced-regression-techniques/test.csv'
train_path = '../input/house-prices-advanced-regression-techniques/train.csv'

test = pd.read_csv(test_path, index_col= 'Id')
train = pd.read_csv(train_path, index_col = 'Id')


# In[ ]:


print("test shape:",test.shape)
print("train shape:",train.shape)


# In[ ]:


train.head()


# # Step 2 : explore the data
# * Checking on the missing data
# * Looking at the distribution of sale prices
# * plot the correlation between sale prices and numerical data

# # Missing Data

# In[ ]:


missing_num = train.isnull().sum()
missing_cols = missing_num[missing_num>0]
print(missing_cols)
print("--- as percentage ---")
print(missing_cols/1460*100)


# * If PoolArea is 0, then PoolQC does not apply
# * NaN on the other data means don't have it - NA
# * Not very important columns, since there are better features to use -- not worrying about it for now

# ## Distribution of sale price
# * skewed a little bit
# * lowest value is 1460, so no 0 values
# * the most expensive is extremly expensive --- has a tail

# In[ ]:


sns.distplot(train['SalePrice'])
plt.show()


# In[ ]:


train['SalePrice'].describe()


# # Numerical Data
# ## Looking at correlation between Sale Price and other numerical features and numerical features to each other
# * There are some closely related features  - Total BsmtSF and 1stFlrSF, GarageCars and GarageArea,  BsmtFullBath and BsmtFinSF1 etc
# * Just looking at sale priec correlation with others:
#     * high correlation with OveralQual, YearBuilt, GarageArea,FullBath,TotalRoomAbvGround / GrLiveArea, 1stFlSF/ TOtalBsmtSF

# In[ ]:


plt.figure(figsize=(15,15))
corrmat = train.corr()
sns.heatmap(corrmat, vmax=.8, square=True)


# * GrLivArea: Above grade (ground) living area square feet would be 1st and 2nd floor SF
# * TotalBasmtSF - basement area
# * Looks like OpenPorchSF and WoodDeckSF is treated similary even though they are not very correlated - take the average of the two?

# In[ ]:


plt.figure(figsize=(20,20))
k = 20 # take the top k most correlated with SalePrice
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': k}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


# potentially useful columns: ['LotFrontage', 'OverallQual', 'YearBuilt', 'MasVnrArea', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'HalfBath', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF','SalePrice']
explore_cols=['MasVnrArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF']
explore_corr = train[explore_cols].corr()
sns.heatmap(explore_corr, vmax=.8, square=True)
plt.show()


# ## Feature Engineering
# * Let's makea total living area
# * Sum the WoodDeckSF and OpenPorchSF into one, then ignore lotfrontage
# * Make number of bathroom be full bath + (HalfBath/2)

# In[ ]:


train['RecSF'] = train['WoodDeckSF'] + train['OpenPorchSF']
train['HouseSF'] = train['TotalBsmtSF'] + train['GrLivArea']
train['Baths'] = train['FullBath'] + (train['HalfBath']/2)


# Let's plot pairwise plot to see if there is any correlations or something that look like outliers
# * HouseSF has two points that are extra large but with low saleprice
# * Again, two points in LotFrontage that are high
# * Looks like partial sale is not correlates with lower price

# In[ ]:


useful_num_cols = ['OverallQual', 'YearBuilt', 'MasVnrArea', 'RecSF','HouseSF','Baths', 'GarageArea','SalePrice']
sns.pairplot(train[useful_num_cols])
plt.show();


# In[ ]:


train[train['LotFrontage']>=300]
# 935 and 1299


# In[ ]:


sns.violinplot(x=train['SaleCondition'], y= train["SalePrice"])
plt.show()
# Looks like partial sell has two population?


# In[ ]:


# These two points are because of partial sale but still, very off
train[train['HouseSF']>=5200]
# 524, 1299


# In[ ]:


sns.violinplot(x=train['LandContour'], y= train["SalePrice"])
plt.show()


# # Dropping the three data points that are way off

# In[ ]:


train.drop([524,935,1299], inplace=True)


# # Cateogorical Data

# In[ ]:


# useful_num_cols = ['OverallQual', 'YearBuilt', 'MasVnrArea', 'RecSF','HouseSF','Baths', 'GarageArea','SalePrice']

cat_cols = train.select_dtypes(include=['object']).columns
# print(cat_cols)# 43 total

fig, ax = plt.subplots(11,4, figsize=(15,40), dpi=100)
i=1
for col in cat_cols:
    plt.subplot(11,4,i)
    sns.violinplot(x=train[col], y= train["SalePrice"])
    i+=1
    
plt.show()
    


# These might matter:
# * MSZoning
# * Neighborhood
# * Condition1
# * HouseStyle
# * Exterior1st
# * MasVnrType
# * ExterQual
# * ExterCond - get rid of Po option?
# * BsmtQual
# * BsmtCond
# * Heating - get rid of Floor?
# * HeatingOC
# * CentralAir - make it binary
# * KitchenQual
# * FireplaceQual -- only matters if it is Excellet
# * Pool - only matter if there is one?
# * MiscFeature  - might be adding some bonus if Gar2 or Shed present
# * SaleType
# * SaleCondition
# 
# Can eliminate options
# * RoofMatl
# 
# Look like these features don't matter much:
# * LotShape
# * LandContour
# * Utilities
# * LotConfig
# * LandSlope
# * Condition2
# * BldgType
# * Exterior2nd
# * MasVnrType
# * Foundation
# * BsmtExposure
# * BsmtFinType1
# * BsmtFinTyp

# In[ ]:


pontential_cat_col=['MSZoning','Neighborhood','Condition1', 'HouseStyle','Exterior1st','MasVnrType','SaleType','SaleCondition']
# 'ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu' - make it into 1 to 5, numerical
"""
       Ex	Excellent - 5
       Gd	Good - 4
       TA	Typical/Average - 3
       Fa	Fair - 2
       Po	Poor - 1
"""
# make 'CentralAir' and 'Pool' binary
# # Heating - get rid of Floor furnace?
# 'MiscFeature' - might be adding bonus money if Gar2 or Shed present
# FireplaceQual -- only matters if it is Excellet

