#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


import os
print(os.listdir("../input")) 


# In[ ]:


train = pd.read_csv('../input/housing_train.csv')
test = pd.read_csv('../input/housing_train.csv') 


# In[ ]:


train.head() 


# In[ ]:


train.shape # 80 feature + 1 label. Total of 81 columns and 1460 rows.


# In[ ]:


train.info() 


# **# Segregating categorical and continuous variable**

# In[ ]:


sns.countplot(train.dtypes) 


# In[ ]:


train.dtypes.value_counts() 


# In[ ]:


cat_feature = train.columns[train.dtypes == object]
cat_feature


# In[ ]:


con_feature = train.columns[train.dtypes != object]
con_feature


# **Categorical Variable**

# In[ ]:


cat_feature


# In[ ]:


# Let's work with missing values first before moving forward and replace or drop if needed.
train[cat_feature].isna().sum().sort_values(ascending=True)


# # Dropping the features with too many null values

# In[ ]:


train = train.drop(['FireplaceQu', 'Fence', 'Alley', 'MiscFeature', 'PoolQC'], axis = 1) 
# these columns has been dropped due to too may null values.


# In[ ]:


cat_feature = train.columns[train.dtypes == object]
cat_feature


# # Replacing the null values

# In[ ]:


train[cat_feature].isna().sum().sort_values(ascending=False)


# # Replacing null values and finding relationship with sale prices

# **All the Garage Related Feature - ['GarageCond', 'GarageQual', 'GarageType' and 'GarageFinish']**

# In[ ]:


fig, axes = plt.subplots(2,2, figsize=(12,10)) 

train.GarageCond.value_counts().sort_index().plot.bar(ax=axes[0][0]) 
axes[0][0].set_title("GarageCond", fontsize=18)

train.GarageQual.value_counts().sort_index().plot.bar(ax=axes[0][1]) 
axes[0][1].set_title("GarageQual", fontsize=18)

train.GarageFinish.value_counts().sort_index().plot.bar(ax=axes[1][0]) 
axes[1][0].set_title("GarageFinish", fontsize=18)

train.GarageType.value_counts().sort_index().plot.bar(ax=axes[1][1]) 
axes[1][1].set_title("GarageType", fontsize=18)


# In[ ]:


# Replacing Null Values


# In[ ]:


train['GarageCond'] = train['GarageCond'].fillna('None') 
train['GarageQual'] = train['GarageQual'].fillna('None') 
train['GarageType'] = train['GarageType'].fillna('None') 
train['GarageFinish'] = train['GarageFinish'].fillna('None') 


# In[ ]:


# Plotting Dependencies on Sale Price


# In[ ]:


fig, axis = plt.subplots(2,2, figsize = (12,10))
sns.boxplot(data=train, x = 'GarageCond', y = 'SalePrice', ax=axis[0,0])
axis[0,0].set_title('GarageCond Vs Sale Price' )

sns.boxplot(data=train, x = 'GarageQual', y = 'SalePrice', ax=axis[0,1])
axis[0,1].set_title('GarageQual Vs Sale Price' )

sns.boxplot(data=train, x = 'GarageType', y = 'SalePrice', ax=axis[1,0])
axis[1,0].set_title('GarageType Vs Sale Price' )

sns.boxplot(data=train, x = 'GarageFinish', y = 'SalePrice', ax=axis[1,1])
axis[1,1].set_title('GarageFinish Vs Sale Price')

plt.tight_layout()


# ****So, what did we get from the above plot? If the forage quality is excellent/good and if furnished built-in garage is there then sale price increases.

# **All the basment related feature - BsmtFinType2', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1**

# In[ ]:


fig, axes = plt.subplots(3,2, figsize=(12,10)) 

train.BsmtFinType2.value_counts().sort_index().plot.bar(ax=axes[0][0]) 
axes[0][0].set_title("BsmtFinType2", fontsize=16)

train.BsmtExposure.value_counts().sort_index().plot.bar(ax=axes[0][1]) 
axes[0][1].set_title("BsmtExposure", fontsize=16)

train.BsmtCond.value_counts().sort_index().plot.bar(ax=axes[1][0]) 
axes[1][0].set_title("BsmtCond", fontsize=16)

train.BsmtQual.value_counts().sort_index().plot.bar(ax=axes[1][1]) 
axes[1][1].set_title("BsmtQual", fontsize=16)

train.BsmtFinType1.value_counts().sort_index().plot.bar(ax = axes[2][0])
axes[2][0].set_title("BsmtFinType1", fontsize=16)

train.MasVnrType.value_counts().sort_index().plot.bar(ax = axes[2][1])
axes[2][1].set_title("MasVnrType", fontsize=16)

plt.tight_layout() 


# **Replacing null values**

# In[ ]:


train['BsmtFinType2'] = train['BsmtFinType2'].fillna('None') 
train['BsmtExposure'] = train['BsmtExposure'].fillna('None') 
train['BsmtCond'] = train['BsmtCond'].fillna('None') 
train['BsmtQual'] = train['BsmtQual'].fillna('None') 
train['BsmtFinType1'] = train['BsmtFinType1'].fillna('None') 
train['MasVnrType'] = train['MasVnrType'].fillna('None') 


# **Plotting Dependecies on Sale Price**

# In[ ]:


fig, axis = plt.subplots(3,2, figsize = (14,12))

sns.boxplot(data=train, x = 'BsmtFinType2', y = 'SalePrice', ax=axis[0,0])
axis[0,0].set_title('BsmtFinType2 Vs Sale Price' )

sns.violinplot(data=train, x = 'BsmtExposure', y = 'SalePrice', ax=axis[0,1])
axis[0,1].set_title('BsmtExposure Vs Sale Price' )

sns.boxplot(data=train, x = 'BsmtCond', y = 'SalePrice', ax=axis[1,0])
axis[1,0].set_title('BsmtCond Vs Sale Price' )

sns.violinplot(data=train, x = 'BsmtQual', y = 'SalePrice', ax=axis[1,1])
axis[1,1].set_title('BsmtQual Vs Sale Price')

sns.boxplot(data= train, x = 'BsmtFinType1', y = 'SalePrice', ax = axis[2,0])
axis[2,0].set_title('BsmtFinType1 Vs Sale Price')

sns.violinplot(data= train, x = 'MasVnrType', y = 'SalePrice', ax = axis[2,1])
axis[2,1].set_title('MasVnrType Vs Sale Price')

plt.tight_layout() 


# So, what did we learn from the above plot?
# 
# * The house sale price increases; if the quality of basment is excellent/good.
# * The house sale price increases; if the basment condition is good.

# **'Electrical', 'Heating', 'HeatingQC', 'CentralAir'**

# In[ ]:


fig, axis = plt.subplots(2,2, figsize = (12,10))

sns.violinplot(data=train, x = 'Electrical', y = 'SalePrice', ax=axis[0,0])
axis[0,0].set_title('Electrical Vs Sale Price' )

sns.boxplot(data=train, x = 'Heating', y = 'SalePrice', ax=axis[0,1])
axis[0,1].set_title('Heating Vs Sale Price' )

sns.boxplot(data=train, x = 'HeatingQC', y = 'SalePrice', ax=axis[1,0])
axis[1,0].set_title('HeatingQC Vs Sale Price' )

sns.violinplot(data=train, x = 'CentralAir', y = 'SalePrice', ax=axis[1,1])
axis[1,1].set_title('CentralAir Vs Sale Price')

plt.tight_layout()


# In[ ]:


# So, we can see from the above plot that having Excellent Heating Quality and Central Air Conditioning increase the sale prices.


# **SaleCondition , Condition2,  Condition1 , SaleType**

# In[ ]:


fig, axis = plt.subplots(2,2, figsize = (14,12))

sns.boxplot(data=train, x = 'SaleCondition', y = 'SalePrice', ax=axis[0,0])
axis[0,0].set_title('SaleCondition Vs Sale Price' )

sns.boxplot(data=train, x = 'Condition2', y = 'SalePrice', ax=axis[0,1])
axis[0,1].set_title('Condition2  Vs Sale Price' )

sns.boxplot(data=train, x = 'Condition1', y = 'SalePrice', ax=axis[1,0])
axis[1,0].set_title('Condition1 Vs Sale Price' )

sns.boxplot(data=train, x = 'SaleType', y = 'SalePrice', ax=axis[1,1])
axis[1,1].set_title('SaleType Vs Sale Price')

plt.tight_layout()


# So, what did we get from the above graph. If the sale type is new then the price of the home increase same can be said for the partial sale condition. I am not too sure about the Condition1 and Condition2 as the attributes are not too familiar.
# 
# The most important thing in EDA is the proper understanding all the features and their attributes. If we fail to understand them, then we won't be able to predict the hosuing prices properly.

# **Some other important feature and their relation with SalePrice**

# In[ ]:


fig, axis = plt.subplots(5,2, figsize=(12, 30))

sns.violinplot(data = train, x = 'BldgType', y = 'SalePrice', ax = axis[0][0])
axis[0][0].set_title('BldgType Vs Sale Price', fontsize= 14)

sns.boxplot(data= train, x ='RoofMatl', y = 'SalePrice', ax = axis[0,1])
axis[0,1].set_title('RoofMatl Vs Sale Price')

sns.violinplot(data = train, x = 'LandSlope', y = 'SalePrice' , ax = axis[1,0])
axis[1,0].set_title('LandSlope Vs Sale Price')

sns.boxplot(data = train, x = 'RoofStyle', y = 'SalePrice' , ax = axis[1,1])
axis[1,1].set_title('RoofStyle Vs Sale Price')

sns.boxplot(data = train, x = 'LotConfig', y = 'SalePrice' , ax = axis[2,0])
axis[2,0].set_title('LotConfig Vs Sale Price')

sns.violinplot(data = train, x = 'Utilities', y = 'SalePrice' , ax = axis[2,1])
axis[2,1].set_title('Utilities  Vs Sale Price')

sns.violinplot(data = train, x = 'LandContour', y = 'SalePrice' , ax = axis[3,0])
axis[3,0].set_title('LandContour  Vs Sale Price')

sns.boxplot(data = train, x = 'LotShape', y = 'SalePrice' , ax = axis[3,1])
axis[3,1].set_title('LotShape  Vs Sale Price')

sns.violinplot(data = train, x = 'Street', y = 'SalePrice' , ax = axis[4,0])
axis[4,0].set_title('Street Vs Sale Price')

sns.boxplot(data = train, x = 'HouseStyle', y = 'SalePrice' , ax = axis[4,1])
axis[4,1].set_title('HouseStyle Vs Sale Price')

plt.tight_layout()

