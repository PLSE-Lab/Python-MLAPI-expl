#!/usr/bin/env python
# coding: utf-8

# ## **0. Introduction**

# In[ ]:


from datetime import datetime
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)
from scipy.stats import probplot
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('ignore')

SEED = 42
PATH = '../input/'


# In[ ]:


def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set on axis 0
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:1459], all_data.loc[1460:].drop(['SalePrice'], axis=1)

df_train = pd.read_csv(PATH + 'train.csv')
df_test = pd.read_csv(PATH + 'test.csv')
df_all = concat_df(df_train, df_test)

df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set' 

dfs = [df_train, df_test]

print('Number of Training Examples = {}'.format(df_train.shape[0]))
print('Number of Test Examples = {}\n'.format(df_test.shape[0]))
print('Training X Shape = {}'.format(df_train.shape))
print('Training y Shape = {}\n'.format(df_train['SalePrice'].shape[0]))
print('Test X Shape = {}'.format(df_test.shape))
print('Test y Shape = {}\n'.format(df_test.shape[0]))


# ## **1. Exploratory Data Analysis**

# ### **1.1 Overview**
# * Training set has **1460** samples and test set has **1459** samples
# * Training set have **81** features and test set have **80** features
# * One extra feature in training set is `SalePrice` which is the target to predict

# In[ ]:


print(df_all.info())
df_all.sample(5)


# ### **1.2 Missing Values**
# There are many features with missing values in both training set and test set. `display_missing` function shows the count of missing values in every feature if they have at least 1 missing value.
# * Training set has missing values in **19** features
# * Test set has missing values in **32** features
# 
# There are two types of missing data in this dataset; **systematically** missing data and **randomly** missing data. Majority of the missing data are **systematically** missing which is easier to fix, but **randomly** missing data requires extra effort. It is better to deal with those types of missing data separately.

# In[ ]:


def display_missing(df):    
    for col in df.columns.tolist():
        if df[col].isnull().sum():
            print('{} column missing values: {}/{}'.format(col, df[col].isnull().sum(), len(df)))
    print('\n')
    
for df in dfs:
    print('{}'.format(df.name))
    display_missing(df)


# #### **1.2.1 Systematically Missing Data**
# `MasVnrArea` and `MasVnrType` are defined as masonry veneer area in square feet and masonry veneer type. Missing values in those features mean that there is no masonry veneer in those houses. Missing values in `MasVnrArea` are filled with **0** and missing values in `MasVnrType` are filled with **None**.
# 
# `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF` and `TotalBsmtSF` are basement area in square feet. There is only **1** missing value in those features and it is the same house. That house has missing values in other basement features as well and it most likely doesn't have basement. That's why missing values in `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF` and `TotalBsmtSF` are filled with **0**. The other basement features are categorical. `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2` and `BsmtQual` are missing in houses without basements, so they are filled with **None**. There are also ordinal basement features with missing values such as `BsmtFullBath` and `BsmtHalfBath`. Missing values in those features are filled with **0**.
# 
# There are **7** garage features and all of them have missing values. `GarageArea` is defined as garage area in square feet. There is only **1** house with missing `GarageArea`. That house also has missing value in `GarageCars` feature. Both `GarageArea` and `GarageCars` features are filled with **0** because that house has no garage. `GarageType`, `GarageYrBlt`, `GarageFinish`, `GarageQual` and `GarageCond` are categorical garage features. All of them except `GarageYrBlt` are filled with **None**. `GarageYrBlt` is filled with **0** because it is a numerical type.
# 
# Other systematically missing categorical features are `Alley`, `Fence`, `FireplaceQu`, `MiscFeature` and `PoolQC`. There are missing values in them because those features doesn't exist in those houses. Missing values in those features are also filled with **None**.

# In[ ]:


# Filling masonry veneer features
df_all['MasVnrArea'] = df_all['MasVnrArea'].fillna(0)
df_all['MasVnrType'] = df_all['MasVnrType'].fillna('None')

# Filling continuous basement features
for feature in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
    df_all[feature] = df_all[feature].fillna(0)

# Filling categorical basement features
for feature in ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']:
    df_all[feature] = df_all[feature].fillna('None')

# Filling continuous garage features
for feature in ['GarageArea', 'GarageCars', 'GarageYrBlt']:
    df_all[feature] = df_all[feature].fillna(0)

# Filling categorical garage features
for feature in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    df_all[feature] = df_all[feature].fillna('None')
    
# Filling other categorical features
for feature in ['Alley', 'Fence', 'FireplaceQu', 'MiscFeature', 'PoolQC']:
    df_all[feature] = df_all[feature].fillna('None')

display_missing(df_all)


# #### **1.2.2 Randomly Missing Data**
# After filling systematically missing data, there are few features left with missing values. Those are the remaining randomly missing data. A house can't exist without those features, so they can't be filled with **0** or **None**. The amount of randomly missing data in a feature is also smaller which makes it possible to fill them with descriptive statistical measures. The statistical measures are based on the groups of neighborhoods and building classes because a house would most likely look like its neighbors.
# *  `Electrical`, `Exterior1st`, `Exterior2nd`, `Functional`, `KitchenQual`, `MSZoning`, `SaleType` and `Utilities` are categorical features with randomly missing data and missing values in those features are filled with mode of building class and neighborhood groups
# * `LotFrontage` is the only randomly missing continuous feature and missing values in `LotFrontage` are filled with the median values of neighborhoods

# In[ ]:


# Filling missing values in categorical features with the mode value of neighborhood and house type
for feature in ['Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual', 'MSZoning', 'SaleType', 'Utilities']:
    df_all[feature] = df_all.groupby(['Neighborhood', 'MSSubClass'])[feature].apply(lambda x: x.fillna(x.mode()[0]))

# Filling the missing values in LotFrontage with the median of neighborhood
df_all['LotFrontage'] = df_all.groupby(['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))

display_missing(df_all)


# ### **1.3 Target Distribution**
# Target is not normally distributed. Training set `SalePrice` skew is **1.88** which clearly shows that the target has high positive skew. The distribution is asymmetrical because of the extremely high outliers. 
# 
# Training set `SalePrice` kurtosis is **6.53** which is an indicator of the tail extremity. Mean `SalePrice` is **180921.2**, however, median is **163000**. This huge gap between mean and median is also the effect of outliers.
# 
# Probability plot clearly illustrates that outliers will strongly affect the regression models since a single outlier may result in all predictor coefficients being biased. The probability being a convex curve rather than a straight line is the result of the skewness.

# In[ ]:


print('Training Set SalePrice Skew: {}'.format(df_train['SalePrice'].skew()))
print('Training Set SalePrice Kurtosis: {}'.format(df_train['SalePrice'].kurt()))
print('Training Set SalePrice Mean: {}'.format(df_train['SalePrice'].mean()))
print('Training Set SalePrice Median: {}'.format(df_train['SalePrice'].median()))
print('Training Set SalePrice Max: {}'.format(df_train['SalePrice'].max()))

fig, axs = plt.subplots(nrows=2, figsize=(16, 16))
plt.subplots_adjust(left=None, bottom=5, right=None, top=6, wspace=None, hspace=None)

sns.distplot(df_train['SalePrice'], hist=True, ax=axs[0])
probplot(df_train['SalePrice'], plot=axs[1])

axs[0].set_xlabel('Sale Price', size=12.5, labelpad=12.5)
axs[1].set_xlabel('Theoretical Quantiles', size=12.5, labelpad=12.5)
axs[1].set_ylabel('Ordered Values', size=12.5, labelpad=12.5)

for i in range(2):
    axs[i].tick_params(axis='x', labelsize=12.5)
    axs[i].tick_params(axis='y', labelsize=12.5)

axs[0].set_title('Distribution of Sale Price in Training Set', size=15, y=1.05)
axs[1].set_title('Sale Price Probability Plot', size=15, y=1.05)

plt.show()


# ### **1.4 Correlations**
# Features are strongly correlated with target. **8** features have more than **0.3** correlation coefficient with `SalePrice` and **3** of them are higher than **0.6**. It looks like multicollinearity occurs in the dataset. 
# 
# The other features are also strongly correlated with each other and dependent to each other. There are more than **25** correlations with at least **0.5** coefficient. The highest among them is between `GarageArea` and `GarageCars` which is **0.88**.
# 
# The correlation coefficients of training set and test set are very close.

# In[ ]:


df_train, df_test = divide_df(df_all)
# Dropping categorical features
cols = ['GarageYrBlt', 'Id', 'MSSubClass', 'MoSold', 'YearBuilt', 'YearRemodAdd', 'YrSold']

df_train_corr = df_train.drop(cols, axis=1).corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_train_corr.drop(df_train_corr.iloc[1::2].index, inplace=True)
df_train_corr_nd = df_train_corr.drop(df_train_corr[df_train_corr['Correlation Coefficient'] == 1.0].index)

df_test_corr = df_test.drop(cols, axis=1).corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_test_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_test_corr.drop(df_test_corr.iloc[1::2].index, inplace=True)
df_test_corr_nd = df_test_corr.drop(df_test_corr[df_test_corr['Correlation Coefficient'] == 1.0].index)


# In[ ]:


# Features correlated with target
df_train_corr_nd[df_train_corr_nd['Feature 1'] == 'SalePrice']


# In[ ]:


# Training set high correlations
df_train_corr_nd.head(10)


# In[ ]:


fig, axs = plt.subplots(nrows=2, figsize=(50, 50))

sns.heatmap(df_train.drop(cols, axis=1).corr().round(2), ax=axs[0], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 12})
sns.heatmap(df_test.drop(cols, axis=1).corr().round(2), ax=axs[1], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 12})

for i in range(2):    
    axs[i].tick_params(axis='x', labelsize=13)
    axs[i].tick_params(axis='y', labelsize=13)
    
axs[0].set_title('Training Set Correlations', size=15)
axs[1].set_title('Test Set Correlations', size=15)

plt.show()


# ### **1.5 Target vs Features**

# #### **1.5.1 Numerical Features**
# Data points of `1stFlrSF`, `BsmtUnfSF`, `GarageArea`, `GrLivArea`, `LotArea`, `LotFrontage` and `TotalBsmtSF` features are not stacked at **0** as much as other numerical features. Those features exist in almost every single house. Fitting the regression line is easier for those features. In addition to that, outliers are very visible in those features.
# 
# Data points of `2ndFlrSF`, `3SsnPorch`, `BsmtFinSF1`, `BsmtFinSF2`, `EnclosedPorch`, `LowQualFinSF`, `MasVnrArea`, `MiscVal`, `OpenPorchSF`, `PoolArea`, `ScreenPorch` and `WoodDeckSF` features are heavily stacked at **0**. Those features are rarer than the previous ones and they don't exist in every house, so they are sparse features. Those sparse features may not be reliable as the previous features when they are used as continuous features, because they are going to introduce bias to the regression function.
# 
# `GarageYrBlt`, `YearBuilt` and `YearRemodAdd` are ordinal features, but a linear relationship can be seen from their plots. Houses with recent dates are more likely to be sold at higher prices.

# In[ ]:


num_features = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2', 
                'BsmtUnfSF', 'EnclosedPorch', 'GarageArea', 'GarageYrBlt', 'GrLivArea', 
                'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 
                'OpenPorchSF', 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF', 
                'YearBuilt', 'YearRemodAdd']

fig, axs = plt.subplots(ncols=2, nrows=11, figsize=(12, 80))
plt.subplots_adjust(right=1.5)
cmap = sns.cubehelix_palette(dark=0.3, light=0.8, as_cmap=True)

for i, feature in enumerate(num_features, 1):    
    plt.subplot(11, 2, i)
    sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', size='SalePrice', palette=cmap, data=df_train)
        
    plt.xlabel('{}'.format(feature), size=15)
    plt.ylabel('SalePrice', size=15, labelpad=12.5)
    
    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
    
    plt.legend(loc='best', prop={'size': 12})
        
plt.show()


# #### **1.5.2 Categorical Features**
# Categorical features are not strongly correlated with `SalePrice`. There are only **2** categorical features that have significant correlation with `SalePrice`, and they are `OverallQual` and `TotRmsAbvGrd`. A linear relationship can easily be seen from their plots. When the number of `OverallQual` and `TotRmsAbvGrd` increases, `SalePrice` tends to increase as well.
# 
# Data points of `MoSold` and `YrSold` are uniformly distributed between classes. Those two features might have the least information about `SalePrice` among other categorical features.
# 
# The other categorical features don't have significant correlation with `SalePrice`. However, values in some of those features have very distinct `SalePrice` maximums, minimums and interquartile ranges. Those features could be useful in tree based algorithms.  

# In[ ]:


cat_features = ['Alley', 'BedroomAbvGr', 'BldgType', 'BsmtCond', 'BsmtExposure', 
                'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 
                'CentralAir', 'Condition1', 'Condition2', 'Electrical', 'ExterCond', 
                'ExterQual', 'Exterior1st', 'Exterior2nd', 'Fence', 'FireplaceQu', 
                'Fireplaces', 'Foundation', 'FullBath', 'Functional', 'GarageCars', 
                'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'HalfBath', 
                'Heating', 'HeatingQC', 'KitchenAbvGr', 'KitchenQual', 'LandContour', 
                'LandSlope', 'LotConfig', 'LotShape', 'MSSubClass', 'MSZoning', 
                'MasVnrType', 'MiscFeature', 'MoSold', 'Neighborhood', 'OverallCond', 
                'OverallQual', 'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 
                'SaleCondition', 'SaleType', 'Street', 'TotRmsAbvGrd', 'Utilities', 'YrSold']

fig, axs = plt.subplots(ncols=2, nrows=28, figsize=(18, 120))
plt.subplots_adjust(right=1.5, top=1.5)

for i, feature in enumerate(cat_features, 1):    
    plt.subplot(28, 2, i)
    sns.swarmplot(x=feature, y='SalePrice', data=df_train, palette='Set3')
        
    plt.xlabel('{}'.format(feature), size=25)
    plt.ylabel('SalePrice', size=25, labelpad=15)
    
    for j in range(2):
        if df_train[feature].value_counts().shape[0] > 10:        
            plt.tick_params(axis='x', labelsize=7)
        else:
            plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
            
plt.show()


# ### **1.6 Training vs Test Set**

# #### **1.6.1 Numerical Features**
# `1stFloorSF`, `2ndFloorSF`, `BsmtFinSF1`, `BsmtUnfSF`, `GarageArea`, `GarageYrBlt`, `GrLivArea`, `LotArea`, `LotFrontage`, `MasVnrArea`, `OpenPorchSF`, `TotalBsmtSF`, `WoodDeckSF`, `YearBuilt` and `YearRemodAdd` features have similar distributions in training set and test set. Models using these features, are less likely to overfit.
# 
# `3SsnPorch`, `BsmtFinSF2`, `EnclosedPorch`, `LowQualFinSF`, `MiscVal`, `PoolArea` and `ScreenPorch` are too noisy. The distributions of those features in training set and test set doesn't match, so they might introduce bias to the models.

# In[ ]:


fig, axs = plt.subplots(ncols=2, nrows=11, figsize=(12, 80))
plt.subplots_adjust(right=1.5)

for i, feature in enumerate(num_features, 1):    
    plt.subplot(11, 2, i)
    sns.kdeplot(df_train[feature], bw='silverman', label='Training Set', shade=True)
    sns.kdeplot(df_test[feature], bw='silverman', label='Test Set', shade=True)
        
    plt.xlabel('{}'.format(feature), size=15)
    
    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
    
    plt.legend(loc='best', prop={'size': 15})
        
plt.show()


# #### **1.6.2 Categorical Features**
# Values in some categorical features exist in one set, but doesn't exist in another set. This problem exists in `BedroomAbvGr`, `Condition2`, `Electrical`, `Exterior1st`, `Exterior2nd`, `Fireplaces`, `FullBath`, `GarageCars`, `GarageQual`, `Heating`, `KitchenAbvGr`, `MSSubClass`, `MiscFeature`, `PoolQC`, `RoofMatl` and `Utilities` features. This is a potential problem because some of the categorical features are going to be one-hot encoded, and because of that process, the feature counts of training set and test set might not match.
# 
# Other categorical feature value counts in training set and test set are close to each other. Those features have similar distributions in both data sets, so they are more reliable than the previous ones.

# In[ ]:


df_train['Dataset'] = 'Training Set'
df_test['Dataset'] = 'Test Set'
df_all = concat_df(df_train, df_test)

fig, axs = plt.subplots(ncols=2, nrows=28, figsize=(18, 120))
plt.subplots_adjust(right=1.5, top=1.5)

for i, feature in enumerate(cat_features, 1):    
    plt.subplot(28, 2, i)
    sns.countplot(x=feature, hue='Dataset', data=df_all, palette='Set2')
        
    plt.xlabel('{}'.format(feature), size=25)
    plt.ylabel('Count', size=25)
    
    for j in range(2):
        if df_train[feature].value_counts().shape[0] > 10:        
            plt.tick_params(axis='x', labelsize=7)
        else:
            plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        
    plt.legend(loc='upper right', prop={'size': 15})
            
plt.show()


# ### **1.7 Conclusion**
# There are features with ambiguous types. `GarageYrBlt`, `MoSold`, `YearBuilt`, `YearRemodAdd` and `YrSold` are date features. Those are numerical features by default and they imply linear relationship. It might be better to use some of them as categorical features. 
# 
# There were **24** features with systematically missing data. Those features are filled with **None** and **0** depending on their types which made them sparse. Converting those features from multi class or continuous to binary, could give better results.
# 
# Target (`SalePrice`) distribution is highly skewed and long tailed because of the outliers. It requires a transformation in order to perform better in models. Dealing with the outliers could also achieve better model performance.
# 
# Many features are strongly correlated with each other and target. This relationship can be used to create new features with feature interaction in order to overcome multicollinearity issue.
# 
# **12** continuous features are heavily stacked at **0**. They are also sparse features. Those features may add bias to the model. Some categorical features are not informative for two reasons. The feature is either too homogenous like `Utilities` feature, or all of the values have the same characteristics like `MoSold` feature. Those features can be conbined with other features or dropped completely.
# 
# There are some numerical feature distributions that are too noisy. Their distributions in training and test set are quite different. They may require grouping to overcome this problem. The same problem occurs for categorical features as well. There are some categorical feature values that doesn't exist in both training set and test set. They need to be grouped with other values.

# ## **2. Feature Engineering**

# ### **2.1 Feature Interactions**
# Created **12** new features in this section. **6** of them are continuous, and **6** of them are categorical. `YearBuiltRemod`, `TotalSF`, `TotalSquare`, `TotalBath`,`TotalPorch` and `OverallRating` are the new continuous features. They are the total number of some related features. `HasPool`, `Has2ndFloor`, `HasGarage`, `HasBsmt` and `HasFireplace` are the new categorical features. They are the binary forms of some rare features. `NewHouse` is a categorical feature to separate houses which were built and sold at the same year.

# In[ ]:


df_all['YearBuiltRemod'] = df_all['YearBuilt'] + df_all['YearRemodAdd']
df_all['TotalSF'] = df_all['TotalBsmtSF'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']
df_all['TotalSquareFootage'] = df_all['BsmtFinSF1'] + df_all['BsmtFinSF2'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']
df_all['TotalBath'] = df_all['FullBath'] + (0.5 * df_all['HalfBath']) + df_all['BsmtFullBath'] + (0.5 * df_all['BsmtHalfBath'])
df_all['TotalPorchSF'] = df_all['OpenPorchSF'] + df_all['3SsnPorch'] + df_all['EnclosedPorch'] + df_all['ScreenPorch'] + df_all['WoodDeckSF']
df_all['OverallRating'] = df_all['OverallQual'] + df_all['OverallCond']

df_all['HasPool'] = df_all['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df_all['Has2ndFloor'] = df_all['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df_all['HasGarage'] = df_all['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_all['HasBsmt'] = df_all['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df_all['HasFireplace'] = df_all['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

df_all['NewHouse'] = 0
idx = df_all[df_all['YrSold'] == df_all['YearBuilt']].index
df_all.loc[idx, 'NewHouse'] = 1


# ### **2.2 Outliers**
# There are two houses larger than **4500** square feet and sold for less than **300000**. They are too large for their prices. Those two houses are dropped because they are affecting the regression coefficient of `GrLivArea` drastically. There is a house with less than **5** `OverallQual`, but sold for more than **200000**. That is an overly paid price for a low quality house, so it is dropped as well.

# In[ ]:


fig = plt.figure(figsize=(12, 6))
cmap = sns.color_palette('Set1', n_colors=10)

sns.scatterplot(x=df_all['GrLivArea'], y='SalePrice', hue='OverallQual', palette=cmap, data=df_all)

plt.xlabel('GrLivArea', size=15)
plt.ylabel('SalePrice', size=15)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12) 
    
plt.title('GrLivArea & OverallQual vs SalePrice', size=15, y=1.05)

plt.show()


# In[ ]:


df_all.drop(df_all[np.logical_and(df_all['OverallQual'] < 5, df_all['SalePrice'] > 200000)].index, inplace=True)
df_all.drop(df_all[np.logical_and(df_all['GrLivArea'] > 4000, df_all['SalePrice'] < 300000)].index, inplace=True)
df_all.reset_index(drop=True, inplace=True)


# ### **2.3 Encoding**

# #### **2.3.1 Label Encoding Ordinal Features**
# There are **45** categorical features, and **23** of them are ordinal. Different set of integers are mapped to those ordinal features depending on their values.
# 
# **3** of the **23** ordinal features are dropped because they are not informative. Those features are `Street`, `Utilities` and `PoolQC`. The `PoolQC` and `Street` features have only different values in **10** houses, and `Utilities` feature has only **1** different value in one house. That's why those features don't provide any useful information.
# 
# There are also partial ordinal features that are not label encoded because all of their values are not ordered.

# In[ ]:


bsmtcond_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4}
bsmtexposure_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
bsmtfintype_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
bsmtqual_map = {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
centralair_map = {'Y': 1, 'N': 0}
extercond_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
exterqual_map = {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
fireplacequ_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
functional_map = {'Typ': 0, 'Min1': 1, 'Min2': 1, 'Mod': 2, 'Maj1': 3, 'Maj2': 3, 'Sev': 4}
garagecond_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
garagefinish_map = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
garagequal_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
heatingqc_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
kitchenqual_map = {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
landslope_map = {'Gtl': 1, 'Mod': 2, 'Sev': 3}
lotshape_map = {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}
paveddrive_map = {'N': 0, 'P': 1, 'Y': 2}

df_all['BsmtCond'] = df_all['BsmtCond'].map(bsmtcond_map)
df_all['BsmtExposure'] = df_all['BsmtExposure'].map(bsmtexposure_map)
df_all['BsmtFinType1'] = df_all['BsmtFinType1'].map(bsmtfintype_map)
df_all['BsmtFinType2'] = df_all['BsmtFinType2'].map(bsmtfintype_map)
df_all['BsmtQual'] = df_all['BsmtQual'].map(bsmtqual_map)
df_all['CentralAir'] = df_all['CentralAir'].map(centralair_map)
df_all['ExterCond'] = df_all['ExterCond'].map(extercond_map)
df_all['ExterQual'] = df_all['ExterQual'].map(exterqual_map)
df_all['FireplaceQu'] = df_all['FireplaceQu'].map(fireplacequ_map)
df_all['Functional'] = df_all['Functional'].map(functional_map)
df_all['GarageCond'] = df_all['GarageCond'].map(garagecond_map)
df_all['GarageFinish'] = df_all['GarageFinish'].map(garagefinish_map)
df_all['GarageQual'] = df_all['GarageQual'].map(garagequal_map)
df_all['HeatingQC'] = df_all['HeatingQC'].map(heatingqc_map)
df_all['KitchenQual'] = df_all['KitchenQual'].map(kitchenqual_map)
df_all['LandSlope'] = df_all['LandSlope'].map(landslope_map)
df_all['LotShape'] = df_all['LotShape'].map(lotshape_map)
df_all['PavedDrive'] = df_all['PavedDrive'].map(paveddrive_map)

df_all.drop(columns=['Street', 'Utilities', 'PoolQC'], inplace=True)


# #### **2.3.2 One-Hot Encoding Nominal Features**
# The rest of the categorical features are nominal, and there are **25** of them. Those features are one-hot encoded because there is no order in their values. Partial ordinal features are also one-hot encoded along with nominal features.

# In[ ]:


nominal_features = ['Alley', 'BldgType', 'Condition1', 'Condition2', 'Electrical', 
                    'Exterior1st', 'Exterior2nd', 'Fence', 'Foundation', 'GarageType', 
                    'Heating', 'HouseStyle', 'LandContour', 'LotConfig', 'MSSubClass',
                    'MSZoning', 'MasVnrType', 'MiscFeature', 'MoSold', 'Neighborhood',
                    'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'YrSold']

encoded_features = []

for feature in nominal_features:
    encoded_df = pd.get_dummies(df_all[feature])
    n = df_all[feature].nunique()
    encoded_df.columns = ['{}_{}'.format(feature, col) for col in encoded_df.columns]
    encoded_features.append(encoded_df)

df_all = pd.concat([df_all, *encoded_features], axis=1)
df_all.drop(columns=nominal_features, inplace=True)


# ### **2.4 Dealing with the Skewness**
# The skewed and long tailed distribution of the `SalePrice` is solved by applying **log(1 + x)** transformation. This transformation reduced skewness from **1.88** to **0.12** and reduced kurtosis from **6.53** to **0.80**. Target is normally distributed after this transformation. Probability is a straight line rather than a convex curve, which is an indicator of the reduced skewness.

# In[ ]:


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
plt.subplots_adjust(top=1.5, right=1.5)

sns.distplot(df_all['SalePrice'].dropna(), hist=True, ax=axs[0][0])
probplot(df_all['SalePrice'].dropna(), plot=axs[0][1])

df_all['SalePrice'] = np.log1p(df_all['SalePrice'])

sns.distplot(df_all['SalePrice'].dropna(), hist=True, ax=axs[1][0])
probplot(df_all['SalePrice'].dropna(), plot=axs[1][1])

axs[0][0].set_xlabel('Sale Price', size=20, labelpad=12.5)
axs[1][0].set_xlabel('Sale Price', size=20, labelpad=12.5)
axs[0][1].set_xlabel('Theoretical Quantiles', size=20, labelpad=12.5)
axs[0][1].set_ylabel('Ordered Values', size=20)
axs[1][1].set_xlabel('Theoretical Quantiles', size=20, labelpad=12.5)
axs[1][1].set_ylabel('Ordered Values', size=20)

for i in range(2):
    for j in range(2):
        axs[i][j].tick_params(axis='x', labelsize=15)
        axs[i][j].tick_params(axis='y', labelsize=15)
        
axs[0][0].set_title('Distribution of Sale Price', size=25, y=1.05)
axs[0][1].set_title('Sale Price Probability Plot', size=25, y=1.05)
axs[1][0].set_title('Distribution of Sale Price After log1p Transformation', size=25, y=1.05)
axs[1][1].set_title('Sale Price Probability Plot After log1p Transformation', size=25, y=1.05)

plt.show()

print('Training Set SalePrice Skew: {}'.format(df_all['SalePrice'].skew()))
print('Training Set SalePrice Kurtosis: {}'.format(df_all['SalePrice'].kurt()))


# The other highly skewed features are also transformed, but **boxcox1p** transformation is used for those features. **boxcox1p** is defined as $((1 + x)^{\lambda} - 1)$ if $\lambda$ is not **0**, and $log(1 + x)$ if $\lambda$ is **0**.
# 
# **0.5** is used as a skewness threshold, and transformation is applied to features which have higher skew than the threshold. However, this doesn't work on every feature because some of them are sparse.

# In[ ]:


cont_features = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2',
                 'BsmtUnfSF', 'EnclosedPorch', 'GarageArea', 'GrLivArea', 'LotArea', 
                 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 
                 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF']

skewed_features = {feature: df_all[feature].skew() for feature in cont_features if df_all[feature].skew() >= 0.5}
transformed_skews = {}

for feature in skewed_features.keys():
    df_all[feature] = boxcox1p(df_all[feature], boxcox_normmax(df_all[feature] + 1))
    transformed_skews[feature] = df_all[feature].skew()
    
df_skew = pd.DataFrame(index=skewed_features.keys(), columns=['Skew', 'Skew after boxcox1p'])
df_skew['Skew'] = skewed_features.values()
df_skew['Skew after boxcox1p'] = transformed_skews.values()

fig = plt.figure(figsize=(24, 12))

sns.pointplot(x=df_skew.index, y='Skew', data=df_skew, markers=['o'], linestyles=['-'])
sns.pointplot(x=df_skew.index, y='Skew after boxcox1p', data=df_skew, markers=['x'], linestyles=['--'], color='#bb3f3f')

plt.xlabel('Skewed Features', size=20, labelpad=12.5)
plt.ylabel('Skewness', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=11)
plt.tick_params(axis='y', labelsize=15)

plt.title('Skewed Features Before and After boxcox1p Transformation', size=20)

plt.show()


# ## **3. Models**

# ### **3.1 Feature Selection**
# Sparse features are dropped because they are tend to be ignored by tree algorithms. **99.94** is the threshold for the sparse features. If **99.94%** of the values of a feature are zeros, the feature is dropped. Other useless features like `Id` and `Dataset` are also dropped. Finally, `X_train`, `y_train` and `X_test` are separated and ready for the machine learning models.

# In[ ]:


sparse = []

for feature in df_all.columns:
    counts = df_all[feature].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(df_all) * 100 > 99.94:
        sparse.append(feature)
        
df_all.drop(columns=sparse, inplace=True)

df_train, df_test = df_all.loc[:1456], df_all.loc[1457:].drop(['SalePrice'], axis=1)
drop_cols = ['Id', 'Dataset']
X_train = df_train.drop(columns=drop_cols + ['SalePrice']).values
y_train = df_train['SalePrice'].values
X_test = df_test.drop(columns=drop_cols).values

print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('X_test shape: {}'.format(X_test.shape))


# ### **3.2 Cost Function & Cross Validation**
# `rmse` calculates the root of **mean squared error**, and since the target variable is already at log space, this function calculates **root mean squared log error** which is the competition score metric.
# 
# `cv_rmse` has to be implemented with `cross_val_score` function, which returns a vector of scores of the specified cost function (`rmse`) for every fold. Square root of that vector is the `rmse` of every fold. A **10** fold shuffled cross validation is used for validation sets.

# In[ ]:


def rmse(y_train, y_pred):
     return np.sqrt(mean_squared_error(y_train, y_pred))

def cv_rmse(model, X=X_train, y=y_train):    
    return np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf))

K = 10
kf = KFold(n_splits=K, shuffle=True, random_state=SEED)


# ### **3.3 Models & Stacking**
# Created **8** models and one of them is a stack of those models. 
# * `RidgeCV`, `LassoCV` and `ElasticNetCV` are linear models with built-in cross validation
# * `SVR` is a linear support vector machine algorithm
# * `GradientBoostingRegressor`, `LGBMRegressor` and `XGBRegressor` are tree based regression model
# * Lastly, `StackingCVRegressor` is the stack of those models
# 
# 
# **Stacking** is an ensemble learning technique to combine multiple regression models via a meta-regressor to give improved prediction accuracy. In the standard stacking procedure, the first-level regressors are fit to the same training set that is used prepare the inputs for the second-level regressor, which may lead to overfitting. The `StackingCVRegressor`, however, uses the concept of **out-of-fold predictions** the dataset is split into k folds, and in k successive rounds, k-1 folds are used to fit the first level regressor. In each round, the first-level regressors are then applied to the remaining 1 subset that was not used for model fitting in each iteration. The resulting predictions are then stacked and provided -- as input data -- to the second-level regressor. After the training of the `StackingCVRegressor`, the first-level regressors are fit to the entire dataset for optimal predicitons.
# 
# All of the models are stacked in `StackingCVRegressor`, and `XGBRegressor` is used as the meta-regressor. 
# 
# `use_features_in_secondary` parameter is set to **True**, which means that the meta-regressor will be trained both on the predictions of the original regressors and the original dataset. (If it is set to **False**, the meta-regressor will be trained only on the predictions of the original regressors.)
# 
# ![alt](https://i.ibb.co/71k3JR0/stacking-cv-regressor-overview.png)

# In[ ]:


ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=np.arange(14.5, 15.6, 0.1), cv=kf))
lasso = make_pipeline(RobustScaler(), LassoCV(alphas=np.arange(0.0001, 0.0009, 0.0001), random_state=SEED, cv=kf))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(alphas=np.arange(0.0001, 0.0008, 0.0001), l1_ratio=np.arange(0.8, 1, 0.025), cv=kf))
svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=SEED)
lgbmr = LGBMRegressor(objective='regression', 
                      num_leaves=4,
                      learning_rate=0.01, 
                      n_estimators=5000,
                      max_bin=200, 
                      bagging_fraction=0.75,
                      bagging_freq=5, 
                      bagging_seed=SEED,
                      feature_fraction=0.2,
                      feature_fraction_seed=SEED,
                      verbose=0)
xgbr = XGBRegressor(learning_rate=0.01,
                    n_estimators=3500,
                    max_depth=3,
                    gamma=0.001,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    objective='reg:squarederror',
                    nthread=-1,
                    seed=SEED,
                    reg_alpha=0.0001)
stack = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, svr, gbr, lgbmr, xgbr), meta_regressor=xgbr, use_features_in_secondary=True)

models = {'RidgeCV': ridge,
          'LassoCV': lasso, 
          'ElasticNetCV': elasticnet,
          'SupportVectorRegressor': svr, 
          'GradientBoostingRegressor': gbr, 
          'LightGBMRegressor': lgbmr, 
          'XGBoostRegressor': xgbr, 
          'StackingCVRegressor': stack}
predictions = {}
scores = {}

for name, model in models.items():
    start = datetime.now()
    print('[{}] Running {}'.format(start, name))
    
    model.fit(X_train, y_train)
    predictions[name] = np.expm1(model.predict(X_train))
    
    score = cv_rmse(model, X=X_train, y=y_train)
    scores[name] = (score.mean(), score.std())
    
    end = datetime.now()
    
    print('[{}] Finished Running {} in {:.2f}s'.format(end, name, (end - start).total_seconds()))
    print('[{}] {} Mean RMSE: {:.6f} / Std: {:.6f}\n'.format(datetime.now(), name, scores[name][0], scores[name][1]))


# ### **3.4 Evaluation & Blending**
# All of the models individually achieved scores between **0.10** and **0.12**, but when the predictions of those models are blended, they get **0.058**. That's because those models are actually overfitting to certain degree. They are very good at predicting a subset of houses, and they fail at predicting the rest of the dataset. When their predictions are blended, they complement each other. 

# In[ ]:


def blend_predict(X):
    return ((0.1 * elasticnet.predict(X)) + 
            (0.05 * lasso.predict(X)) +
            (0.1 * ridge.predict(X)) +
            (0.1 * svr.predict(X)) +
            (0.1 * gbr.predict(X)) +
            (0.15 * xgbr.predict(X)) +
            (0.1 * lgbmr.predict(X)) +
            (0.3 * stack.predict(X)))

blended_score = rmse(y_train, blend_predict(X_train))
print('Blended Prediction RMSE: {}'.format(blended_score))


# In[ ]:


fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(18, 36))
plt.subplots_adjust(top=1.5, right=1.5)

for i, model in enumerate(models, 1):
    plt.subplot(4, 2, i)
    plt.scatter(predictions[model], np.expm1(y_train))
    plt.plot([0, 800000], [0, 800000], '--r')

    plt.xlabel('{} Predictions (y_pred)'.format(model), size=20)
    plt.ylabel('Real Values (y_train)', size=20)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    plt.title('{} Predictions vs Real Values'.format(model), size=25)
    plt.text(0, 700000, 'Mean RMSE: {:.6f} / Std: {:.6f}'.format(scores[model][0], scores[model][1]), fontsize=25)

plt.show()


# In[ ]:


scores['Blender'] = (blended_score, 0)

fig = plt.figure(figsize=(24, 12))

ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(scores.values()):
    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.xlabel('Model', size=20, labelpad=12.5)
plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=11)
plt.tick_params(axis='y', labelsize=12.5)

plt.title('Scores of Models', size=20)

plt.show()


# ### **3.5 Submission**

# In[ ]:


submission_df = pd.DataFrame(columns=['Id', 'SalePrice'])
submission_df['Id'] = df_test['Id']
submission_df['SalePrice'] = np.expm1(blend_predict(X_test))
submission_df.to_csv('submissions.csv', header=True, index=False)
submission_df.head(10)

