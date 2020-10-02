#!/usr/bin/env python
# coding: utf-8

# # House Prices: Advanced Regression Techniques

# # Introduction
# 
# This kernel addresses the **House Prices: Advanced Regression Techniques** competition.
# 
# **Focus:**
# - Feature analysis with respect to missing values
# - Comparison of different algorithms:
#  - **XGBoost**
#  - **Gradient Boosting Regressor**
#  - **Random Forest Regressor**
#  - **LASSO**
# 
# The kernel consists of 4 sections:
# - **Exploratory Data Analysis**
# - **Data Preprocessing**
# - **Model Selection**
# - **Model Diagnostics**

# **Import libraries:**

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
import xgboost
import warnings

register_matplotlib_converters()
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 2500, 'display.max_rows', 2500, 'display.width', None)


# **Read the data:**

# In[ ]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train.name = 'Training set'

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test.name = 'Test set'

df = pd.concat([train, test]).reset_index(drop=True)
df.name = 'Total database'


# **Define functions:**

# In[ ]:


def correlation(value):
    correlation = df.drop(['Id'], axis=1).apply(lambda x: x.factorize()[0]).corr().abs().unstack().sort_values(kind='quicksort', ascending=False).drop_duplicates(keep='first')
    correlation = correlation.reset_index().rename(columns={0: 'Correlation'})
    correlation = correlation[correlation['level_0'].str.contains(value) |
                              correlation['level_1'].str.contains(value)]
    return correlation[:5]

def countplot(feature, data):
    sns.set(style='darkgrid')
    sns.countplot(data[feature])
    plt.title('{0} ({1})'.format(feature, data.name))
    if len(df.groupby(['Exterior1st']).sum()) > 6:
        plt.xticks(rotation=45, ha='right')
    plt.show()

def catplot(feature, data):
    sns.catplot(x=feature, y='SalePrice', data=data, kind="bar")
    plt.title('Effect of {} on SalePrice'.format(feature)), plt.show()


# # 1. Exploratory Data Analysis
# 
# ## 1.1 Overview

# In[ ]:


print('Training set:', train.shape)
print('Test set:', test.shape)
print('\nColumns:\n', list(df.columns))

# Data types
print('\nData types:\n{}'.format(df.dtypes))

# Descriptive statistics
df.describe()


# The following column ***MSSubClass*** has a numerical data type. It needs to be converted to categorical columns. Otherwise the model would compare the values which may yield poor results.

# In[ ]:


for dataset in (train, test, df):
    dataset['MSSubClass'] = dataset['MSSubClass'].astype('str')


# ## 1.2 Target value *SalePrice*
# 
# There are no missing values in the training set for *SalePrice*.

# In[ ]:


print('Missing values in training set: {}'.format(train['SalePrice'].isna().sum()))

train['SalePrice'].describe()


# 
# The target value looks normally distributed.

# In[ ]:


sns.set(style='darkgrid')
sns.distplot(df['SalePrice'], 20),
plt.xticks(rotation=45, ha='right')
plt.show()


# In[ ]:


sns.set(style='darkgrid')
sns.boxplot(x=train['SalePrice'])
plt.title('Boxplot SalePrice', fontsize=12), plt.xlabel('SalePrice', fontsize=10), plt.xticks(fontsize=10, rotation=90)
plt.show()


# There are some outliers, as the boxplot shows, but the prices of the most expensive houses seem not to be unrealistic (*OverallQual*=10, *CentralAir*=Y, *FullBath*=3, *KitchenQual*=Ex, *TotRmsAbvGrd*=10 etc.).

# In[ ]:


df[df['SalePrice'] > 700000]


# ## 1.3 Correlations
# 
# The target value *SalePrice* is highly correlated with some of the feature which indicates that the housing prices highly depend on these features. In addition, there are considerable correlations across some of the features.

# In[ ]:


highest_correlation_target = df.drop(['Id'], axis=1).corr().abs().unstack().sort_values(kind='quicksort', ascending=False)#.drop_duplicates(keep='first')
highest_correlation_target = highest_correlation_target.reset_index().rename(columns={0: 'Correlation'})
highest_correlation_target = highest_correlation_target[highest_correlation_target['level_0'].str.contains('SalePrice') |
                                                        highest_correlation_target['level_1'].str.contains('SalePrice')]
highest_correlation_target = highest_correlation_target[highest_correlation_target['Correlation'] < 1]
highest_correlation_target.drop_duplicates(subset='Correlation')[:10]


# In[ ]:


correlation_matrix = df.drop(['Id'], axis=1).corr()#.drop_duplicates(keep='first')

plt.figure(figsize=(12,12))
sns.set(font_scale=0.75)
ax = sns.heatmap(correlation_matrix, vmin=-1, vmax=1, center=0, linewidths=0.5, cmap='coolwarm', square=True, annot=False)


# The plots below illustrate the effect of the highest correlating features *OverallQual* and *GrLivArea* on the *SalePrice*. These features are expected to be important for the sales predictions.

# **1.3.1 Effect of selected numerical features on *SalePrice***
# 
# **a) Feature *OverallQual*:**

# In[ ]:


catplot('OverallQual', train)


# **b) Feature *GrLivArea*:**

# In[ ]:


sns.lmplot(x="GrLivArea", y="SalePrice", data=train)
plt.show()


# There are 2 outliers in *GrLivArea* with a large *GrLivArea* but a low *SalePrice*. They will be removed.

# In[ ]:


train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)#.reset_index(drop=False)
df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index)#.reset_index(drop=False)


# **c) Features *YearBuilt* & *TotRmsAbvGrd***
# 
# The following relational plots captures the effects of *YearBuilt* and *TotRmsAbvGrd* on *SalePrice* in one figure.

# In[ ]:


# Relplot
sns.relplot(x='YearBuilt', y='SalePrice', hue='TotRmsAbvGrd', data=train)
plt.show()


# **1.3.2 Effect of selected categorical features on *SalePrice***
# 
# For most of the categorical features, there are considerable differences in the *SalePrice* among the groups. They are expected to have predictive power for the target as well. This is illustrated based on the features *Neighborhood* and *CentralAir*.
# 
# **a) Feature *Neighborhood*:**

# In[ ]:


mean_sale_price_neighborhood = train.groupby('Neighborhood')['SalePrice'].mean().sort_values()

sns.pointplot(x =mean_sale_price_neighborhood.index, y =mean_sale_price_neighborhood.values, data=train,
              order=mean_sale_price_neighborhood.index)
plt.xticks(rotation=45)
plt.show()


# **b) Feature *CentralAir*:**
# 
# *CentralAir* has a positive effect on the target value. This is most relevant when it comes to predicting cheaper houses. 

# In[ ]:


central_air = train.groupby(['CentralAir'])['SalePrice'].mean()
central_air = central_air.sort_index(ascending=False)

plt.figure()
sns.barplot(x=central_air.index, y=central_air.values)
plt.title('Effect of CentralAir on SalePrice')
plt.show()

sns.set(style='darkgrid')
sns.countplot(x=pd.qcut(train['SalePrice'], 5), hue='CentralAir', data=train)
plt.xticks(ha='right', rotation=45)
plt.show()


# # 2. Data Preprocessing

# ## 2.1 Missing values
# 
# Columns with missing values will be analyzed and cleaned in detail. The outputs below provide a first overview of missing values.

# In[ ]:


missing = df.isna().sum()
missing = missing[missing.values != 0].sort_values(ascending=False)

plt.figure(figsize=(14,8))
sns.barplot(missing.index, missing.values)
plt.xticks(rotation=90), plt.ylabel('Missing values'), plt.title('Missing values by feature\n(total dataset: {} observations)'.format(len(df)))
for i, v in enumerate(np.around(np.array(missing.values), 4)):
    plt.text(i, v+20, str('%.0f' % v), ha='center', fontsize=8)
plt.show()


# - Some columns have lots of missing values others have only a few missing values.
# - The columns with missing values might simply be dropped. By doing so, information would get lost though. Thereby, every column with a missing value will be considered in detail.
# - In the housing price data, missing values are not completely meaningless. In *PoolQC* and many other columns, a missing value indicates that this item, here a pool, is not avaialable in this house. An additional group 'Not available' has been created.
# - This should only be done for categorical features (dtype 'object'). Missing values in numerical columns should be replace with a number.

# In[ ]:


for dataset in (train, test, df):
    # Categorial features
    for column in ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']:
        dataset[column] = dataset[column].fillna('Not available')
    
    # Numerical features
    for column in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']:
        dataset[column] = dataset[column].fillna(0)


# **2.1.1 Feature *GarageYrBlt***
# 
# First of all, the scatterplot reveals an incorrect value in *GarageYrBlt*. According to this, the garage was/will be built in 2207 which does not make sense. This house was built in 2006 and remodeled in 2007. Therefore, 2207 is expected to be a typing error that should mean 2007. The value has been replaced.

# In[ ]:


sns.set(style='darkgrid')
sns.scatterplot(x='GarageYrBlt', y='YearBuilt', data=df)
plt.show()

correlation('GarageYrBlt')


# In[ ]:


df.loc[:, ('Id', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd')][df['GarageYrBlt'] == 2207]


# In[ ]:


df['GarageYrBlt'] = df['GarageYrBlt'].replace({2207: 2007})

sns.set(style='darkgrid')
sns.scatterplot(x='GarageYrBlt', y='YearBuilt', data=df)
plt.show()


# There are 159 missing values in 'GarageYrBlt' which indicate that 159 houses have no garage. By replacing these values with 'Not available', the column would become data type 'object'. This could be critical when it comes to feature scaling and one-hot encoding. NaN values have been replaced with 0. Thereby, the column remains data type 'float' and the feature has no effect on houses without garage.

# In[ ]:


print('Missing values:', len(df[df['GarageYrBlt'].isna()]))


# In[ ]:


for dataset in (train, test, df):
    dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(0)


# **2.1.2 Feature *GarageCars***
# 
# There is one house with a missing value in *GarageCars*. It has no garage and thus no car capacity so that the NaN is replaced with 0.

# In[ ]:


df.loc[:, ('Id', 'GarageCars', 'GarageQual', 'GarageCond')][df['GarageCars'].isna()]


# In[ ]:


for dataset in (train, test, df):
    dataset['GarageCars'] = dataset['GarageCars'].fillna(0)


# **2.1.3 Feature *GarageArea***
# 
# There is one NaN in the numerical column *GarageArea*. The missing value addresses a house without garage. Therefore, the NaN can be replaced with 0.

# In[ ]:


df.loc[:, ('Id', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond')][df['GarageArea'].isna()]


# In[ ]:


for dataset in (train, test, df):
    dataset['GarageArea'] = dataset['GarageArea'].fillna(0)


# **2.1.4 Features *BsmtQual*, *BsmtCond*, *BsmtExposure*, *BsmtFinType1*, *BsmtFinType2***
# 
# The missing value overview has shown that houses without basement have not consistently 'Not available' in the 'object' columns addressing the basement (*BsmtQual*, *BsmtCond*, *BsmtExposure*, *BsmtFinType1*, *BsmtFinType2*). See output below.

# In[ ]:


for feature in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    print('{0}: {1}'.format(feature, len(df[df[feature] == 'Not available'])))


# In[ ]:


basement = df
basement['Count'] = basement['BsmtQual'].str.count('Not available') + basement['BsmtCond'].str.count('Not available') + basement['BsmtExposure'].str.count('Not available') + basement['BsmtFinType1'].str.count('Not available') + basement['BsmtFinType2'].str.count('Not available')
basement.loc[:, ('Id', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'TotalBsmtSF')][(basement['Count'] < 5) & (basement['Count'] > 0)]


# It becomes clear that there are incorrect 'Not available' values which need to be cleaned.
# The following columns need to be considered in detail:
# 
# a) *BsmtQual*
# 
# b) *BsmtCond*
# 
# c) *BsmtExposure*
# 
# d) *BsmtFinType2*

# **a) Feature *BsmtQual***
# 
# This feature correlates most with *OverallQual*. The two 'Not available' values that should be cleaned might be replaced based on the *OverallQual* values.

# In[ ]:


correlation('BsmtQual')


# Both houses have an *OverallQual* of 4 while the mean *OverallQual* is round about 6. *BsmtQual* is thus not replaced by 'TA' (='Typical') which is supposed to indicate the default value, but by 'FA' (='Fair') which is one class below 'TA'.

# In[ ]:


df.loc[2217:2218, ('BsmtQual', 'OverallQual')]


# In[ ]:


print('Mean OverallQual: {}'.format(round(df['OverallQual'].mean(), 2)))

for dataset in (train, test, df):
    for value in (2218, 2219):
        dataset.loc[dataset['Id'] == value, 'BsmtQual'] = 'TA'


# **b) Feature *BsmtCond***
# 
# Results have shown earlier that there is 'Not available' in the *BsmtCond* of three houses, although these houses have a basement. *BsmtCond* correlates most with *BsmtQual*. Since both features have the same values, the missing values in *BsmtCond* are thus replaced by the values for *BsmtQual*.

# In[ ]:


correlation('BsmtCond')


# In[ ]:


for dataset in (train, test, df):
    dataset['BsmtCond'] = np.where((dataset['BsmtCond'] == 'Not available') &
                                   (dataset['BsmtQual'] != 'Not available'), dataset['BsmtQual'], dataset['BsmtCond'])
        
df.loc[(2040, 2185, 2524), ('BsmtQual', 'BsmtCond')]


# **c) Feature *BsmtExposure***
# 
# *BsmtExposure* correlates most with *HouseStyle* and the houses with missing values have the style '2Story' or '1Story'.

# In[ ]:


correlation('BsmtExposure')


# In[ ]:


df.loc[(948, 1487, 2348), ('BsmtExposure', 'HouseStyle')]


# Most houses with the style '2Story' and '1Story' have no basement exposure. The *BsmtExposure* value for the 3 houses was changed from 'Not available' (since the houses clearly have a basement) to 'No'. 

# In[ ]:


correlation('BsmtExposure')
sns.catplot(x='HouseStyle', hue='BsmtExposure', data=df, kind='count')
plt.show()


# In[ ]:


for dataset in (train, test, df):
    for value in (949, 1488, 2349):
        dataset.loc[dataset['Id'] == value, 'BsmtExposure'] = 'No'


# **d) Feature *BsmtFinType2***
# 
# *BsmtFinType2* shows the highest correlation with *BsmtFinSF2*. The house with the incorrect 'Not available' in *BsmtFinType2* has 479.0 type 2 finished square feet in the basement.

# In[ ]:


correlation('BsmtFinType2')


# In[ ]:


df.loc[[332], ('Id', 'BsmtFinType2', 'BsmtFinSF2')]


# There are 107 houses with similar BsmtFinSF2 size, as the bar chart below shows.

# In[ ]:


df['BsmtFinSF2Grouped'] = pd.cut(df['BsmtFinSF2'], 5)

sns.catplot(x='BsmtFinSF2Grouped', data=df, kind='count')
plt.xticks(rotation=45, ha='right')
plt.title('Number of houses with BsmtFinSF2 (305.2, 610.4)')
for i, v in enumerate(np.array(df.groupby('BsmtFinSF2Grouped')['BsmtFinType2'].count())):
    plt.text(i, v+20, v, ha='center', fontsize=10) 
plt.show()


# From the 6 *BsmtFinType1* groups, 'Rec' which indicates an average rec room is the most common one (see pie chart below). The missing value has been replaced with 'Rec'.

# In[ ]:


df_new = df[df['BsmtFinSF2Grouped'] == pd.Interval(305.2, 610.4)]

percentage = 100 * df_new['BsmtFinType2'].value_counts() / df_new['BsmtFinType2'].value_counts().sum()
plt.figure(figsize=(10,6))
plt.pie(percentage)
plt.legend(['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(percentage.index, percentage)], loc='best', fontsize=10, frameon=True)
plt.title('Distribution of BsmtFinType2 in the BsmtFinSF2 group (305.2, 610.4)')
plt.show()

for dataset in (train, test, df):
    dataset.loc[dataset['Id'] == 333, 'BsmtFinType2'] = 'Rec'


# The final output shows that there are consistently 79 houses without a basement after the basement columns have been cleaned.

# In[ ]:


for feature in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    print('{0}: {1}'.format(feature, len(df[df[feature] == 'Not available'])))


# **2.1.5 Feature *MSZoning***

# In[ ]:


df.name = 'Total database'
countplot('MSZoning', df)

print('Missing values in MSZoning: {}'.format(df['MSZoning'].isna().sum()))

correlation('MSZoning')


# *MSZoning* correlates most with *Alley*. The houses with missing values in *MSZoning* have no alley.

# In[ ]:


df.loc[:, ('Id', 'MSZoning', 'Alley')][df['MSZoning'].isna()]


# Most of the houses without alley turned out to have 'RL' in *MSZoning*. The missing values have thus been replaced by 'RL'.

# In[ ]:


sns.catplot(x='Alley', hue='MSZoning', data=df, kind='count')
plt.title('MSZoning for Alley values')
plt.show()

for dataset in (train, test, df):
    dataset['MSZoning'] = dataset['MSZoning'].fillna('RL')


# **2.1.6 Feature *Utilities***
# 
# There are 2 missing values in *Utilities*. The column has 2916 observations with 'AllPub' and 1 observation with 'NoSeWa'. The missing value has been replaced by 'AllPub'.

# In[ ]:


print('Missing values in Utilities: {}'.format(df['Utilities'].isna().sum()))

countplot('Utilities', df)

print(df['Utilities'].value_counts())

for dataset in (train, test, df):
    dataset['Utilities'] = dataset['Utilities'].fillna('AllPub')


# **2.1.7 Features *Exterior1st* & *Exterior2nd***
# 
# There is 1 missing value in 'Exterior1st' and 'Exterior2nd'.

# In[ ]:


print('Missing values in Exterior1st: {}'.format(df['Exterior1st'].isna().sum()))
print('Missing values in Exterior2nd: {}'.format(df['Exterior2nd'].isna().sum()))

fig, ax  = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
sns.countplot(df['Exterior1st'], ax=ax[0])
sns.countplot(df['Exterior2nd'], ax=ax[1])
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=45)
fig.show()


# The features are highly correlated and *Exterior2nd* correlates most with *Foundation*.

# In[ ]:


correlation('Exterior1st|Exterior2nd')


# The missing value has a *Foundation* of 'PConc' ('Poured Contrete').

# In[ ]:


df.loc[:, ('Id', 'Exterior1st', 'Exterior2nd', 'Foundation')][df['Exterior1st'].isnull()]


# In the group of 'PConc', most houses have Exterior 'VinylSd'. The missing value has thus been replaced with the most common value 'VinylSd'.
# 

# In[ ]:


sns.catplot(x='Foundation', data=df, hue='Exterior2nd', kind='count')
plt.xticks(rotation=45, ha='right')
plt.show()

for dataset in (train, test, df):
    dataset['Exterior1st'] = dataset['Exterior1st'].fillna('VinylSd')
    dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna('VinylSd')


# **2.1.8 Features *MasVnrType* & *MasVnrArea***
# 
# The features *MasVnrType* & *MasVnrArea* indicate the masonry veneer type and area. There are 23 and 24 missing values. The values are not trivial according to the countplot.

# In[ ]:


print('Missing values in MasVnrType: {}'.format(df['MasVnrType'].isna().sum()))
print('Missing values in MasVnrArea: {}'.format(df['MasVnrArea'].isna().sum()))

sns.set(style='darkgrid')
sns.countplot(df['MasVnrType'])
plt.title('{0} ({1})'.format('MasVnrType', df.name))
plt.show()


# The masonry veneer may depend on *YearBuilt*. The numeric feature *YearBuilt* has been grouped into 10 bins, as the plot shows.

# In[ ]:


df['YearBuiltGrouped'] = pd.cut(df['YearBuilt'], 10)
countplot('YearBuiltGrouped', df)


# Older houses have mostly 'None' value while newer houses have different *MasVnrType*. The missing values only address newer houses so that they cannot be replaced reliably based on *YearBuilt*.

# In[ ]:


sns.set(style='darkgrid')
sns.catplot(x='YearBuiltGrouped', data=df, hue='MasVnrType', kind='count')
plt.xticks(rotation=45, ha='right')
plt.title('MasVnrType per YearBuiltGrouped')
plt.show()

missing_MasVnrType = df[df["MasVnrType"].isnull()]
sns.set(style='darkgrid')
sns.countplot(missing_MasVnrType['YearBuiltGrouped'])
plt.title('YearBuiltGrouped for missing values in MasVnrType')
plt.xticks(rotation=45, ha='right')
plt.show()


# Correlations may yield better insights. *MasVnrArea* has correlates most with *Fireplaces*.

# In[ ]:


correlation('MasVnrArea')


# Houses with missing values in *MasVnrType* have between 0 and 2 fire places. There is no clear relationship also between *MasVnrType* and *Fireplaces*.

# In[ ]:


sns.catplot(x='Fireplaces', data=df, hue='MasVnrType', kind='count')
plt.xticks(rotation=45, ha='right')
plt.show()

df.loc[:, ('Id', 'MasVnrType', 'Fireplaces')][df['MasVnrType'].isnull()]


# There is no way to reliably replace the NaN values in *MasVnrType* and *MasVnrArea*. Thus, both columns are dropped from the database.

# In[ ]:


train = train.drop(['MasVnrType', 'MasVnrArea'], axis=1)
test = test.drop(['MasVnrType', 'MasVnrArea'], axis=1)
df = df.drop(['MasVnrType', 'MasVnrArea'], axis=1)


# **2.1.9 Feature *LotFrontage***
# 
# There are 486 missing values in *LotFrontage*. Due to the large number of missing values the feature might simply be dropped. The feature has been grouped to get better insights into how it is distributed. There are no irregularities in its distribution. *LotFrontage* correlates most with *BldgType*.

# In[ ]:


print('Missing values in LotFrontage: {}'.format(df['LotFrontage'].isna().sum()))

df['LotFrontageGrouped'] = pd.cut(df['LotFrontage'], 10)

sns.countplot(df['LotFrontageGrouped'])
plt.xticks(rotation=45, ha='right')
plt.show()

correlation('LotFrontage')


# The missing values in *LotFrontage* have different building types. Since *LotFrontage* is a numeric feature, it has been replaced with the mean *LotFrontage* for the respective *BldgType*.

# In[ ]:


missing_LotFrontage = df.loc[:, ('Id', 'LotFrontage', 'BldgType')][df['LotFrontage'].isna()]
sns.countplot(missing_LotFrontage['BldgType'])
plt.title('Missing values in LotFrontage grouped by BldgType')
plt.show()

for dataset in (train, test, df):
    dataset['LotFrontage'] = dataset['LotFrontage'].fillna(df.groupby('BldgType')['LotFrontage'].transform('mean'))


# **2.1.10 Feature *Electrical***
# 
# There is only 1 missing value in *Electrical*. The feature shows a considerable correlation with *CentralAir* which indicates whether the house has a central air conditioning.

# In[ ]:


print('Missing values in Electrical: {}'.format(df['Electrical'].isna().sum()))

sns.countplot(df['Electrical'])
plt.xticks(rotation=45, ha='right')
plt.show()

correlation('Electrical')


# The house with a missing value in *Electrical* has a central air conditioning.

# In[ ]:


df.loc[:, ('Id', 'Electrical', 'CentralAir')][df['Electrical'].isna()]


# Over 94% of  houses with central air conditioning have a 'Standard Circuit Breakers & Romex' electrical sytstem. Therefore, the house with the missing value is expected to have the same electrical system. The missing value has been replaced by 'Sbrkr'.

# In[ ]:


sns.catplot(x='CentralAir', data=df, hue='Electrical', kind='count')
plt.xticks(rotation=45, ha='right')
plt.show()

distribution = df.groupby('CentralAir')['Electrical'].value_counts()[4:]
legend = ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix']    
percentage = 100 * distribution.values / distribution.values.sum()

plt.pie(distribution.values / distribution.values.sum(), wedgeprops=dict(edgecolor='black', linewidth=0.25))
plt.legend(['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(legend, percentage)], loc='best', fontsize=10, frameon=True)
plt.title('Distribution of Electrical for CentralAir=Y', fontsize=12)
plt.show()

for dataset in (train, test, df):
    dataset['Electrical'] = dataset['Electrical'].fillna('Sbrkr')


# **2.1.11 Features *BsmtFullBath* & *BsmtHalfBath***
# 
# There are 2 observations which have a missing value in both *BsmtFullBath* and *BsmtHalfBath*.

# In[ ]:


print('Missing values in BsmtFullBath: {}'.format(df['BsmtFullBath'].isna().sum()))
print('Missing values in BsmtHalfBath: {}'.format(df['BsmtHalfBath'].isna().sum()))


# The output below shows that these houses have no basement. Thus, they obviously have 0 full and 0 half bathrooms in the basement. The vales have been replaced with 0.

# In[ ]:


df.loc[:, ('Id', 'BsmtQual', 'BsmtFullBath', 'BsmtHalfBath')][df['BsmtFullBath'].isna()|df['BsmtHalfBath'].isna()]


# In[ ]:


for dataset in (train, test, df):
    for feature in ('BsmtFullBath', 'BsmtHalfBath'):
        dataset[feature] = dataset[feature].fillna(0)


# **2.1.12 Feature *KitchenQual***
# 
# There is only 1 NaN in *KitchenQual*. This house correlates highly with other features indicating the quality of the house.

# In[ ]:


print('Missing values in KitchenQual: {}'.format(df['KitchenQual'].isna().sum()))

correlation('KitchenQual')


# The house has average values for the remaining quality features.

# In[ ]:


df.loc[:, ('KitchenQual','ExterQual', 'BsmtQual', 'OverallQual')][df['KitchenQual'].isna()]


# The feature *ExterQual* which *KitchenQual* correlates most with has the same column values so that missing value can be replaced easily. Missing value after cleaning:

# In[ ]:


for dataset in (train, test, df):
    dataset['KitchenQual'] = dataset['KitchenQual'].fillna(dataset['ExterQual'])

df.loc[[1555], ('KitchenQual','ExterQual', 'OverallQual')]


# **2.1.13 Feature *Functional***
# 
# This feature indicates the home functionality. 'Typ' is assumed unless deductions are warranted. Two houses have no information for its functionality. In fact, 93% of the houses have a typical functionality. The two missing values were replaced by 'Typ'.

# In[ ]:


print('Missing values in Functional: {}'.format(df['Functional'].isna().sum()))

plt.figure(figsize=(8, 6))
plt.pie(df['Functional'].value_counts() / df['Functional'].value_counts().sum(),
        wedgeprops=dict(edgecolor='black', linewidth=0.25))
percentage = 100. * df['Functional'].value_counts() / df['Functional'].value_counts().sum()
plt.legend(['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(percentage.index, percentage)],
           loc='best', fontsize=10, frameon=True)
plt.title('Distribution of Functional', fontsize=14)
plt.show()

# Replace NaN with 'Typ'
for dataset in (train, test, df):
    dataset['Functional'] = dataset['Functional'].fillna('Typ')


# **2.1.14 Feature *SaleType***
# 
# *SaleType* contains different types, such as conventional, cash or loan. There is only 1 missing value in *SaleType*. However, this feature is assumed to have no effect on the price of the house. Indeed, there is no correlation between *SaleType* and *SalePrice*. The feature has thus been dropped from the database.

# In[ ]:


print('Missing values in SaleType: {}'.format(df['SaleType'].isna().sum()))

df.loc[:, ('SalePrice', 'SaleType')].apply(lambda x: x.factorize()[0]).corr()

train = train.drop('SaleType', axis=1)
test = test.drop('SaleType', axis=1)
df = df.drop('SaleType', axis=1)


# Drop columns that have been created for data visualization.

# In[ ]:


for column in ['Count', 'BsmtFinSF2Grouped', 'YearBuiltGrouped', 'LotFrontageGrouped']:
    df = df.drop(column, axis=1)

for dataset in (train, test,df):
    for column in ['MoSold', 'SaleCondition']:
        dataset = dataset.drop(column, axis=1)


# Columns with missing values after cleaning:

# In[ ]:


df.columns[df.isnull().any()].tolist()


# ## 2.2 Feature engineering

# **2.2.1 Feature *TotalSF***
# 
# A feature indicating the total square feet of the house has been engineered.

# In[ ]:


for dataset in (train, test, df):
    dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']


# **2.2.2 Feature *Bath***
# 
# There are houses without a *FullBath*. Surprisingly, houses with 0 full bathromms are more expensive than houses with 1 full bathroom. There is an additional column *HalfBath*. The effect of both features on the sale price is not trivial.

# In[ ]:


catplot('FullBath', train)
catplot('HalfBath', train)


# Summarizing *FullBath* and *HalfBath* into a feature *Bath* could yield better results.

# In[ ]:


for dataset in (train, test, df):
    dataset['Bath'] = dataset['FullBath'] + dataset['HalfBath']

catplot('Bath', train)


# **2.2.3 Feature *YearRemodAdd***
# 
# *YearBuilt* is highly correlated with *YearRemodAdd*. There is a clear positive effect on *SalePrice*. The feature *YearRemodAdd* may be biased due to the fact that it is same as *YearBuilt* if there was no remodeling or addition. Therefore, only *YearBuilt* will be used.

# In[ ]:


df.loc[:, ('YearBuilt', 'YearRemodAdd')].corr()


# In[ ]:


fig, ax  = plt.subplots(nrows=1, ncols=2, figsize=(14,4))
sns.scatterplot(x='YearBuilt', y='SalePrice', data=train, ax=ax[0])
sns.scatterplot(x='YearRemodAdd', y='SalePrice', data=train, ax=ax[1])

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(fontsize=8), plt.yticks(fontsize=8)
fig.show()


# In[ ]:


train = train.drop(['YearRemodAdd'], axis=1)
test = test.drop(['YearRemodAdd'], axis=1)
df = df.drop(['YearRemodAdd'], axis=1)


# ## 2.3 One-hot encoding for categorical features

# In[ ]:


X = df.drop(['SalePrice'], axis=1)

print('Columns before one-hot encoding:', X.shape[1])
        
for column in X:
    if X[column].dtypes == 'object':
        one_hot_encoding = pd.get_dummies(X[column])
        one_hot_encoding.columns = column + '_' + one_hot_encoding.columns.astype('str')
        X = pd.concat([X, one_hot_encoding], ignore_index=False, axis=1, sort=False)
        X = X.drop(column, axis=1)

print('Columns after one-hot encoding:', X.shape[1])


# ## 2.4 Polynomial features for numerical features
# 
# Polynomial features can be powerful for identifying interaction effects in the data. Given two features *a* and *b*, the degree-3 polynomial features are *a*, *b*, *a^2*, *ab*, *b^2*, *a^3*, *a^2b*, *ab^2* and *b^3*. Care must be taken since high degree polynomials make the model prone to overfitting.
# 
# In addition, a large number of features can make models computationally expensive. Therefore, degree-3 polynomials have only been created for the 10 features that correlate most with the target value *SalePrice*.

# In[ ]:


highest_correlation_target = df.drop(['Id'], axis=1).corr().abs().unstack().sort_values(kind='quicksort', ascending=False)#.drop_duplicates(keep='first')
highest_correlation_target = highest_correlation_target.reset_index().rename(columns={0: 'Correlation'})
highest_correlation_target = highest_correlation_target[highest_correlation_target['level_0'].str.contains('SalePrice') |
                                                        highest_correlation_target['level_1'].str.contains('SalePrice')]
highest_correlation_target = highest_correlation_target[highest_correlation_target['Correlation'] < 1]
highest_correlation_target.drop_duplicates(subset='Correlation')[:10]


# In[ ]:


poly = PolynomialFeatures(degree=3)
polynomial = pd.DataFrame(poly.fit_transform(X.loc[:, ('TotalSF', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', '1stFlrSF', 'GarageArea', 'Bath', 'FullBath', 'TotRmsAbvGrd')]),
                          columns=poly.get_feature_names(X.loc[:, ('TotalSF', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', '1stFlrSF', 'GarageArea', 'Bath', 'FullBath','TotRmsAbvGrd')].columns))

X = pd.concat([X, polynomial.loc[:, 'TotalSF^2':'TotRmsAbvGrd^3']], ignore_index=False, axis=1, sort=False)


# ## 2.5 Feature scaling for numerical features

# In[ ]:


for column in X:
    if column != 'Id':
        X[column] = MinMaxScaler().fit_transform(X[[column]])


# ## 2.6 Prepare the datasets
# 
# **2.6.1 Estimators**
# 
# The column *Id* is not used for model training.

# In[ ]:


X_trainval = X.loc[X['Id'] < 1461]
X_trainval = X_trainval.drop(['Id'], axis=1)
X_trainval.name = 'X_trainval'

X_test = X.loc[X['Id'] >= 1461]  
X_test = X_test.drop(['Id'], axis=1)
X_test.name = 'X_test'


# Split trainval set into training and validation set to be able to estimate the generalization performance.

# In[ ]:


X_training, X_validation = train_test_split(X_trainval, test_size=0.2, shuffle=False)
X_training.name = 'X_training'
X_validation.name = 'X_validation'


# In[ ]:


print('Datasets for estimating generalization performance:')
print('X_training: {}'.format(X_training.shape))
print('X_validation: {}\n'.format(X_validation.shape))

print('Datasets for predicting test set:')
print('X_trainval: {}'.format(X_trainval.shape))
print('X_test: {}'.format(X_test.shape))


# Overview of the final trainval set:

# In[ ]:


X_trainval[:5]


# ### 2.6.2 Target value
# 
# Similar to the estimators, the array which captures the *SalePrice* is split into an array y_trainval and an array y_validation to be able to estimate the generalization performance.

# In[ ]:


y_trainval = np.ravel(train[['SalePrice']])
y_training = y_trainval[:len(X_training)]
y_validation = y_trainval[len(X_training):]


# # 3. Model Selection: Comparison of generalization performance
# 
# Findings:
# - XGBoost and Gradient boosting perform best in terms of RMSE.
# - Polynomial features do not yield better results.

# In[ ]:


model_comparison = pd.DataFrame({'Model': [], 'RMSE': []})

xg_boost = xgboost.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.1)
xg_boost.name ='XGBoost'
gradient_boosting = GradientBoostingRegressor(learning_rate=0.1, n_estimators=500, max_depth=3, alpha=0.9)
gradient_boosting.name = 'Gradient boosting'
random_forest = RandomForestRegressor(n_estimators=200, max_features=40, max_depth=40)
random_forest.name = 'Random forest'
lasso = LassoCV(alphas=None, n_alphas=50, cv=10)
lasso.name = 'LASSO'

for model in [xg_boost, gradient_boosting, random_forest, lasso]:
    # Train the model with polynomials
    model.fit(X_training, y_training)
    rmse_poly = round(mean_squared_error(np.log(y_validation), np.log(model.predict(X_validation))) ** (1/2), 5)

    model_results = pd.DataFrame({'Model': [model.name + ' (polynomials)'], 'RMSE': [rmse_poly]})
    model_comparison = model_comparison.append(model_results, ignore_index=True)

    # Train the model without polynomials
    X_training_no_poly = X_training.loc[:, :'SaleCondition_Partial']
    X_validation_no_poly = X_validation.loc[:, :'SaleCondition_Partial']

    model.fit(X_training_no_poly, y_training)

    rmse_no_poly = round(mean_squared_error(np.log(y_validation), np.log(model.predict(X_validation_no_poly))) ** (1/2), 5)

    model_results = pd.DataFrame({'Model': [model.name + ' (no polynomials)'], 'RMSE': [rmse_no_poly]})
    model_comparison = model_comparison.append(model_results, ignore_index=True)
    
model_comparison.sort_values(by='RMSE', ascending=True).reset_index(drop=True)


# # 4. Model Diagnostics
# 
# The predictions of the better performing model for every algorithm are analyzed more in-depth:
# - GXBoost (no polynomials)
# - Gradient boosting (no polynomials)
# - Random forest (no polynomials)
# - LASSO (no polynomials)
# 
# The following diagnostics will be adressed:
# - Feature importance (10 most important features)
# - Learning curve
# - True vs. predicted *SalePrice*
# 
# Learning curves are a powerful technique to get insights into the generalization performance of the model. They show whether the algorithm suffers from overfitting and whether more training data is expected to improve the performance of the algorithm.

# **Define functions:**

# In[ ]:


def learning(model):
    train_sizes, train_scores, test_scores = learning_curve(model, X_trainval, y_trainval, cv=5)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.style.use('seaborn-darkgrid')
    plt.plot(train_sizes, train_mean, color='#1f77b4', label='Training set', linewidth=2)  # Draw lines train
    plt.plot(train_sizes, test_mean, color='#d62728', label='Validation set', linewidth=2)  # Draw lines validation
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='#1f77b4', alpha=0.25)  # Draw band train
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='#d62728', alpha=0.25)  # Draw band validation
    plt.title('Learning curve {}'.format(model.name)), plt.xlabel('Training set size'), plt.ylabel('Score'), plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.show()


# ## 4.1 XGBoost
# 
# **4.1.1 Feature importance**

# In[ ]:


X_training_no_poly = X_training.loc[:, :'SaleCondition_Partial']
X_validation_no_poly = X_validation.loc[:, :'SaleCondition_Partial']

xg_boost.fit(X_training_no_poly, y_training)

feature_importance = pd.DataFrame({'Feature': X_training_no_poly.columns, 'Relative Importance': xg_boost.feature_importances_})
feature_importance = feature_importance.iloc[feature_importance['Relative Importance'].abs().argsort()[::-1]].reset_index(drop=True)
feature_importance[:10]


# **4.1.2 Learning curve**

# In[ ]:


learning(xg_boost)


# ## 4.2 Gradient boosting
# 
# **4.2.1 Feature importance**

# In[ ]:


gradient_boosting.fit(X_training_no_poly, y_training)

feature_importance = pd.DataFrame({'Feature': X_training_no_poly.columns, 'Relative Importance': gradient_boosting.feature_importances_})
feature_importance = feature_importance.iloc[feature_importance['Relative Importance'].abs().argsort()[::-1]].reset_index(drop=True)
feature_importance[:10]


# **4.2.2 Learning curve**

# In[ ]:


learning(gradient_boosting)


# ## 4.3 Random forest
# 
# **4.3.1 Feature importance**

# In[ ]:


random_forest.fit(X_training_no_poly, y_training)

feature_importance = pd.DataFrame({'Feature': X_training_no_poly.columns, 'Relative Importance': random_forest.feature_importances_})
feature_importance = feature_importance.iloc[feature_importance['Relative Importance'].abs().argsort()[::-1]].reset_index(drop=True)
feature_importance[:10]


# **4.3.2 Learning curve**

# In[ ]:


learning(random_forest)


# ## 4.4 LASSO

# **4.4.1 Feature importance (highest coefficients)**

# In[ ]:


lasso.fit(X_training_no_poly, y_training)

alphas = pd.DataFrame(list(lasso.alphas_), columns=['Alpha'])
coefficient_path = lasso.path(X_training_no_poly, y_training, alphas=alphas)
coefficients = pd.DataFrame(coefficient_path[1], index=X_training_no_poly.columns).T.iloc[::-1].reset_index(drop=True)
result = pd.concat([alphas, coefficients], axis=1, sort=False)
best_alpha = result[result['Alpha'] == lasso.alpha_].tail(1)

used_features = best_alpha.columns[(best_alpha != 0).iloc[0]].tolist()
used_features.remove('Alpha')

best_coefficients = best_alpha[best_alpha['Alpha'] == lasso.alpha_].reset_index(drop=True)
best_coefficients = best_coefficients.drop(['Alpha'], axis=1)
best_coefficients = best_coefficients.iloc[0]
best_coefficients = pd.DataFrame({'Feature': best_coefficients.index, 'Coefficient': best_coefficients.values},
                                columns=['Feature', 'Coefficient'])

best_coefficients = best_coefficients.iloc[(-best_coefficients['Coefficient'].abs()).argsort()].reset_index(drop=True)

print('At the best alpha {0}, LASSO set {1} of the total {2} features equal to zero.'.format(
    round(lasso.alpha_, 5),  sum(best_coefficients['Coefficient'] == 0), len(best_coefficients['Coefficient'])))

best_coefficients[:10]


# **4.4.2 Learning curve**

# In[ ]:


learning(lasso)


# ## 4.5 True vs. predicted *SalePice*
# 
# The predictions of the algorithms are plotted below to get more insights into how robust the models are. For better visibility, the first plot displays only the least expensive houses and the second plot the most expensive houses. 

# **4.5.1 Lowest *SalePrice***
# 
# Findings:
# - XGBoost and Gradient boosting are very robust to outliers.
# - LASSO overestimates and Random forest underestimates the *SalePrice*.
# - Random forest performs best for houses with a *SalePrice* between 60K and 100K.

# In[ ]:


prediction_validation_xg_boost = pd.DataFrame({'Id': X['Id'][:len(X_validation_no_poly)], 'Actual': y_validation, 'SalePrice': xg_boost.predict(X_validation_no_poly)})
prediction_validation_gradient_boosting = pd.DataFrame({'Id': X['Id'][:len(X_validation_no_poly)], 'Actual': y_validation, 'SalePrice': gradient_boosting.predict(X_validation_no_poly)})
prediction_validation_random_forest = pd.DataFrame({'Id': X['Id'][:len(X_validation_no_poly)], 'Actual': y_validation, 'SalePrice': random_forest.predict(X_validation_no_poly)})
prediction_validation_lasso = pd.DataFrame({'Id': X['Id'][:len(X_validation_no_poly)], 'Actual': y_validation, 'SalePrice': lasso.predict(X_validation_no_poly)})


# In[ ]:


plt.figure(figsize=(16,10))
plt.plot(prediction_validation_gradient_boosting['Id'][:50], prediction_validation_gradient_boosting['Actual'].sort_values()[:50])
plt.plot(prediction_validation_gradient_boosting['Id'][:50], prediction_validation_gradient_boosting['SalePrice'].sort_values()[:50])
plt.plot(prediction_validation_xg_boost['Id'][:50], prediction_validation_xg_boost['SalePrice'].sort_values()[:50])
plt.plot(prediction_validation_lasso['Id'][:50], prediction_validation_lasso['SalePrice'].sort_values()[:50])
plt.plot(prediction_validation_random_forest['Id'][:50], prediction_validation_random_forest['SalePrice'].sort_values()[:50])
plt.legend(['True SalePrice', 'XG boost', 'Gradient boosting', 'Random forest', 'LASSO']), plt.ylabel('SalePrice'), plt.xlabel('Observation')
plt.title('True SalePrice vs. predictions (cheapest houses)')
plt.show()


# **4.5.2 Houses with highest *SalePrice***
# 
# Finding:
# - Gradient Boosting and XGBoost seem to be very robust to outliers.

# In[ ]:


plt.figure(figsize=(16,10))
plt.plot(prediction_validation_gradient_boosting['Id'][240:], prediction_validation_gradient_boosting['Actual'].sort_values()[240:])
plt.plot(prediction_validation_gradient_boosting['Id'][240:], prediction_validation_gradient_boosting['SalePrice'].sort_values()[240:])
plt.plot(prediction_validation_xg_boost['Id'][240:], prediction_validation_xg_boost['SalePrice'].sort_values()[240:])
plt.plot(prediction_validation_random_forest['Id'][240:], prediction_validation_random_forest['SalePrice'].sort_values()[240:])
plt.plot(prediction_validation_lasso['Id'][240:], prediction_validation_lasso['SalePrice'].sort_values()[240:])
plt.legend(['True SalePrice', 'Gradient boosting', 'XG boost', 'Random forest', 'LASSO']), plt.ylabel('SalePrice'), plt.xlabel('Observation')
plt.title('True SalePrice vs. predictions (most expensive houses)')
plt.show()


# With these insights, the average predictions of the algorithms were calculated. Thereby, the RMSE in the validation set was further reduced.

# In[ ]:


prediction_validation_average = ((gradient_boosting.predict(X_validation_no_poly) +
                                  xg_boost.predict(X_validation_no_poly) +
                                  random_forest.predict(X_validation_no_poly) +
                                  lasso.predict(X_validation_no_poly)) / 4)

rmse_average = round(mean_squared_error(np.log(y_validation), (np.log(prediction_validation_average))) ** (1/2), 5)
print('XGBoost, Gradient Boosting, Random Forest & LASSO:')
print('RMSE =', rmse_average)


# In light of the finding that XGBoost and Gradient boosting predict outliers most accurately, houses with a predicted prices greater than 400K were predicted again only using XGBoost and Gradient boosting. The RMSE further decreased due to this.
# 
# - Indeed, XGBoost and gradient boosting make higher predictions for the outliers and the RMSE decreases.

# In[ ]:


xg_boost.fit(X_training_no_poly, y_training)
gradient_boosting.fit(X_training_no_poly, y_training)
random_forest.fit(X_training_no_poly, y_training)
lasso.fit(X_training_no_poly, y_training)

# Index has to be reset to match indices of X_validation_no_poly and prediction_validation_average (since observations have been dropped earlier)
X_validation_no_poly.set_index([list(range(len(X_training)+1, len(X_training)+1+len(X_validation_no_poly)))], inplace=True)

for index, value in enumerate(prediction_validation_average):
    if value > 400000:
        print('Index:', index)
        print(value)
        prediction_validation_average[index] = (gradient_boosting.predict(X_validation_no_poly.loc[[index+1167], :]) +
                                                xg_boost.predict(X_validation_no_poly.loc[[index+1167], :])) / 2
        print(prediction_validation_average[index])


# In[ ]:


rmse_average = round(mean_squared_error(np.log(y_validation), (np.log(prediction_validation_average))) ** (1/2), 5)
print('Outliers only XGBoost, Gradient Boosting, Random Forest & LASSO:')
print('RMSE =', rmse_average)


# **Make test set predictions:**

# In[ ]:


X_trainval_no_poly = X_trainval.loc[:, :'SaleCondition_Partial']
X_test = X_test.loc[:, :'SaleCondition_Partial']

xg_boost.fit(X_trainval_no_poly, y_trainval)
gradient_boosting.fit(X_trainval_no_poly, y_trainval)
random_forest.fit(X_trainval_no_poly, y_trainval)
lasso.fit(X_trainval_no_poly, y_trainval)

prediction_test = pd.DataFrame({'Id': test['Id'], 'SalePrice': ((xg_boost.predict(X_test) +
                                                                 gradient_boosting.predict(X_test) +
                                                                 random_forest.predict(X_test) +
                                                                 lasso.predict(X_test)) / 4)})

# Make XGBoost and gradient boosting predict houses with predicted SalePrice >400K
for index, column in prediction_test.iterrows():
    if column['SalePrice'] > 400000:
        print('Index:', index)
        print(column['SalePrice'])
        prediction_test.loc[index, 'SalePrice'] = ((xg_boost.predict(X_test.loc[[1460+index], :]) + 
                                                   gradient_boosting.predict(X_test.loc[[1460+index], :]))/2)
        print(prediction_test.loc[index, 'SalePrice'])


# In[ ]:


prediction_test.to_csv('my_submission.csv', index=False)

