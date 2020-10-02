#!/usr/bin/env python
# coding: utf-8

# # House Prices: Advanced Regression Techniques

# ## 1- Load libraries

# In[ ]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats


# ## 2- Import data

# In[ ]:


data_df = pd.read_csv('../input/train.csv')
data_df.head()


# In[ ]:


test_data_df = pd.read_csv('../input/test.csv')


# Check out the dimension of the input data:

# In[ ]:


data_df.shape


# Check out all the features given:

# In[ ]:


data_df.columns


# Make a copy of data_df and test_data_df for preprocessing.

# In[ ]:


train_df = data_df.copy()
test_df = test_data_df.copy()


# ## 3- Exploratory Data Analysis

# First, let's have a look at the range of sale prices, mean, median, as well as the heatmap plot to see how features are correlated.
# 
# 

# > ### Range of sale prices

# In[ ]:


(data_df['SalePrice'].min(), data_df['SalePrice'].max())


# > ### Mean and median of sale [](http://)prices

# In[ ]:


data_df['SalePrice'].mean()


# In[ ]:


data_df['SalePrice'].median()


# Distribution of sale prices

# In[ ]:


data_df['SalePrice'].plot.hist()


# The distribution is somewhat skewed. Let's transform the sale prices into log scale and see how the distribution looks.

# In[ ]:


np.log1p(data_df['SalePrice']).plot.hist()


# This looks much better. The predictions should probably be made on the log of the sale prices. 

# ### Heatmap

# In[ ]:


plt.subplots(figsize=(20,15))
sns.heatmap(data=data_df.corr())


# We can see at a glance that SalePrice is correlated more strongly to a number of features such as:
# * OverallQual (Most strongly correlated)
# * YearBuilt
# * YearRemodAdd
# * MasVnrArea
# * TotalBsmtSF
# * 1stFlrSF
# * GrLivArea (2nd most strongly correlated)
# * FullBath
# * TotRmsAbvGrd
# * GarageCars
# * GarageArea
# 
# I note that some features that I personally think are important don't show up in the map:
# * LotArea
# * LotFrontage 

# In[ ]:


data_df['LotFrontage'].isnull().sum()


# `LotFrontage` has 259 values missing so that may be why.
# 
# Let's have a look at the scatter plot `LotArea` vs. `SalePrice`.

# In[ ]:


sns.regplot(data_df['LotArea'], data_df['SalePrice'])


# There indeed doesn't seem to be as clear a correlation. Most houses have `LotArea` smaller than 25000 SF. Something to think about.

# ### OverallQual
# 
# This is definitely a key feature that decides the range of the sale price for each house. Let's have a closer look.

# In[ ]:


data_df['OverallQual'].plot.hist()


# In[ ]:


sns.regplot(data_df['OverallQual'], data_df['SalePrice'])


# This is not very surprising that houses with higher overall quality get sold for higher prices. The distribution also looks very normal. However, note that the price range is quite large still given a particular value of OverallQual, especially for the highest quality OverallQual=10.

# Now let's look at YearBuilt.

# In[ ]:


data_df['YearBuilt'].plot.hist()


# It appears that more houses are built recently.

# In[ ]:


(data_df['YearBuilt'].min(), data_df['YearBuilt'].max())


# In[ ]:


sns.regplot(data_df['YearBuilt'], data_df['SalePrice'])


# There appears to be a week trend of increasing sale prices with how recent the house is. However, besides the normal variations, there appear to be some outliers.

# In[ ]:


sns.regplot(data_df['TotalBsmtSF'], data_df['SalePrice'])


# There is one point definitely out of the ordinary here with `TotalBsmtSF` > 6000 but very low `SalePrice`. Let's take a look.

# In[ ]:


data_df[data_df['TotalBsmtSF'] > 6000]


# This point has identity `1298`. Let's add this to our list of outliers.

# In[ ]:


outliers = set(data_df[data_df['TotalBsmtSF'] > 6000].index.values)

print('Running list of outliers: ', outliers)


# In[ ]:


sns.regplot(data_df['1stFlrSF'], data_df['SalePrice'])


# Again, one point appears very unusual here with `1stFlrSF` >> 4000 but very low `SalePrice`. Let's take a look.

# In[ ]:


data_df[data_df['1stFlrSF'] > 4000]


# This point is again the house number `1298` that is already in our running list of outliers!

# In[ ]:


sns.regplot(data_df['GrLivArea'], data_df['SalePrice'])


# Although it's a little bit harder to call, the points where `GrLivArea` > 4000 are out of the ordinary with extreme `SalePrice`. Let's have a look to see if our man `1298` is among them. I wouldn't be surprised.

# In[ ]:


data_df[data_df['GrLivArea'] > 4000].loc[:, ['GrLivArea', 'SalePrice']]


# As expected, our man `1298` is among them.  This house must be haunted!!! However, there's also 3 new faces `523`, `691`, and `1182`. Before updating  our list of outliers, let's check zscore.

# In[ ]:


zscore = stats.zscore(data_df['GrLivArea'])
thresh = 4
print(np.where(zscore > thresh), zscore[np.where(zscore > thresh)])


# It's quite obvious that house `1298` is very much an outlier here, but it's a harder call to make for others.  I think since the zscore for `1169` is quite lower than the others, I will for now call the 4 houses we discovered above outliers and add them to our running list.

# In[ ]:


outliers.update([outlier for outlier in list(np.where(data_df['GrLivArea'] > 4000)[0]) if outlier not in outliers])
print('Running list of outliers: ', outliers)


# ## 5- Data Preprocessing: missing values
# I got this idea from `firstbloodY` and found it useful.
# https://www.kaggle.com/firstbloody/an-uncomplicated-model-top-2-or-top-1

# In[ ]:


missing_data = (data_df.isnull().sum()/len(data_df)*100).sort_values(ascending=False)
plt.figure(figsize=(25, 12))
plt.xticks(rotation="90")
sns.barplot(missing_data.index, missing_data)


# ### PoolQC

# According to the feature descriptions, this feature should have 5 possible values:
# * Ex   Excellent
# * Gd   Good
# * TA   Average/Typical
# * Fa   Fair
# * NA   No Pool

# Should all the missing values then be None (no pool) or something else?

# In[ ]:


data_df[data_df['PoolArea'] != 0].loc[:, ['PoolArea', 'PoolQC']]


# It appears that all houses with missing values for `PoolQC` in the training set do not have a pool. All the missing values should therefore be filled with `None`. 

# In[ ]:


train_df['PoolQC'].fillna('None', inplace=True)


# Now, let's look at the test set. There are also many missing values for `PoolQC`. Should all the missing values be filled with None also?

# In[ ]:


test_df['PoolQC'].isnull().sum()


# In[ ]:


test_df[test_df['PoolArea'] != 0].loc[:, ['PoolArea', 'PoolQC']]


# Now, there are 3 houses with nonzero `PoolArea` but no values for `PoolQC` that should not be filled in as `None` but perhaps `Fa`. The rest can be filled in as `None`.

# In[ ]:


test_df[(test_df['PoolArea'] != 0) & (test_df['PoolQC'].isnull())].loc[:, ['PoolQC']].fillna('Fa', inplace=True)
test_df['PoolQC'].fillna('None', inplace=True)


# ### MiscFeature
# 
# Next, look at `MiscFeature` where most values are missing. The possible values are:
# * Elev Elevator
# * Gar2 2nd Garage (if not described in garage section)
# * Othr Other
# * Shed Shed (over 100 SF)
# * TenC Tennis Court
# * NA   None

# In[ ]:


data_df['MiscVal'].plot.hist()


# Since most of the `MiscFeature` results in 0 value, the missing values should consequently be filled with `None`. This feature can probably be dropped.

# In[ ]:


train_df.drop(columns=['MiscFeature'], inplace=True)


# Now let's have a look at the test data.

# In[ ]:


print('Number of missing values in MiscFeature in test data = ', test_df['MiscFeature'].isnull().sum())

test_df['MiscVal'].plot.hist()


# Again, most misc features result in very little value and thus the type of features should not play an important role and can probably be safely removed.

# In[ ]:


test_df.drop(columns=['MiscFeature'], inplace=True)


# ### Alley
# 
# Now let's check out `Alley`. The possible values for `Alley` are:
# * Grvl: Gravel
# * Pave: Paved
# * NA: No Alley
# 
# Usually most of the houses don't have alley and thus the missing values perhaps can be safely  filled with `None`. I should consider removing this feature all together.

# In[ ]:


train_df['Alley'].fillna('None', inplace=True)
test_df['Alley'].fillna('None', inplace=True)


# ### Fence
# 
# The possible values for `Fence` are:
# * GdPrv: Good privacy
# * MnPrv: Minium privacy
# * GdWo: Good wood
# * MnWw: Minium wood/wire
# * NA: No fence
# 
# Most houses don't have fences and hence the missing values are filled with `None`. This feature can be also considered a miscellaneous feature, and it's value, if any, has already been included in `MiscVal`, so can be probably be ignored. 

# In[ ]:


train_df.drop(columns=['Fence'], inplace=True)
test_df.drop(columns=['Fence'], inplace=True)


# ### FireplaceQu
# 
# The possible values for `FireplaceQu` are:
# * Ex	Excellent - Exceptional Masonry Fireplace
# * Gd	Good - Masonry Fireplace in main level
# * TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
# * Fa	Fair - Prefabricated Fireplace in basement
# * Po	Poor - Ben Franklin Stove
# * NA	No Fireplace

# In[ ]:


print('Number of missing values in FireplaceQu = ', data_df['FireplaceQu'].isnull().sum())
print('Number of missing values in Fireplaces = ', data_df['Fireplaces'].isnull().sum())


# In[ ]:


data_df['Fireplaces'].plot.hist()


# It appears that about half of the houses don't have a fireplace. Perhaps this half is the one with no fireplace quality. In addition, `FireplaceQu` can be considered a `MiscFeature` and perhaps can be removed.

# In[ ]:


len(data_df[(data_df['Fireplaces']!=0) & (data_df['FireplaceQu'].isnull())])


# Now go on to the test set. Doublecheck that no house has missing values in `FireplaceQu` with at least a fireplace.

# In[ ]:


len(test_df[(test_df['Fireplaces']!=0) & (test_df['FireplaceQu'].isnull())])


# So as predicted, all houses with missing values in `FireplaceQu` don't have a fireplace, so the missing values should be filled with `None` for both the train set and test set.

# In[ ]:


train_df['FireplaceQu'].fillna('None', inplace=True)


# In[ ]:


test_df['FireplaceQu'].fillna('None', inplace=True)


# ### LotFrontage

# In[ ]:


temp = data_df[data_df['LotArea'] < 55000]
sns.regplot(temp['LotArea'], temp['SalePrice'])


# Let's fill in the missing values for `LotFrontage` with the median. I think we can fill in the missing values with the mean, but median is slightly better in my opinion in case of a skewed distribution or outliers present. 

# In[ ]:


data_df['LotFrontage'].mean()


# In[ ]:


data_df['LotFrontage'].median()


# Let's have a look at the distribution of LotFrontage.

# In[ ]:


data_df['LotFrontage'].plot.hist()


# It turns out to not matter in this case weather we choose the mean or the median anyway.

# In[ ]:


train_df['LotFrontage'].fillna(data_df['LotFrontage'].median(), inplace=True)


# Do the same for the test set, but first, check how many missing values there are and if the mean, median, and distribution look reasonable.

# In[ ]:


print('Number of missing values for LotFrontage in test set = ', test_df['LotFrontage'].isnull().sum())
print('Mean = ', test_df['LotFrontage'].mean())
print('Median = ', test_df['LotFrontage'].median())


# In[ ]:


test_df['LotFrontage'].plot.hist()


# As the mean and median and distribution look similar enough to that of the training data, I think the missing values can be safely filled with the median.

# In[ ]:


test_df['LotFrontage'].fillna(test_df['LotFrontage'].median(), inplace=True)


# Now let's have a look at the relationship between `LotFrontage` and `SalePrice`.

# In[ ]:


sns.regplot(data_df['LotFrontage'], data_df['SalePrice'])


# Two points appear to be outliers from the plot above where `LotFrontage` > 300. We can double check these points are outliers using Seaborn's boxplot or some statistical tools.

# In[ ]:


sns.boxplot(data_df['LotFrontage'])


# In[ ]:


zscore = np.abs(stats.zscore(data_df['LotFrontage']))

thresh = 4
print(np.where(zscore > thresh), zscore[np.where(zscore > thresh)])


# We can see that out of these 7 points, points 934 and 1298 have very much larger zscore compared to others. Let's have a look at these 2 points.

# In[ ]:


data_df[data_df['LotFrontage'] > 300]


# Nothing else strikes me as out of the ordinary. They have very much lower `SalePrice` given very high `LotFrontage` and normal everything else. Let's save the identity of these data points in a list of outliers for future consideration.

# In[ ]:


outliers.update([outlier for outlier in list(np.where(data_df['LotFrontage'] > 300)[0]) if outlier not in outliers])

print('Running list of outliers: ', outliers)


# Let's have a look at the test set.

# In[ ]:


sns.boxplot(test_df['LotFrontage'])


# Although there are some points close to 200 that look out of the ordinary, these points are quite within range of the training data, so they should be OK.

# ### GarageCond, GarageType, GarageYrBlt, GarageFinish, GarageQual
# 
# From my experience, this missing values are expectedly filled with `None`, that is, the houses don't have a garage. Therefore, let's check the `GarageArea`.

# In[ ]:


data_df['GarageArea'].isnull().sum(), data_df['GarageArea'].mean(), data_df['GarageArea'].median()


# In[ ]:


data_df['GarageArea'].plot.hist()


# OK there is no missing value in `GarageArea`, and it appears that most houses have a garage. The distribution looks pretty health. In addidtion, it's indeed true that all the missing values come from houses without a garage, and should therefore be filled with `None`.

# In[ ]:


# Check if there is any house with missing values in the garage category with nonzero garage area
len(data_df[(data_df['GarageArea']!=0) & ((data_df['GarageCond'].isnull()) | 
                                          (data_df['GarageType'].isnull()) | 
                                          (data_df['GarageYrBlt'].isnull()) | 
                                          (data_df['GarageFinish'].isnull()) | 
                                          (data_df['GarageQual'].isnull()))])


# In[ ]:


train_df['GarageCond'].fillna('None', inplace=True)
train_df['GarageType'].fillna('None', inplace=True)
train_df['GarageYrBlt'].fillna('None', inplace=True)
train_df['GarageFinish'].fillna('None', inplace=True)
train_df['GarageQual'].fillna('None', inplace=True)


# Now check the test set:

# In[ ]:


print('Number of missing values in garage area = ', test_df['GarageArea'].isnull().sum())


# In[ ]:


test_df[(test_df['GarageArea']!=0) & ((test_df['GarageType'].isnull()) | 
                                      (test_df['GarageYrBlt'].isnull()) | 
                                      (test_df['GarageFinish'].isnull()) | 
                                      (test_df['GarageQual'].isnull())
                                     )].loc[:, ['GarageArea', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual']]


# OK the one missing value in `GarageArea` is nonzero because there is apparently a  garage type value. This value should then be filled with the median. 

# In[ ]:


test_df['GarageArea'].fillna(test_df['GarageArea'].median(), inplace=True)


# In[ ]:


print('Year most garages were built: ', test_df['GarageYrBlt'].median())
print('Year house 666 was built and year it was remodelled: ', test_df.at[666, 'YearBuilt'], test_df.at[666, 'YearRemodAdd'])
print('Year house 666 was built and year it was remodelled: ', test_df.at[1116, 'YearBuilt'], test_df.at[1116, 'YearRemodAdd'])


# The two houses appeared to have been built in 1910 and 1923 respectively, so I guess having the garage built in 1979 is not that unreasonable. However, it's probably also reasonable to say that the garage was built when the house was remodelled?

# In[ ]:


test_df['GarageFinish'].value_counts()


# In[ ]:


test_df['GarageQual'].value_counts()


# As for the `GarageFinish`, fill in with `Unf` and for `GarageQual`, fill in with `TA`, although these 2 values seem to contradict.

# In[ ]:


test_df.at[666, 'GarageYrBlt'] = 1983
test_df.at[1116, 'GarageYrBlt'] = 1999

test_df.at[666, 'GarageFinish'] = 'Unf'
test_df.at[1116, 'GarageFinish'] = 'Unf'

test_df.at[666, 'GarageQual'] = 'TA'
test_df.at[1116, 'GarageQual'] = 'TA'


# Now fill in the rest of the missing values with `None` because they don't have a garage.

# In[ ]:


test_df['GarageCond'].fillna('None', inplace=True)
test_df['GarageType'].fillna('None', inplace=True)
test_df['GarageYrBlt'].fillna('None', inplace=True)
test_df['GarageFinish'].fillna('None', inplace=True)
test_df['GarageQual'].fillna('None', inplace=True)


# In[ ]:


train_df.isnull().sum().sum()


# ### BsmtExposure, BsmtFinType2, BsmtFinType1, BsmtCond, BsmtQual
# 
# Let's look at these features. Following are the number of missing values:

# In[ ]:


(data_df['BsmtExposure'].isnull().sum(),
data_df['BsmtFinType2'].isnull().sum(),
data_df['BsmtFinType1'].isnull().sum(),
data_df['BsmtCond'].isnull().sum(),
data_df['BsmtQual'].isnull().sum())


# It would not be surprising if (at least most of) these houses don't have a basement at all. We can check the `TotalBsmtSF` to see if the corresponding values are nonzero.

# In[ ]:


data_df['TotalBsmtSF'].isnull().sum()


# There is no missing value in `TotalBsmtSF` so that's good. Now print the houses with non-zero `TotalBsmtSF` but missing values for one of the above features.

# In[ ]:


data_df[(data_df['TotalBsmtSF'] != 0) & ((data_df['BsmtExposure'].isnull()) | 
                                         (data_df['BsmtFinType2'].isnull()) | 
                                         (data_df['BsmtFinType1'].isnull()) |
                                         (data_df['BsmtCond'].isnull()) |
                                         (data_df['BsmtQual'].isnull())
                                        )].loc[:, ['TotalBsmtSF', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']]


# OK so apparently these two houses do have a basement. Now fill in the missing values manually for these 2.

# In[ ]:


data_df['BsmtExposure'].value_counts()


# In[ ]:


data_df['BsmtFinType2'].value_counts()


# In[ ]:


train_df.at[332, 'BsmtFinType2'] = 'Unf'
train_df.at[948, 'BsmtExposure'] = 'No'


# Other missing values are filled in with `None` for no basement.

# In[ ]:


train_df['BsmtExposure'].fillna('None', inplace=True)
train_df['BsmtFinType2'].fillna('None', inplace=True)
train_df['BsmtFinType1'].fillna('None', inplace=True)
train_df['BsmtCond'].fillna('None', inplace=True)
train_df['BsmtQual'].fillna('None', inplace=True)


# Now, check test data to see if there are missing values for these features, and if they can all be filled with `None` for no basement or must be filled in manually.

# In[ ]:


(test_df['BsmtExposure'].isnull().sum(),
test_df['BsmtFinType2'].isnull().sum(),
test_df['BsmtFinType1'].isnull().sum(),
test_df['BsmtCond'].isnull().sum(),
test_df['BsmtQual'].isnull().sum(),
test_df['TotalBsmtSF'].isnull().sum())


# In[ ]:


test_df[(test_df['TotalBsmtSF'] != 0) & ((test_df['BsmtExposure'].isnull()) | 
                                         (test_df['BsmtFinType2'].isnull()) | 
                                         (test_df['BsmtFinType1'].isnull()) |
                                         (test_df['BsmtCond'].isnull()) |
                                         (test_df['BsmtQual'].isnull())
                                        )].loc[:, ['TotalBsmtSF', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']]


# Manually fill in data for these 8 houses using training data. Except for number 660, where all values appear to be missing.

# In[ ]:


basement = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
            'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
test_df.loc[660, basement]


# For number 660, I assume that there is no basement.

# In[ ]:


test_df.at[660, 'BsmtQual'] = None
test_df.at[660, 'BsmtCond'] = None
test_df.at[660, 'BsmtExposure'] = None
test_df.at[660, 'BsmtFinType1'] = None
test_df.at[660, 'BsmtFinSF1'] = 0
test_df.at[660, 'BsmtFinType2'] = None
test_df.at[660, 'BsmtFinSF2'] = 0
test_df.at[660, 'BsmtUnfSF'] = 0
test_df.at[660, 'TotalBsmtSF'] = 0
test_df.at[660, 'BsmtFullBath'] = None
test_df.at[660, 'BsmtHalfBath'] = None


# In[ ]:


train_df['BsmtCond'].value_counts()


# In[ ]:


train_df['BsmtQual'].value_counts()


# In[ ]:


test_df.at[27, 'BsmtExposure'] = 'No'
test_df.at[580, 'BsmtCond'] = 'TA'
test_df.at[725, 'BsmtCond'] = 'TA'
test_df.at[757, 'BsmtQual'] = 'TA'
test_df.at[758, 'BsmtQual'] = 'TA'
test_df.at[888, 'BsmtExposure'] = 'No'
test_df.at[1064, 'BsmtCond'] = 'TA'


# The rest is now filled with `None`.

# In[ ]:


test_df['BsmtExposure'].fillna('None', inplace=True)
test_df['BsmtFinType2'].fillna('None', inplace=True)
test_df['BsmtFinType1'].fillna('None', inplace=True)
test_df['BsmtCond'].fillna('None', inplace=True)
test_df['BsmtQual'].fillna('None', inplace=True)


# ### MasVnrType, MasVnrArea
# 
# How many missing values are we talking about, and are they for the same house?

# In[ ]:


(data_df['MasVnrType'].isnull().sum(), data_df['MasVnrArea'].isnull().sum())


# In[ ]:


data_df[(data_df['MasVnrType'].isnull()) | (data_df['MasVnrArea'].isnull())].loc[:, ['MasVnrType', 'MasVnrArea']]


# `MasVnrArea` has 8 missing values, which are for the same houses that have missing values for `MasVnrType`. `MasVnrArea`  describes the masonry veneer area in square feet and so it may be OK to fill in the missing values with the mean or the median. However, looking at the actual mean and  median, it shows that the data are very skewed and need a second look. This is an example showing that it's safer to go with the median, and not the mean. I go with the median, which is a reasonable choice looking at the distribution plot.

# In[ ]:


data_df['MasVnrArea'].mean()


# In[ ]:


data_df['MasVnrArea'].median()


# In[ ]:


data_df['MasVnrArea'].plot.hist()


# Percentage of houses with 0 masonry veneer area:

# In[ ]:


len(data_df[data_df['MasVnrArea'] == 0])/len(data_df['MasVnrArea'])


# Because these 8 houses have 0 masonry veneer area, the corresponding values for `MasVnrType` should be `None`.

# In[ ]:


train_df['MasVnrArea'].fillna(0, inplace=True)


# In[ ]:


train_df['MasVnrType'].fillna('None', inplace=True)


# Now let's have a look at the test set.

# In[ ]:


test_df['MasVnrArea'].plot.hist()


# In[ ]:


print('Number of missing values in MasVnrType ', test_df['MasVnrType'].isnull().sum(), 'and MasVnArea ', test_df['MasVnrArea'].isnull().sum())


# In[ ]:


test_df[(test_df['MasVnrType'].isnull()) |
        (test_df['MasVnrArea'].isnull())].loc[:, ['MasVnrType', 'MasVnrArea']]


# It appears that there is 1 house with a nonzero `MasVnrArea` but nodata on `MasVnrType`, the missing value thus should be filled with some median value from the set. The rest should be filled with `None`.

# In[ ]:


test_df['MasVnrType'].value_counts()


# In[ ]:


test_df.at[1150, 'MasVnrType'] = 'BrkFace'
test_df['MasVnrType'].fillna('None', inplace=True)
test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].median(), inplace=True)


# ### Electrical
# 
# There is one missing value. that perhaps should just be filled with the most common type.

# In[ ]:


print('Number of missing values = ', data_df['Electrical'].isnull().sum())


# In[ ]:


data_df['Electrical'].value_counts()


# In[ ]:


train_df['Electrical'].fillna('Sbrkr', inplace=True)


# Let's have a look at the test set.

# In[ ]:


print('Number of missing values = ', test_df['Electrical'].isnull().sum())


# ### Final check for missing values
# 
# Now check if there are still missing values in the training data.

# In[ ]:


train_df.isnull().sum().sum()


# However, looking at the test set, there are still a number of missing values. Check what those features are:

# In[ ]:


test_df.columns[np.where(test_df.isnull().sum() != 0)]


# Filling in these missing data with the median values.

# In[ ]:


test_df['MSZoning'].fillna('RL', inplace=True)
test_df['Utilities'].fillna('AllPub', inplace=True)
test_df['Exterior1st'].fillna('VinylSd', inplace=True)
test_df['Exterior2nd'].fillna('VinylSd', inplace=True)
test_df['BsmtFullBath'].fillna(0, inplace=True)
test_df['BsmtHalfBath'].fillna(0, inplace=True)
test_df['KitchenQual'].fillna('TA', inplace=True)
test_df['Functional'].fillna('Typ', inplace=True)
test_df['GarageCars'].fillna(2, inplace=True)
test_df['SaleType'].fillna('WD', inplace=True)


# In[ ]:


test_df.isnull().sum().sum()


# ### Drop Id, SalePrice

# In[ ]:


train_df.drop(columns=['Id', 'SalePrice'], inplace=True)
test_df.drop(columns=['Id'], inplace=True)


# In[ ]:


train_df.shape, test_df.shape


# ### Process outliers

# In[ ]:


print('Current list of outliers ', list(outliers))


# In[ ]:


train_df.drop(index=list(outliers), inplace=True)


# In[ ]:


data_df.drop(index=list(outliers), inplace=True)


# ### Process categorical features
# 
# There are ordinal as well as nominal features, which should be processed with label encoding and one hot encoding separately I would think. However, having tried that it doesn't seem to make it better.

# In[ ]:


dummies = pd.get_dummies(pd.concat((train_df, test_df), axis=0))


# In[ ]:


dummies.shape


# In[ ]:


X = dummies.iloc[:train_df.shape[0]]
X_test = dummies.iloc[train_df.shape[0]:]


# In[ ]:


X.shape, X_test.shape


# ## 6 - Train models with cross validation

# In[ ]:


y = np.log(data_df['SalePrice'] + 1)


# Normalizing features aren't necessary in this problem.

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

GBR3 = GradientBoostingRegressor(learning_rate= 0.1, n_estimators=150, max_depth=4)
cv_score = cross_val_score(GBR3, X, y, cv=5, scoring='neg_mean_squared_error')
print('Cross validation scores for GBR model:', np.sqrt(-cv_score).mean())


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
RF3 = RandomForestRegressor(n_estimators=100, max_features=20, random_state=0)
cv_score2 = cross_val_score(RF3, X, y, cv=5, scoring='neg_mean_squared_error')
print('Cross validation scores for GBR model:', np.sqrt(-cv_score2).mean())


# Random forest doesn't appear to do as well. I'm going to submit the predictions using GBR.

# In[ ]:


GBR3.fit(X, y)
y_pred = np.exp(GBR3.predict(X_test)) - 1
answer = pd.DataFrame(data=y_pred, columns=['SalePrice'])
answer.insert(loc=0, column='Id', value=test_data_df['Id'])

answer.to_csv('submission.csv', index=False)


# 

# In[ ]:




