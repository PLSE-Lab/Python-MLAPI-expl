#!/usr/bin/env python
# coding: utf-8

# # Predicting House Prices with Regression - ML Squad (EDSA_Team19) <a class="anchor" id="BTT"></a>

# # Table of Contents:
# * [Importing Packages and Loading Data](#section-1)
# * [Research](#section-2)
# * [Initial Data Investigation](#section-3)
# * [EDA and Pre-Processing](#section-4)
# * [Feature Engineering](#section-5)
# * [Modelling](#section-6)
# * [Submission Output](#section-7)

# ## Import packages <a class="anchor" id="section-1"></a>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import metrics
from scipy.special import boxcox1p
from sklearn import metrics
from scipy.stats import norm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ## Load Data

# In[ ]:


train_df = pd.read_csv('../input/train.csv', index_col = [0])
test_df = pd.read_csv('../input/test.csv', index_col = [0])


# # Research <a class="anchor" id = "section-2"></a>

# Cosidering the problem of predicting house prices, some research was done as to the factors that affect house prices. These are listed below.
# 
# What was determined from the research was that factors determining house prices can be divided into namely two categories. Intrinsic factors (those that are specific to the house and its immediate surrounds, eg. lot size) and those that are extrinsic (macro factors that determine house prices as a whole eg. economic climate).
# 
# Since extrinsic factors can be difficult to predict accurately enough it was decided that a predictive model such as this one would be more beneficial if it was more focused towards **interpretability** and less towards predictive power.

# ### Factor affecting house prices

# - **Location**: Proximity to schools, work, and shopping centers
# - **Ready-made home**: Finished homes, as well as modern and upgraded fittings
# - **Favourable inspection report**
# - **Neighborhood**: Prices of surrounding homes sets an overall price range for local homes
# - **Appraisals**
# - **Size**: Bigger is better but the size of the house should be appropriate for the area (for example large houses won't sell as easy in a student area)
# - **Number of rooms and features**
# - **Space & flow**
# - **Technology**
# - **Economic indicators**: Current economic climate
# - **Interest rates**
# - **Investors**: Investors may purchase homes in a seller's market in order to upgarde and sell on
# - **Zoning and restrictions**: Dependent on current requirements
# - **Proximity to amenities**: Parks, road access, etc.
# 
# 
# *POINTers - resources.point.com/8-biggest-factors-affect-real-estate-prices/
# <br>
# Property24 - www.property24.com/articles/5-factors-that-will-influence-your-property-value/25480*

# The data provided is from the *Ames Housing dataset*. There is a link that directs to the documentation for the data set.
# 
# *http://jse.amstat.org/v19n3/decock.pdf*

# ### Interpretability

# As the goal is to focus on interpretability, it was decided to look at less flexible models. Models that have interpretable methods.
# 
# These models are generally based off linear regression which makes a few assumptions that need to be taken into account going in to pre-processisng.

# #### Linear Regression makes the following assumptions:
# - **Linear relationship**
# - **No multicollinearity**
# - **Homoscedasticity (consistent variance in error)**: This can be usually be ensured by having the target variable follow a normal distribution
# 
# Since linear regression minimizes using least squares, if target is skewed the model puts more emphasis to reduce error for the skewed data.

# # Initial Look at Data <a class="anchor" id = "section-3"></a>

# ### Data

# The data provided is from the *Ames Housing dataset*. There is a link that directs to the documentation for the data set.
# 
# *http://jse.amstat.org/v19n3/decock.pdf*

# ### Outliers

# In the documentation it is outlined that there are 5 obvious outliers observations that were left in but are recommended to be removed. The way these outliers can be identified is by simply plotting the 'SalePrice' against 'GrLivArea'. It also outlines the best way to remove these outliers from the dataset; by removing any observation with more than 4000 square feet.

# In[ ]:


'''
Here, 4 outliers are clearly seen. Two of these are partial sales hence
the large area and low sale price and the other two have large areas but
with sale prices well above the rest of the observations.

Note: Can see some evidence of heteroscedasticity here.
'''
sns.regplot(train_df['GrLivArea'], train_df['SalePrice'])
plt.title('Outliers in GrLivArea', fontsize = 14)
plt.xlabel('General Living Area', fontsize = 14)
plt.ylabel('Sale Price', fontsize = 14)
plt.show()


# In[ ]:


'''
Following the dataset's author's recommendations to remove any observation
with a living area of more than 4000 square feet, we only come across the
4 outliers mentioned above and not 5 as is mentioned in the documentation.
'''
train_df[train_df['GrLivArea'] > 4000].index


# In[ ]:


# These 4 outliers are removed.

train = train_df[train_df['GrLivArea'] <= 4000]


# We can see at the bottom right two with extremely large GrLivArea that are of a low price. These values are huge oultliers. Therefore, we can safely delete them

# In[ ]:


'''
From the plot below the regression plot for the axes has lower variability.
'''
sns.regplot(train['GrLivArea'], train['SalePrice'])
plt.title('Outliers remmoved from GrLivArea', fontsize = 14)
plt.xlabel('General Living Area', fontsize = 14)
plt.ylabel('Sale Price', fontsize = 14)
plt.show()


# **Note:** The following outliers were identified in during the data analysis done below. It has been placed here in the notebook as it makes removing these outliers from the data much easier.

# In[ ]:


train = train[train['LotArea'] < 100000]
train = train[train['TotalBsmtSF'] < 3000]
train = train[train['1stFlrSF'] < 2500]
train = train[train['BsmtFinSF1'] < 2000]


# ### Concatenate Data

# Concatenate Test and Train data so as to deal with missing data on all data.

# In[ ]:


X = pd.concat((train.iloc[:,:-1], test_df), sort = False)
y = train.iloc[:, -1]


# In[ ]:


numerical_df = X.select_dtypes(include = np.number)
categorical_df = X.select_dtypes(exclude = np.number)


# ### Investigating Assumptions

# - #### Homoscedasticity

# First need to check for homoscedasticity with the target variable, ie. that the target variable follows a normal distribution.

# **Note:** This was not done when first creating the model but after going through the theory and implementing the assumptions, the model accuracy improved.

# In[ ]:


'''
The plot below shows a skewed distribution for SalePrice or heteroscedasticity.

A normal distribution should follow the red line in the probability plot
on the right.
'''
plt.figure(figsize = (12,6))

plt.subplot(1,2,1)
sns.distplot(y, fit = norm)
plt.title('Distribution plot for SalePrice', fontsize = 14)
plt.xlabel('SalePrice', fontsize = 14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.subplot(1,2,2)
stats.probplot(y, plot=plt)
plt.title('Probability plot for SalePrice', fontsize = 14)
plt.xlabel('Theoretical Quantiles', fontsize = 14)

plt.show()


# In[ ]:


'''
Since the above distribution look like a log-normal distribution,
a log transformation is done to SalePrice.

Below the plot shows a normal-like distribution after performing
a log + 1 transformation.

The probability plot of right follows the red line more closely.

The + 1 is to avoid taking a log of 0, which is undefined as well as ensuring the result is not negative. 
'''
plt.figure(figsize = (12,6))

plt.subplot(1,2,1)
sns.distplot(np.log1p(y), fit = norm)
plt.title('Distribution plot for log of SalePrice', fontsize = 14)
plt.xlabel('SalePrice', fontsize = 14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.subplot(1,2,2)
stats.probplot(np.log1p(y), plot=plt)
plt.title('Probability plot for log of SalePrice', fontsize = 14)
plt.xlabel('Theoretical Quantiles', fontsize = 14)

plt.show()


# In[ ]:


'''
Performing a log transformation on the target variable.
'''
y = np.log1p(y)


# To better illustrate how normalising can help with homoscedasticity, the graphs below were plotted before transformation and after transformatiob.
# 
# Can see that after the transformation the plot of SalePrice against GrLivArea is more linear and the error is more homogeneous.

# In[ ]:


plt.figure(figsize = (14,6))

plt.subplot(1,2,1)
sns.regplot(train['GrLivArea'], train['SalePrice'])
plt.title('Before Log Transformation', fontsize = 14)
plt.xlabel('Above Ground Living Area', fontsize = 14)
plt.ylabel('Sale Price', fontsize = 14)

plt.subplot(1,2,2)
sns.regplot(train['GrLivArea'], y)
plt.title('After Log Transformation', fontsize = 14)
plt.xlabel('Above Ground Living Area', fontsize = 14)
plt.ylabel('Sale Price', fontsize = 14)
plt.show()


# - #### Multicollinearity

# To check for this, a correlation map of all numerical columns with each other is done. The highly correlated features will be idnetified and by using the data descriptions, decide what to do with those features.

# ##### Numerical

# Idenitfying the numerical features that show multi-collinearity by using a heatmap of correlation values between features.

# In[ ]:


plt.figure(figsize=(15, 15))
sns.heatmap(numerical_df.corr(), cmap = 'coolwarm', square = True, vmax=.8)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.show()


# In[ ]:


'''
Below the feature with a correlation of 0.8 or higher are identified.

Filtered for correlation less than 1 to rule out self-correlated instances.
'''
unstacked_corr_df = numerical_df.corr().abs().unstack()
sorted_corr_df = unstacked_corr_df.sort_values()
sorted_corr_df[(sorted_corr_df > 0.8) & (sorted_corr_df < 1)]


# ***Based off column descriptions that were provided with the data:***
# 
#     - GarageCars:    Size of garage in car capacity
#     - GarageArea:    Size of garage in square feet
#     - TotRmsAbvGrd:  Total rooms above grade/ground (not bathrooms)
#     - GrLivArea:     Above grade/ground living area square feet
#     - GarageYrBlt:   Year garage was built
#     - YearBuilt:     Original construction date
#     - TotalBsmtSF:   Total square feet of basement area
#     - 1stFlrSF:      First Floor square feet

# Using the above information, it will be determined what to do with these columns.

# - #### GarageCars and GarageArea

# GarageCars is slightly more correlated and since these two features are so highly correlated to each other, GarageArea will be dropped.
# 
# **Note**: All columns won't be dropped immediately until missing values are all filled in. The redundant columns miight be able to provide information about the missing values.

# In[ ]:


numerical_df[['GarageArea', 'GarageCars']].corrwith(y)


# - #### YearBuilt and GarageYrBlt

# The percentage of observations without missing values for the respective columns, that have equal YearBuilt and GarageYrBlt values.

# In[ ]:


temp_df = numerical_df[['YearBuilt', 'GarageYrBlt']].dropna()
equal_cols = len(temp_df[temp_df['YearBuilt'] == temp_df['GarageYrBlt']].index)
equal_cols/len(temp_df.index)*100


# Seeing as 80% is a fairly significant percentage, one of the columns will have to be dropped. Since YearBuilt has no missing values GarageYrBlt will be dropped.

# In[ ]:


print('Number of missing values for GarageYrBlt:', numerical_df['GarageYrBlt'].isnull().sum())
print('Number of missing values for YearBuilt:', numerical_df['YearBuilt'].isnull().sum())


# - #### 1stFlrSF and TotalBsmtSF

# From the description it can be seen that these two features are describing different things. The correlation though still make sense as basements are usually a similar size to the first floor. Thus both features should be kept.

# **Note:** Separate ordinal columns were created in order to help with the collinearity but these columns did not improve the model thus it was decided to keep the columns as is.

# - #### GrLivArea and TotRmsAbvGrd

# The correlation here is logical as a larger living area naturally leads to more rooms. But depending on the size of the rooms this is not usually the case so both features should be kept since they both have a realtively good correaltion with the target variable, SalePrice.

# In[ ]:


numerical_df[['GrLivArea', 'TotRmsAbvGrd']].corrwith(y)


# - #### Columns to drop

# In[ ]:


'''
Creating a list of columns to drop.
'''
drop_cols = ['GarageArea', 'GarageYrBlt']


# **Note:** There are featuers with correlations of around 0.5 and higher. Some of these features will be combined into new features where possible in order to combat multicollinearity.
# 
# This will be done in **[Feature Engineering](#section-5)** below.

# # Data Analysis and Pre-Processing <a class="anchor" id = "section-4"></a>

# ### Missing Values

# First thing to check is for observations with too many null values. Since there are a large amount of features (79), the threshold will be set at 20.
# 
# There no observations with more than 20 missing values.

# In[ ]:


X[X.apply(lambda x: 79 - x.count(), axis=1) > 20].apply(lambda x: 79 - x.count(), axis=1)


# #### Getting percentage of missing values

# From the values below, PoolQC, MiscFeature, Alley and Fence all have more than 80% missing values.
# 
# The graph shows the percentage of missing values per feature. If the feature is on the plot than it has at least 1 missing value.

# In[ ]:


null_cols = X.columns[X.isnull().any()]
missing_data = (X[null_cols].isnull().sum()/len(X)*100).sort_values(ascending = False)

plt.figure(figsize = (12,6))
plt.bar(missing_data.index, missing_data)
plt.title('Percentage of missing values in each feature', fontsize = 14)
plt.ylabel('Percentage', fontsize = 14)
plt.xticks(rotation = 90)
plt.show()


# Usually the case is to drop features with more than a certain threshold of missing values but according to the data description some of these missing values are actual 'None' values or 0's. Thus each feature will be looked at individually where it makes sense to do so.

# - #### PoolQC, MiscFeature and Fence

# Too many missing values for these features, thus these are dropped.

# In[ ]:


X.drop(['PoolQC', 'MiscFeature', 'Fence'], axis = 1, inplace = True)


# Datatypes of columns with missing values.
# 
# Can see that a sizeable number of features have missing data. The null values will have to be imputed carefully so as to not significantly alter the original data.

# In[ ]:


cols = X.columns[X.isnull().any()]

print('Number of numeric columns with missing data', len(X[cols].select_dtypes(include = np.number).columns))
print('Number of categorical columns with missing data', len(X[cols].select_dtypes(exclude = np.number).columns))


# Number of missing values in each numeric columns

# In[ ]:


X_float_missing = X[cols].select_dtypes(include=np.number)
X_float_missing.isnull().sum()


# - #### LotFrontage

# LotFrontage is the linear feet of street connected to property. One strategy is to look at related features and base the missing values of these features.
# 
# LotFrontage is compared per neighborhood below. As can be seen that the mean of the LotFrontage varies significantly based on neighborhood. Even though a few neighborhhods have a larger spread of values, the majority of the nieghborhoods have a small spread, thus this will be used to impute the missing values.

# In[ ]:


plt.figure(figsize=(15,8))
sns.boxplot(X['Neighborhood'], X['LotFrontage'], width=0.7, linewidth=0.8)
plt.xticks(rotation = 60)
plt.title('Boxplot of LotFrontage per Neighborhood', fontsize = 14)
plt.xlabel('Neighborhood', fontsize = 14)
plt.ylabel('LotFrontage', fontsize = 14)
plt.show()


# In[ ]:


X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.mean()))


# - #### MasVnrArea & MasVnrType

# Based on documentation MasVnrArea will be 0 if MasVnrType is 'None'.
# 
# Notice that MasVnrType has one true missing value.

# In[ ]:


X[['MasVnrType', 'MasVnrArea']].isnull().sum()


# In[ ]:


X[(X['MasVnrType'].isnull()) & (X['MasVnrArea'].notnull())][['MasVnrType', 'MasVnrArea']]


# In[ ]:


X['MasVnrType'].value_counts(dropna = False)


# Since the mode of MasVnrType is 'None', but know that it is truely a missing value, fill the missing value with 'BrkFace'.

# In[ ]:


X.at[2611, 'MasVnrType'] = 'BrkFace'


# Replacing null values of MasVnrType with 'NoVnr' and replace null values of MasVnrArea with 0.

# In[ ]:


X['MasVnrType'] = X['MasVnrType'].fillna('NoVnr')
X['MasVnrArea'] = X['MasVnrArea'].fillna(0)


# - #### Bsmt features

# A null value indicates no basement. But takinga closer look at the missing values, it can be seen that there are some true missing values, namely in BsmtExposure, BsmtBsmtQual and BsmtCond.

# In[ ]:


X[['BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'TotalBsmtSF']].isnull().sum()


# For BsmtExposure the most common value will be used to fill in the missing values.

# In[ ]:


X[(X['BsmtExposure'].isnull()) & (X['BsmtFinType1'].notnull())][['BsmtExposure', 'BsmtFinType1']]


# In[ ]:


X['BsmtExposure'].mode()[0]


# In[ ]:


X.update(X[(X['BsmtExposure'].isnull()) & (X['BsmtFinType1'].notnull())]['BsmtExposure'].fillna(X['BsmtExposure'].mode()[0]))


# Replace the BsmtCond with the same value from BsmtQual.

# In[ ]:


X[(X['BsmtCond'].isnull()) & (X['BsmtQual'].notnull())][['BsmtQual','BsmtCond']]


# In[ ]:


X.update(X[(X['BsmtCond'].isnull()) & (X['BsmtQual'] == 'Gd')]['BsmtCond'].fillna('Gd'))


# In[ ]:


X.update(X[(X['BsmtCond'].isnull()) & (X['BsmtQual'] == 'TA')]['BsmtCond'].fillna('TA'))


# Replace BsmtQual with the corresponding value in BsmtCond.

# In[ ]:


X[(X['BsmtCond'].notnull()) & (X['BsmtQual'].isnull())][['BsmtQual','BsmtCond']]


# In[ ]:


X.update(X[(X['BsmtQual'].isnull()) & (X['BsmtCond'] == 'Fa')]['BsmtCond'].fillna('Fa'))


# In[ ]:


X.update(X[(X['BsmtQual'].isnull()) & (X['BsmtCond'] == 'TA')]['BsmtCond'].fillna('TA'))


# This leaves only the null values corresponding to no basement. Fill the Bsmt features with 'NoBsmt'

# In[ ]:


X.update(X[['BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure']].fillna('NoBsmt'))


# Fill null value in TotalBsmtSF with 0.

# In[ ]:


X.update(X[['TotalBsmtSF', 'BsmtQual']].fillna(0))


# - #### BsmtFullBath and BsmtHalfBath

# In[ ]:


cols = X.columns[X.isnull().any()]
X_float_missing = X[cols].select_dtypes(include=['float64'])
X_float_missing.isnull().sum()


# A null value for these features indicate 0 but looking at the TotalBsmtSF, only one of these corresponds to no bathrooms.
# 
# Since we imputed 'NoBsmt' for index 2121, the baths will be set to 0 and TotalBsmtSF as well as the othe features desribing Bsmt square feet will be set to 0 as well since these value are also missing.

# In[ ]:


X[X['BsmtFullBath'].isnull()][['BsmtFullBath', 'BsmtHalfBath', 'BsmtQual','TotalBsmtSF']]


# Since TotalBsmtSF is 0 BsmtFullBath and BsmtHalfBath will be set to zero.

# In[ ]:


X[['BsmtFullBath', 'BsmtHalfBath']] = X[['BsmtFullBath', 'BsmtHalfBath']].fillna(0)


# ##### Categorical Data

# Number of missing values for each categorical feature.

# In[ ]:


cols = X.columns[X.isnull().any()]
X[cols].isnull().sum()


# - #### BsmtFinSF1, BsmtFinSF2 and BsmtUnfSF

# Since these values are meant to be numerical and the corresponding TotalBsmtSF is 0, thus set these null values as 0.

# In[ ]:


X[X['BsmtFinSF1'].isnull()][['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']]


# In[ ]:


X[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']] = X[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']].fillna(0)


# - #### Alley and Garage Features

# Based on documentation Alley and Garage related columns that have null values are null if these do not exist, thus set to None

# In[ ]:


X.update(X[['Alley','FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']].fillna('None'))


# In[ ]:


cols = X.columns[X.isnull().any()]
X[cols].isnull().sum()


# - #### GarageCars

# In this case we have null values for both GarageCars and GarageArea thus there is no good way of determining the missing value.
# 
# The missing value will be filled in with the mode of GarageCars.

# In[ ]:


X[X['GarageCars'].isnull()][['GarageCars', 'GarageArea']]


# In[ ]:


X['GarageCars'] = X['GarageCars'].fillna(X['GarageCars'].mode()[0])


# - #### Utilties

# As can be seen below that the Utilities feature is almost entirely dominated by 'AllPub' in the train data.
# 
# When looking at the test data, see that it only contains 'AllPub' thus the feature would not provide any useful information for the predictive model.
# 
# Drop Utilities.

# In[ ]:


plt.subplots(figsize =(12, 6))

plt.subplot(1, 2, 1)
sns.countplot('Utilities', data = X).set_title('Train data - Utilities')

plt.subplot(1, 2, 2)
sns.countplot('Utilities', data = test_df).set_title('Test data - Utilities')

plt.show()


# In[ ]:


'''
Adding column to drop to drop list.
'''
drop_cols = drop_cols + ['Utilities']


# - #### Remaining Features

# Since the remaining features only have one null value, these values will be imputed with their respective modes.

# In[ ]:


cols = ['MSZoning', 'Functional', 'SaleType', 'KitchenQual', 'Electrical', 'Exterior1st', 'Exterior2nd']
for col in cols:
    X.update(X[col].fillna(X[col].mode()[0]))


# Drop columns from drop_cols list.

# In[ ]:


X.drop(drop_cols, axis = 1, inplace = True)


# **All missing values are taken care of, as there are no more columns with null values.**

# In[ ]:


cols = X.columns[X.isnull().any()]
X[cols].isnull().sum()


# *Since dropped MiscFeature thus it would make sense to drop MiscVal.*

# In[ ]:


X.drop(['MiscVal'], axis = 1, inplace = True)


# *Since vast Majority of PoolArea is 0 and just a handful have a pool, PoolArea is also dropped.*

# In[ ]:


len(X[X['PoolArea'] == 0].index)/len(X.index)*100


# In[ ]:


X.drop(['PoolArea'], axis = 1 , inplace = True)


# # Feature Engineering <a class="anchor" id = "section-5"></a>

# ### Features vs Target

# Plot all numerical features against the target varaible.
# 
# A look at the behaviour of some features vs the target variable is done in order to ensure that the data satifies our **assumptions** for applying linear regression or models based off linear regression.

# In[ ]:


fig, axs = plt.subplots(ncols=2, nrows=16, figsize=(12, 80))

for i, feature in enumerate(X[:train.shape[0]].select_dtypes(include=np.number).columns, 1):    
    plt.subplot(16, 2, i)
    plt.scatter(x=feature, y= y, data=(X[:train.shape[0]].select_dtypes(include=np.number)))
        
    plt.xlabel('{}'.format(feature), size=15)
    plt.ylabel('SalePrice', size=15, labelpad=12.5)
    
    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
    
plt.show()


# Taking the three biggest correlated features(LotArea, GrLivArea, TotalBsmtSF) and creating a pairplot to further investigate their multicolinearity as well as to investigate the homoscedasticity of the data.

# In[ ]:


sns.pairplot(X[['GrLivArea', 'LotArea', 'TotalBsmtSF']])
plt.show()


# ### Normalising

# From the above plots we can see some evidence for heteroscedasticity.
# 
# A few non-sparse features with relatively high correlation with the target variable (namely 'LotArea', 'GrLivArea' and 'TotalBsmntSF') will be transformed in order to have a more normal distribution. This will aid in obtaining homoscedasticity.

# In[ ]:


cols = ['LotArea', 'GrLivArea', 'TotalBsmtSF']

# Using boxcox + 1 to transform the features.
for col in cols:
    X[col] = boxcox1p(X[col], 0.1)


# Comparing the two pairplots, the transofrmation has helped with the normalising the data and now the features show less signs of heteroscadisticity.

# In[ ]:


sns.pairplot(X[['GrLivArea', 'LotArea', 'TotalBsmtSF']])
plt.show()


# ### Creating new variables

# As can be seen in the above plots, a few features are quite sparse(have domination values).
# 
# These features will be combined into a few new features in order to help with multicollinearity. 

# In[ ]:


X['TotalBath'] = X['FullBath'] + X['HalfBath']*0.5 + X['BsmtFullBath'] + X['BsmtHalfBath']*0.5
X['TotalFlrSF'] = X['1stFlrSF'] + X['2ndFlrSF']
X['BsmtFinSF'] = X['BsmtFinSF1'] + X['BsmtFinSF2']


# In[ ]:


# Old features are dropped

comb_cols_drop = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2']
for col in comb_cols_drop:
    X.drop([col], axis =1, inplace = True)


# ### Enconding

# Encoding ordinal features (label encoding).

# In[ ]:


X['LandContour'] = X['LandContour'].replace(dict(Lvl=4, Bnk=3, HLS=2, Low=1))
X['LandSlope'] = X['LandSlope'].replace(dict(Gtl=3, Mod=2, Sev=1))
X['ExterQual'] = X['ExterQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))
X['ExterCond'] = X['ExterCond'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

X['BsmtQual'] = X['BsmtQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1, NoBsmt=0))
X['BsmtCond'] = X['BsmtCond'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1, NoBsmt=0))
X['BsmtExposure'] = X['BsmtExposure'].replace(dict(Gd=4, Av=3, Mn=2, No=1, NoBsmt=0))
X['BsmtFinType1'] = X['BsmtFinType1'].replace(dict(GLQ=6, ALQ=5, BLQ=4, Rec=3, LwQ=2, Unf=1, NoBsmt=0))
X['BsmtFinType2'] = X['BsmtFinType2'].replace(dict(GLQ=6, ALQ=5, BLQ=4, Rec=3, LwQ=2, Unf=1, NoBsmt=0))

X['HeatingQC'] = X['HeatingQC'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))
X['CentralAir'] = X['CentralAir'].replace(dict(Y=1, N=0))
X['KitchenQual'] = X['KitchenQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))
X['Functional'] = X['Functional'].replace(dict(Typ=8, Min1=7, Min2=6, Mod=5, Maj1=4, Maj2=3, Sev=2, Sal=1))
X['FireplaceQu'] = X['FireplaceQu'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))
X['FireplaceQu'] = X['FireplaceQu'].replace('None', 0)

X['GarageQual'] = X['GarageQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))
X['GarageQual'] = X['GarageQual'].replace('None', 0)
X['GarageCond'] = X['GarageCond'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))
X['GarageCond'] = X['GarageCond'].replace('None', 0)
X['GarageFinish'] = X['GarageFinish'].replace(dict(Fin=3, RFn=2, Unf=1))
X['GarageFinish'] = X['GarageFinish'].replace('None', 0)


X['LotShape'] = X['LotShape'].replace(dict(Reg=4, IR1=3, IR2=2, IR3=1))
X['PavedDrive'] = X['PavedDrive'].replace(dict(Y=3, P=2, N=1))


# Transforming some numerical variables that work work better as categorical data.

# In[ ]:


X['MSSubClass'] = X['MSSubClass'].astype('category')
X['MoSold'] = X['MoSold'].astype('category')
X['YrSold'] = X['YrSold'].astype('category')


# Creating dummy variables while also avoiding the dummy variable trap.

# In[ ]:


X = pd.get_dummies(X, drop_first=True)


# # Modelling <a class="anchor" id = "section-6"></a>

# ### Fitting and splitting data

# The data is separated into the modelling set and the prediction set.

# In[ ]:


X_tr = X[:train.shape[0]] # Modelling set
X_t = X[train.shape[0]:] # Prediction set


# The modelling set will be split between 80% train and 20% test. This was chosen as the number of observations in the data are relatively low. By ensuring that the train set is as large as possible, the model will be better trained.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tr, y, test_size=0.20, shuffle=False)


# ### Models

# #### Linear Regression Models

# Taking into account the goal of interpretability over predictive power, but also wanting to have fairly accurate predictions. Three candidates are avaialable,Ridge Regression, Lasso Regression and ElasticNet.
# 
# These models all would be very suitable as in the spirit of interpretability and simplification no feature selection was done. The models mentioned all perform regularisation which will either penalise features which are not important or drop them completely.
# 
# Now since the approach is focused on interpretability, the predictive power is hampered a bit. In order to compensate for this a model would need a larger training data set to improve its accuracy. With this in mind Ridge Regression was chosen.
# 
# **Ridge Regression** is easy to implement and computation less complex than Lasso and certainly less complex as compared to ElasticNet. This will allow for easy scalabilty to very large datasets.

# - ### Ridge

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()
parameters = {'alpha': [1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=10, iid = True)

ridge_regressor.fit(X_train, y_train)


# In[ ]:


# RMSLE for train set and test set
print('RMSLE tests')

train_rr = ridge_regressor.predict(X_train)
print('Ridge Train:', np.sqrt(metrics.mean_squared_log_error(np.expm1(y_train), np.expm1(train_rr))))

test_rr = ridge_regressor.predict(X_test)
print('Ridge Test:', np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(test_rr))))


# From the above RMSLE (root mean square log error) it seems like the test error is a fair bit larger than the train error but not too large to indicate gross overfitting.

# #### R-squared and Adjusted R-squared

# Checking R-squared and adjusted R-squared values for our model to ensure a good fit.

# ##### Train set

# In[ ]:


yhat = ridge_regressor.predict(X_train)
SS_Residual = sum((y_train-yhat)**2)
SS_Total = sum((y-np.mean(y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
print('Train Scores')
print(r_squared, adjusted_r_squared)


# ##### Test set

# In[ ]:


yhat = ridge_regressor.predict(X_test)
SS_Residual = sum((y_test-yhat)**2)
SS_Total = sum((y-np.mean(y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
print('Test Scores')
print(r_squared, adjusted_r_squared)


# ***

# ### Addendum

# #### Lasso

# This is a Lasso regression model done out of interest sake. It wasn't chosen for the reasons stated above.

# In[ ]:


from sklearn import linear_model

parameters = {'alpha': np.arange(0.0001,0.005, 0.0001)}
ls = linear_model.Lasso()
lasso = GridSearchCV(ls, parameters, scoring='neg_mean_squared_error', cv=10, iid = True)

lasso.fit(X_train, y_train)


# In[ ]:


# RMSLE for train set and test set
print('RMSLE tests')

train_ls = lasso.predict(X_train)
print('Lasso Train:', np.sqrt(metrics.mean_squared_log_error(np.expm1(y_train), np.expm1(train_ls))))

test_ls = lasso.predict(X_test)
print('Lasso Test:', np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(test_ls))))


# #### RandomForest

# This is a RandomForest Regression model that was attempted out of interest sake. It ended up overfitting the data.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

param_grid = {'bootstrap': [True, False], 'max_depth': [5, 10, 15, 20, 30, 40, 60, 80, 100, 120],
    'max_features': [2, 3, 5, 7, 10, 13, 16, 20],
    'min_samples_leaf': [3, 4, 5, 6, 7],
    'min_samples_split': [8, 10, 12, 14, 16],
    'n_estimators': [100, 200, 300, 600, 1000]}

rndm_frst = RandomForestRegressor()  
grid_search = GridSearchCV(estimator = rndm_frst, param_grid = param_grid, 
                          cv = 5, verbose = 2, scoring='neg_mean_squared_error')

rndm_frst.fit(X_train, y_train)


# In[ ]:


# RMSLE for train set and test set
print('RMSLE tests')

train_rf = rndm_frst.predict(X_train)
print('RF Train:', np.sqrt(metrics.mean_squared_log_error(np.expm1(y_train), np.expm1(train_rf))))

test_rf = rndm_frst.predict(X_test)
print('RF Test:', np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(test_rf))))


# ## Submission <a class="anchor" id="section-7"></a>

# In[ ]:


prediction = np.expm1(ridge_regressor.predict(X_t))


# In[ ]:


sub_df = pd.DataFrame({"id":test_df.index, "SalePrice":prediction})
sub_df.to_csv("ridge_reg.csv", index = False)


# ### [Back To Top](#BTT)
