#!/usr/bin/env python
# coding: utf-8

# This notebook is focused on performing **Exploratory Data Analysis** on the [House Prices: Advanced Regression techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
# 
# Before starting I want to thank other kaggle users for their work on this problem. It helped me alot in understanding this problem.
# 
# This and others notebooks onn this project series relies heavily on other great kernels made on this dataset.
# Naming a few:
# 1. [Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
# 2. [A study on Regression applied to the Ames dataset](https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset)
# 3. [Eda and prediction of House Price](https://www.kaggle.com/siddheshpujari/eda-and-prediction-of-house-price)
# 3. [Stacked Regressions to predict House Prices](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)
# 4. [Regularized Linear Models](https://www.kaggle.com/apapiu/regularized-linear-models)
# 5. [How I made top 0.3% on a Kaggle competition](https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition)

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#EDA" data-toc-modified-id="EDA-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>EDA</a></span></li><li><span><a href="#Personal-selection-and-understanding-of-features" data-toc-modified-id="Personal-selection-and-understanding-of-features-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Personal selection and understanding of features</a></span></li><li><span><a href="#Taking-a-look-at-SalePrice" data-toc-modified-id="Taking-a-look-at-SalePrice-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Taking a look at SalePrice</a></span><ul class="toc-item"><li><span><a href="#Relationship-between-selected-features" data-toc-modified-id="Relationship-between-selected-features-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Relationship between selected features</a></span><ul class="toc-item"><li><span><a href="#Relation-with-numerical-variables" data-toc-modified-id="Relation-with-numerical-variables-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Relation with numerical variables</a></span></li><li><span><a href="#Relation-with-categorical-variables" data-toc-modified-id="Relation-with-categorical-variables-3.1.2"><span class="toc-item-num">3.1.2&nbsp;&nbsp;</span>Relation with categorical variables</a></span></li></ul></li></ul></li><li><span><a href="#Analysis-of-other-variables" data-toc-modified-id="Analysis-of-other-variables-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Analysis of other variables</a></span><ul class="toc-item"><li><span><a href="#Correlation-matrix-Heatmap" data-toc-modified-id="Correlation-matrix-Heatmap-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Correlation matrix Heatmap</a></span></li></ul></li><li><span><a href="#Missing-data" data-toc-modified-id="Missing-data-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Missing data</a></span></li><li><span><a href="#Considering-effect-of-Outliers" data-toc-modified-id="Considering-effect-of-Outliers-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Considering effect of Outliers</a></span><ul class="toc-item"><li><span><a href="#Univariate-analysis" data-toc-modified-id="Univariate-analysis-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Univariate analysis</a></span></li><li><span><a href="#Bivariate-analysis" data-toc-modified-id="Bivariate-analysis-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Bivariate analysis</a></span></li></ul></li><li><span><a href="#Target-Assumptions" data-toc-modified-id="Target-Assumptions-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Target Assumptions</a></span><ul class="toc-item"><li><span><a href="#Normality" data-toc-modified-id="Normality-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Normality</a></span></li><li><span><a href="#homoscedasticity" data-toc-modified-id="homoscedasticity-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>homoscedasticity</a></span></li><li><span><a href="#Skewed-Features" data-toc-modified-id="Skewed-Features-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>Skewed Features</a></span></li><li><span><a href="#Temporal-variables" data-toc-modified-id="Temporal-variables-7.4"><span class="toc-item-num">7.4&nbsp;&nbsp;</span>Temporal variables</a></span></li></ul></li><li><span><a href="#Types-of-numerical-variables" data-toc-modified-id="Types-of-numerical-variables-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Types of numerical variables</a></span><ul class="toc-item"><li><span><a href="#Discrete-numerical-values" data-toc-modified-id="Discrete-numerical-values-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Discrete numerical values</a></span></li><li><span><a href="#Continuous-numerical-values" data-toc-modified-id="Continuous-numerical-values-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>Continuous numerical values</a></span><ul class="toc-item"><li><span><a href="#Performing-Logarithmic-transformation-to-the-data" data-toc-modified-id="Performing-Logarithmic-transformation-to-the-data-8.2.1"><span class="toc-item-num">8.2.1&nbsp;&nbsp;</span>Performing Logarithmic transformation to the data</a></span></li></ul></li></ul></li><li><span><a href="#Categorical-variables" data-toc-modified-id="Categorical-variables-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Categorical variables</a></span></li></ul></div>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

#Some styling
sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")
pd.pandas.set_option('display.max_columns', None)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# set dataset path

train_data = os.path.join("/kaggle/input/house-prices-advanced-regression-techniques", "train.csv")
test_data = os.path.join("/kaggle/input/house-prices-advanced-regression-techniques", "test.csv")


# In[ ]:


# read datasets

train = pd.read_csv(train_data)
test = pd.read_csv(test_data)


# In[ ]:


# size of dataset
# as 81 features so we can try different feature selection and feature engineering methods

train.shape


# In[ ]:


# display info about the dataset

train.columns


# In[ ]:


# sneak peek

train.head()


# # EDA

# # Personal selection and understanding of features

# As suggested in the notebook before starting it would be a good idea to to explore the data description file given as it would help to better understand the available features. It will also be helpful because we can see that there are **81** features present and some of them won't provide useful information in terms of the dependant feature i.e. `SalePrice`.

# After going through the proces, we can clearly that the features as selected by [Pedro Marcelino](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) are completely valid with the addition of one more i.e. `TotRmsAbvGrd` could also be an important factor for prediction of SalePrice.
# As pointed out in the kernel, `Neigborhood` feature didn't appeal as much information than the selected ones.
# Features to inspect:
# * OverallQual
# * YearBuilt.
# * TotalBsmtSF.
# * GrLivArea
# * TotRmsAbvGrd

# # Taking a look at SalePrice

# In[ ]:


# let's take look at stats for SalePrice

train["SalePrice"].describe()


# Without even plotting the histogram or KDE we can clearly see that the feature is skewed as the `mean` and meadian i.e.`25%` values are different, also we can see that `75%` of the prices values are between **35000 - 215000** but the max value is way of the scale than it should be. Let's plot this to see if our conclusion is True.

# In[ ]:


_ = sns.distplot(train.SalePrice)


# And...we were right. The data is right skewed(tail is on the right hand side).
# Let's look at how skewed the data is.

# In[ ]:


#skewness and kurtosis

print(f"Skewness: {train['SalePrice'].skew()}")
print(f"Kurtosis: {train['SalePrice'].kurt()}")


# Woah, it's more badly skewed than I hoped it would be, to negate this effect we would have to apply some kind of transformation to the feature viz. `log` or `BoxCox` during the feature engineering process.

# ## Relationship between selected features

# ### Relation with numerical variables

# In[ ]:


#scatter plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(train['GrLivArea'], train['SalePrice'], c='red', alpha=0.25)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.ylim(0,800000)

plt.subplot(1, 2, 2)
plt.scatter(train['TotalBsmtSF'], train['SalePrice'], c='k', alpha=0.25)
plt.xlabel('TotalBsmtSF')
plt.ylabel('SalePrice')
plt.ylim(0,800000)
plt.show()


# We can see that there is a linear relationship between `SalePrice` and `GrLivArea` whereas there's a slight exponential relationship with `TotalBsmtSF`

# ### Relation with categorical variables

# In[ ]:


plt.figure(figsize=(8, 6))
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
_ = sns.boxplot(x=var, y="SalePrice", data=data)
plt.axis(ymin=0, ymax=800000)
plt.show()


# The plot just confirms the fact the higher quality, higher the price.

# In[ ]:


plt.figure(figsize=(20, 8))
var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
_ = plt.xticks(rotation=90)


# Though we can never quantify this for sure, but more recently built houses will have a high price.

# In[ ]:


plt.figure(figsize=(20, 6))
var = 'TotRmsAbvGrd'
plt.subplot(1, 2, 1)
data = pd.concat([train['SalePrice'], train[var]], axis=1)
_ = sns.boxplot(x=var, y="SalePrice", data=data)
plt.axis(ymin=0, ymax=800000)
plt.show()


# It's natural the more room the house has the higher the price.

# **Summary:**
# * `GrLivArea` and `TotalBsmtSF` seem to be linearly related with 'SalePrice'. Both relationships are positive.
# * `OverallQual`, `TotRmsAbvGrd` and `YearBuilt` also seem to be related with 'SalePrice'. The relationship seems to be stronger in the case of `OverallQual`, where the box plot shows how sales prices increase with the overall quality. The same conclusion can be reached with `TotRmsAbvGrd`, though it's not as stronger as `OverallQual`.
# 

# # Analysis of other variables
# 
# So far we only took the variables that we subjectively chose and made performed analysis about them, but it's the best approach, we need to look at the available data subjectively.

# ## Correlation matrix Heatmap
# 
# 

# all numerical features

# In[ ]:


plt.figure(figsize=(18, 10))
corrmat = train.drop('Id', 1).corr()
_ = sns.heatmap(corrmat, vmax=1.0, square=True, fmt='.2f', 
            cmap='coolwarm', annot_kws={'size': 8})
plt.show()


# **From the kernel:**
# 
# 
# At first sight, there are two red colored squares that get my attention. The first one refers to the `TotalBsmtSF` and `1stFlrSF` variables, and the second one refers to the `GarageX` variables. Both cases show how significant the correlation is between these variables. Actually, this correlation is so strong that it can indicate a situation of multicollinearity. If we think about these variables, we can conclude that they give almost the same information so multicollinearity really occurs. Heatmaps are great to detect this kind of situations and in problems dominated by feature selection, like ours, they are an essential tool.
# 
# Another thing that got my attention was the 'SalePrice' correlations. We can see our well-known `GrLivArea`, `TotalBsmtSF`, and `OverallQual` saying a big 'Hi!', but we can also see many other variables that should be taken into account. That's what we will do next.

# In[ ]:


plt.figure(figsize=(6, 6))

k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', cmap='coolwarm',
                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# * `OverallQual`, `GrLivArea` and `TotalBsmtSF` are strongly correlated with `SalePrice`.
# * `TotRmsAbvGrd` and `GrLivArea` are strongly correlated each other. We can drop `TotRmsAbvGrd` as it's not as correlated with `SalePrice` than `GrLivArea`.
# * `GarageCars` and `GarageArea` are also some of the most strongly correlated variables. However, as we discussed in the last sub-point, the number of cars that fit into the garage is a consequence of the garage area. `GarageCars` and `GarageArea` are like twin brothers. You'll never be able to distinguish them. Therefore, we just need one of these variables in our analysis (we can keep 'GarageCars' since its correlation with 'SalePrice' is higher).
# * `TotalBsmtSF` and `1stFloor` also seem to be twin brothers. We can keep `TotalBsmtSF` just to say that our first guess was right.
# * `FullBath` seems odd but expected
# * `TotRmsAbvGrd` and `GrLivArea` are also highly correlated with each other. 
# * `YearBuilt`: It seems that `YearBuilt` is slightly correlated with `SalePrice`. Honestly, it scares me to think about `YearBuilt` because I start feeling that we should do a little bit of time-series analysis to get this right. I'll leave this as a homework for you.

# Let's find the continuous numerical variables in out dataset and plot a correlation matrix Heatmap

# In[ ]:


numerical_features = train.select_dtypes('number').columns.to_list()
numerical_features.pop(0)
print(f"Number of numerical features: {len(numerical_features)}")

# although numerical there are features which reprsent years
# there are some dicrete numerical features too.
# we need to remove them from the list.


year_features = [feature for feature in numerical_features 
                 if 'Yr' in feature or 'Year' in feature]
print(f"Number of Temporal features: {len(year_features)}")


discrete_features = [feature for feature in numerical_features 
                     if train[feature].nunique()<= 15 and feature not in year_features]
print(f"Number of discrete numerical features: {len(discrete_features)}")

continuous_num_features = [feature for feature in numerical_features 
                     if feature not in discrete_features + year_features] 
                                               
print(f"Number of continuous numerical features: {len(continuous_num_features)}")


# In[ ]:


plt.figure(figsize=(15, 10))
corrmat = train[continuous_num_features].corr()
sns.heatmap(corrmat, vmax=1.0, square=True, fmt='.2f', 
            annot=True, cmap='coolwarm', annot_kws={'size': 8});


# Some of the continuous numerical features seem to be correlated with each other though not as strongly. Though some new correlation are noticeable we still reach the same conclusions as above

# In[ ]:


# SalePrice scatter plot with highly correlated features


# In[ ]:


plt.figure(figsize=(10, 8))

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
_ = sns.pairplot(train[cols], size = 2.5, diag_kind='kde')
plt.show()


# # Missing data

# In[ ]:


features_with_na = [features for features in train.columns if train[features].isnull().sum() >= 1]

a = pd.DataFrame({
    'features': features_with_na,
    'Total': [train[i].isnull().sum() for i in features_with_na],
    'Missing_PCT': [np.round(train[i].isnull().sum()/ train.shape[0], 4) for i in features_with_na]
}).sort_values(by='Missing_PCT', ascending=False).reset_index(drop=True)
a.style.background_gradient(cmap='Reds') 


# * We can either delete the features with more than 50$ missing values or can fill them with appropriate values. None of these variables seem to be very important, since most of them are not aspects in which we think about when buying a house
# * The `GarageX` variables have the same number of missing data.Since the most important information regarding garages is expressed by `GarageCars` and considering that we are just talking about 5% of missing data, the same logic can be applied to the `BSMTX` variables.
# * For `MasVnrArea` and `MasVnrType`, these variables can be considered as not essential. Furthermore, they have a strong correlation with `YearBuilt` and `OverallQual` which are already considered. 
# * `Electrical` has just one missing value we can either delete that row or fill it with mode value.

# ##Examining Missing Features
# 
# 1. `FirePlaceQu`: 690 missing values

# In[ ]:


train['FireplaceQu'].value_counts()


# We only have half of Fireplace quality data. 
# 
# Let's have a look at the Fireplaces feature.

# In[ ]:


train[train['FireplaceQu'].isnull()][['Fireplaces','FireplaceQu']]


# * Looks like `FireplaceQu` is missing at places where `FirePlaces` feature is missing.
# * We can't do anything but fill these values with "Not avaliable".
# 
# 2. `MasVnrType`: 8 missing values
# 

# In[ ]:


#Unique elements
train['MasVnrType'].unique()


# In[ ]:


train[train['MasVnrType'].isnull()][['MasVnrType','MasVnrArea']]


# In[ ]:


# Let's look at the repeated value in MasVnrType column

train['MasVnrType'].mode()


# If we look at the types of masonry venner and their corresponding area,
# for all the missing values.
# 
# Area is zero.
# 
# So we can fill these missing values with "None"

# 3. `Bsmt` variable
# 
# * `BsmtQual`: 37 missing values
# * `BsmtCond`: 37 missing values 
# * `BsmtExposure`: 38 missing values
# * `BsmtFinType1`: 37 missing values
# * `BsmtFinType1`: 38 missing values
# 
# 

# In[ ]:


train[train['BsmtQual'].isnull()][['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1',
                        'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']].head(15)


# In[ ]:


train[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
 'BsmtFinType2']].mode()


# * The missing values would most probably  be because there is no basement.
# * We can fill the missing values with _No Basement_
# 

# 4. Electrical: 1 missing value
# 
# We can either just delete that row or fill the missing row with the mode of the feature

# 5. Garage
# * `GarageType`: 81 missing values
# * `GarageFinish`: 81 missing values
# * `GarageQual`: 81 missing values
# * `GarageCond`: 81 missing values
# 

# In[ ]:


train[train['GarageType'].isnull()][['GarageType', 'GarageYrBlt', 'GarageFinish',
       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']]


# In[ ]:


train[['GarageType','GarageFinish',
 'GarageQual','GarageCond']].mode()


# * The GarageX variables have the same number of missing data.
# * We can simply fill the variables with _No garage_.
# * But mostly these values are correlateed and `GarageCars` variable can only be used.

# Check if the columns with missing values has some valuable relationship with the outputs.

# # Considering effect of Outliers

# In[ ]:


plt.figure(figsize=(20, 20))
for i, feature in enumerate(features_with_na, 1):
    plt.subplot(5, 5, i)
    data = train.copy()
    
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    
    # calculate the median SalePrice where the information is missing or present
    temp = data.groupby(feature)['SalePrice'].median()
    _ = sns.barplot(x=temp.index, y=temp.values)
    plt.title(feature)
    plt.xlabel("")
    plt.xticks([0, 1], ["Present", "Missing"])
    plt.ylabel("Sales Price", rotation=90)
plt.tight_layout(h_pad=2, w_pad=2)
plt.show()


# **Observation**
# * We can see that for some features when the value is *NA*, the median value for that set of columns is higher than when values are available.
# * So it is safe to assume that the missing values need to be replaced with some meaningful values.

# ## Univariate analysis

# For labelling a point as an *outlier* we need to define a threshold value that defines the datapoint as an outlier. We can do this bu standardizing the data.

# In[ ]:


#standardizing data
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# How 'SalePrice' looks with her new clothes:
# 
# * Low range values are similar and not too far from 0.
# * High range values are far from 0 and the 7.something values are really out of range.
# 
# For now, we'll not consider any of these values as an outlier but we should be careful with those two 7.something values.

# Let's examine the outliers present in continuous features after applying a log transformation 

# In[ ]:


data = train.copy()
# sale_price = np.log(train['SalePrice'])

for i, feature in enumerate(continuous_num_features, 1):
    data = train[feature].copy()
    if 0 in data.unique(): # as log 0 is undefinedz
        continue
    else:
        data = np.log(data)   
        data.name = feature
        _ = plt.figure(figsize=(6, 6))
        _ = sns.boxplot(y=data)
    
plt.show()


# **Observations**:
# * There are plenty of outliers present even after log transformation in every featufres.
# * These outliers will most likely interfere during the model building process.
# * We will need to handle them during feature engineering
# * There are a few techniques for outlier handling:
# 1. Outlier removal
# 2. Treating outliers as missing values
# 3. Top / bottom / zero coding
# 4. Discretisation
# 

# ## Bivariate analysis

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# We can see at the bottom right two with extremely large GrLivArea that are of a low price. These values are huge oultliers. Therefore, we can safely delete them.

# # Target Assumptions
# 
# **Question:** Who is `SalePrice`?
# 
# 
# The answer to this question lies in testing for the assumptions underlying the statistical bases for multivariate analysis. We already did some data cleaning and discovered a lot about `SalePrice`. Now it's time to go deep and understand how 'SalePrice' complies with the statistical assumptions that enables us to apply multivariate techniques.
# 

# According to [Hair et al. (2013)](https://amzn.to/2uC3j9p), four assumptions should be tested:
# 
# * **Normality** - When we talk about normality what we mean is that the data should look like a normal distribution. This is important because several statistic tests rely on this (e.g. t-statistics). In this exercise we'll just check univariate normality for 'SalePrice' (which is a limited approach). Remember that univariate normality doesn't ensure multivariate normality (which is what we would like to have), but it helps. Another detail to take into account is that in big samples (>200 observations) normality is not such an issue. However, if we solve normality, we avoid a lot of other problems (e.g. heteroscedacity) so that's the main reason why we are doing this analysis.
# 
# * **Homoscedasticity** - I just hope I wrote it right. Homoscedasticity refers to the 'assumption that dependent variable(s) exhibit equal levels of variance across the range of predictor variable(s)' (Hair et al., 2013). Homoscedasticity is desirable because we want the error term to be the same across all values of the independent variables.
# 
# * **Linearity**- The most common way to assess linearity is to examine scatter plots and search for linear patterns. If patterns are not linear, it would be worthwhile to explore data transformations. However, we'll not get into this because most of the scatter plots we've seen appear to have linear relationships.
# 
# * **Absence of correlated errors** - Correlated errors, like the definition suggests, happen when one error is correlated to another. For instance, if one positive error makes a negative error systematically, it means that there's a relationship between these variables. This occurs often in time series, where some patterns are time related. We'll also not get into this. However, if you detect something, try to add a variable that can explain the effect you're getting. That's the most common solution for correlated errors.

# ## Normality
# 
# The point here is to test 'SalePrice' in a very lean way. We'll do this paying attention to:
# 
# * **Histogram** - Kurtosis and skewness.
# * **Normal probability plot** - Data distribution should closely follow the diagonal that represents the normal distribution.

# In[ ]:


train.select_dtypes('number').columns


# In[ ]:


sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend([f'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )'],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# `SalePrice` is not normal. It shows 'peakedness', positive skewness and does not follow the diagonal line.
# 
# A simple data transformation can solve the problem. Similarly with `GrLivArea`

# In[ ]:


# GrLivArea
# histogram and normal probability plot

sns.distplot(train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)


# In[ ]:


# TotalBsmtSF
# histogram and normal probability plot

sns.distplot(train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'], plot=plt)


# **Observations:**
# * We can see the skewness present.
# * We can also see that many of the values are zero.
# * We cannot simply apply the log transformation on this.
# * What we can do is can do is create a variable that can get the effect of having or not having basement (binary variable). Then, we'll do a log transformation to all the non-zero observations, ignoring those with value zero. This way we can transform data, without losing the effect of having or not basement.

# Eg.
# ```python
# #if area>0 it gets 1, for area==0 it gets 0
# train['HasBsmt'] = 0 
# train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1
# 
# # transform data
# train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])
# ```

# ## homoscedasticity
# 
# The best approach to test homoscedasticity for two metric variables is graphically. Departures from an equal dispersion are shown by such shapes as cones (small dispersion at one side of the graph, large dispersion at the opposite side) or diamonds (a large number of points at the center of the distribution).

# ## Skewed Features

# In[ ]:


print("\nSkew in numerical features: \n")
pd.DataFrame({
    'Feature': numerical_features,
    'Skewness': skew(train[numerical_features])}).sort_values(by='Skewness',
                                                              ascending=False).set_index('Feature').head(10)


# * We can see a lot of numerical features are heavily skewed.
# * We would need to apply a **log** or **box cox** transformations to these features.

# **[NOTE]**: We'll look at feature transformation during feature engineering process.

# ## Temporal variables
# 
# * There are total of 4 datetime variables/columns in the dataset.
# * These columns are useful for prediction as we can assume that there will be changes in price values depending upon time.

# In[ ]:


train[year_features].head()


# `SalePrice` w.r.t. `YrSold` 

# In[ ]:


plt.figure(figsize=(15, 9))
for i, feature in enumerate(year_features, 1):
    plt.subplot(2, 2, i)
    temp = train.groupby(feature)['SalePrice'].median().plot()
    plt.xlabel(feature)
    # plt.xlim([2006, 2010])
    # plt.xticks(range(2006, 2011))
    plt.title(f"Median price vs {feature}")
    plt.ylabel("Sale price", rotation=90)
plt.tight_layout(w_pad=1.2, h_pad=1.2)
plt.show()


# **Observations**
# * We can see an unusual trend between Sales price and Year Sold, typically the prices of house increases with time but from the graph we can see the opposite.
# * One reason could due to the financial drought of '08 when in the US collapsed.
# * We can see that the houses and garage which were built and remoddeled during 90s have less Sale Price than the newer ones.
# * With every year new houses that are built, the house price increases.

# Calculate the difference between all `year_features` with `SalePrice`. Doing this will allow us to get the `age` of the houses and with the help of scatter plot can see the relationship between `SalePrice` with the `age` of the house

# In[ ]:


plt.figure(figsize=(20, 6))

for i, feature in enumerate(year_features, 1):
    
    if feature != 'YrSold':
        data = train.copy()
        
        data[feature] = data['YrSold'] - data[feature]
        plt.subplot(1, 3, i)
#         plt.scatter()
        plt.title(feature)
        
        plt.ylabel('SalePrice')
        sns.regplot(data[feature], data['SalePrice'], 
                   scatter_kws={"color": "black"}, line_kws={"color": "red"})
        plt.xlabel(f"No. of years: {feature}")
plt.tight_layout()
plt.show()        


# **Observation:**
# * What we did was take a look at the `SalePrice` values through the years for different features.
# * We can see that although there were different intervals, the regression line passing is almost the same.
# * We can see that new houses built or remodelled or had a newly built garage were sold for higher prices.

# In[ ]:


sns.heatmap(pd.DataFrame({
    'SalePrice': train['SalePrice'],
    'YearBuiltAge': train['YrSold'] - train['YearBuilt'],
    'YearRemodAddAge': train['YrSold'] - train['YearRemodAdd'],
    'GarageYrBltAge': train['YrSold'] - train['GarageYrBlt'],
}).corr(), annot=True, cmap='coolwarm')
plt.title('Features as age vs SalePrice')
plt.show()


# **Observations**
# * We can clearly see that time interval of `YrBuilt` and `GarageYrBuild` are highly correlated with each other, so is `YrRemodAdd` with `GarageYrBuild`.
# * Interval of `YrBuilt` is not correlated with  `SalePrice`.
# * During feature engineering we can convert these temporal features to represent the interval.
# * We can also see that time intervals of `YrBuilt` and `YrRemodAdd` are highly negatively correlated with `SalePrice`. 
# * If we need to keep one of them, from a subjective point of view I'll choose to keep the `YrRemodAdd` as from my perspective, the year the house was built though important we will most likely ask for the date the house was remodelled as a way to get info has an effect on the price of the house. We can use different models that indicate feature importance to check this out.

# # Types of numerical variables
# 1. Continuous
# 2. Discrete

# ## Discrete numerical values

# In[ ]:


discrete_features


# In[ ]:


train[discrete_features].head()


# relationship between discrete features and SalePrice

# In[ ]:


plt.figure(figsize=(20, 20))
for i, feature in enumerate(discrete_features, 1):
    plt.subplot(6, 3, i)
    data = train.copy()
    
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    
    # calculate the median SalePrice where the information is missing or present
    temp = data.groupby(feature)['SalePrice'].median()
    _ = sns.barplot(x=temp.index, y=temp.values)
    plt.title(feature)
    plt.xlabel("")
    plt.ylabel("Sales Price", rotation=90)
    plt.xticks(rotation=45)
plt.tight_layout(h_pad=2, w_pad=2)
plt.show()


# **Observations**:
# * It's clear that the values of discrete numerical features has an effect on the SalePrice.
# * At some places we can see that as the feature value increases the price also increases
# * At some places the bars has pretty high for only some values.
# * We can see some features that have almost the same SalePrice for it's different values
# * Looking from the top, it's not clear how most of the features will have an impact on the SalePrice.
# * Feature selection process might be able to clear this up.

# ## Continuous numerical values

# In[ ]:


print(f"Number of Continuous numerical feature: {len(continuous_num_features)}")


# In[ ]:


continuous_num_features


# In[ ]:


train[continuous_num_features].head()


# In[ ]:


plt.figure(figsize=(20, 20))
for i, feature in enumerate(continuous_num_features, 1):
    plt.subplot(5, 4, i)
#     _ = sns.distplot(train[feature], kde_kws={'bw': 1.05})    
    _ = sns.distplot(train[feature], kde=False, rug=True)
    
plt.tight_layout(h_pad=2, w_pad=2)
plt.show()


# **Observations:**
# * For starters we can see that the most of the continuous features don't follow a normal distribution.
# * Some of the features are heavily skewed so we would have to perform some transformation to them.

# ### Performing Logarithmic transformation to the data

# In[ ]:


data = train.copy()

sale_price = np.log(train['SalePrice'])

for i, feature in enumerate(continuous_num_features[:-1], 1):
    data = train[feature].copy()
    
    if 0 in data.unique(): # as log 0 is undefinedz
        continue
    else:
        data = np.log(data)    
        data.name = feature
        _ = plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        sns.regplot(data, sale_price, fit_reg=True,
                   scatter_kws={"color": "black"}, line_kws={"color": "red"}).set_title(f"Correlation: {data.corr(sale_price)}")
        plt.subplot(1, 2, 2)
        sns.distplot(data).set_title(f"Log transformation of: {feature}")
plt.show()


# **Observations:**
# * Though the data has been transformed we can see some irregulaties for eg. `LotArea` looking at the scatter we can see that for some _same_ values the SalePrice increases. We can see that there's not much of correlation between the two
# * log(`GrLivArea`) is heavily correlated with `SalePrice` and so is `1stFlrSF`
# * We were not able to take log of other continuous features as `log(0)` is undefined

# # Categorical variables

# In[ ]:


categorical_features = [feature for feature in train.columns if train[feature].dtypes=='O']
print(f"Number of categorical feature: {len(categorical_features)}")


# In[ ]:


categorical_features


# In[ ]:


train[categorical_features].head()


# In[ ]:


pd.DataFrame({
    "features": categorical_features,
    "Nunique": [train[feature].nunique() for feature in categorical_features]             
             })


# Find out the relationship between categorical features and SalePrice

# In[ ]:


plt.figure(figsize=(40, 30))

for i, feature in enumerate(categorical_features, 1):
    data = train.copy()
    temp = data.groupby(feature)['SalePrice'].median()
    plt.subplot(9, 5, i)
    sns.barplot(temp.index, temp.values)
    plt.xticks(rotation=45)

plt.tight_layout(h_pad=1.2)
plt.show()


# **Insights on all categorical features**
# * After reading features decription the categorical features, my conclusions are:
# 1. `MSZoning`: Tells about the zone the house is located in, but we already have a neigborhood variable that'll be more effective.
# 2. `Street`, `Alley` and `LotConfig` don't seem that important.
# 3. `LandContour`,  `LotShape`, `LandSlope`: are basically giving the same information, they might be correlated with each other too.
# 4. `Utilities`: is an important feature as it generally affects the price of a house.
# 5. `Neigborhood`: Location is important while buying a house though we ended up discarding it in the initial selection as most of the SalePrice values are have overlapping with each other.
# 6. `Condition1` and `Condition2`: `Condition1` seems a good choice.
# 7. `BldgType` and `HouseStyle`: Both are important factors for pricing.
# 8. `RoofStyle`, `RoofMatl`: Does it matter as long is it sturdy?. 
# 9. `Exterior1st`, `Exterior2nd`: Both seem to have effect on the Sales Price.
# 10. `MasVnrType`: From above it does seem to be important (feature selection will help in determining this).
# 11. `ExterQual` and `ExterCond`: `ExterCond` seems a likely choice as the current condition of the externals determines the price more than what it was.
# 12. `Foundation`: one of the main ingredients in determing house prices.
# 13. `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`: Earlier during missing value analysis there was consistency in  the rows that matched. These features might be correlated with each other so we can either take just one of two features that represents the `BsmtX` variables and have high correlation with the `SalePrice`. `BsmtQual` and `BsmtExposure` seem optimal choices.
# 14. `Heating`, `HeatingQC`: `HeatingQC` seems a good choice of the two.
# 15. `CentralAir`, `Electrical`: Though not many of us would ask the type of electrical system in the house as an initial question but information about the conditioning might play a role in determining the price.
# 16. `KitchenQual`: an important factor we would all agree on.
# 17. `FireplaceQu`: It has a lot of missing values as if fireplace is not present, but does affect the pricing of the house if present.
# 18. `GarageType`, `GarageFinish`, `GarageQual`,  `GarageCond`: as dicussed earlier to represent all these features `GarageCars` might be sufficienct.
# 19. `PoolQC`, `Fence`: too many missing values.
# 20. `MiscFeature`: `MiscVal` feature might be all that's necessary.
# 21. `SaleType` and `SaleCondition`: Personally don't think would affect the Sale price value of the house that much.
# 22. `PavedDrive`: Surroundings do affect the price of a house.
# 

# We have gone through quite a number of differnt numerical, categorical, discrete, continuous and temporal features and have a plan on how to go around the feature engineering process.
# 
# In the next kernel we'll see how we can from the concepts learned from this notebook apply it to the feature engineering process and later the end goal of this project: model building 
# 
# Part 2: Kernel => [Project 2 P2: Model building](https://www.kaggle.com/veb101/project-2-p2-model-building)
