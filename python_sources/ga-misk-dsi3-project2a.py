#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from scipy import stats
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load in dataset
train = pd.read_csv('/Users/afsaja/Desktop/dsi3/dsi3_projects/project_2/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/Users/afsaja/Desktop/dsi3/dsi3_projects/project_2/house-prices-advanced-regression-techniques/test.csv')


# # Data exploration

# In[ ]:


# Exploring the dataset
train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape, test.shape


# In[ ]:


train.info(), test.info()


# In[ ]:


# Removing the 'Id' column from train and test dataset as it provides no additional value and savng it separately
train_id = train[['Id']]
test_id = test[['Id']]

#Dropping 'Id' column
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)


# In[ ]:


train.shape, test.shape


# # Data cleaning

# Instead of graphing a pairplot, we've taken advantage of Kaggle's 'Data' section that has summaries and distributions of each feature

# ### Removing outliers

# In[ ]:


sns.scatterplot(train['GrLivArea'], train['SalePrice']);


# In[ ]:


# Seeing that we have two GrLivArea observations that are significantly higher than the average with low SalePrices, 
# we decide to remove them
train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index, inplace=True)
train.shape


# In[ ]:


# We realize a better scatter plot of SalePrice and GrLivArea after removing outliers 
sns.scatterplot(train['GrLivArea'], train['SalePrice']);


# ### Inspecting our target variable

# In[ ]:


# Below dist plot shows skewness in the target variable making it a candidate for log transformation
sns.distplot(train['SalePrice'], kde=True, bins=20);


# In[ ]:


# Further investigating skewness (measures symmetry) and kurtosis (fat tails) shows that SalePrice is NOT normal
train['SalePrice'].skew(), train['SalePrice'].kurt()


# In[ ]:


# log transforming our target variable and checking resulting histogram
train['SalePrice'] = np.log(train['SalePrice'])
sns.distplot(train['SalePrice'], bins=20, kde=True);


# ### Dealing with missing values and feature engineering

# In[ ]:


# Going through the description on Kaggle, many of the NaN values in the features should be filled with 'None' or 
# ZERO with some to be categorized. Below, we go through each of them feature by feature

# First, we combine the train and test sets to ensure that both sets are taken treated the same with 
# similar columns dropped (if any)
combined = pd.concat((train, test)).reset_index(drop=True)
combined.shape


# In[ ]:


# Setting y_train and removing 'SalePrice' from combined dataset so as not to lose 'SalePrice' column
y_train = train['SalePrice']
combined.drop('SalePrice', axis=1, inplace=True)
combined.shape, y_train.shape


# In[ ]:


# Now we go through each feature and drop or fill in NaNs based on documentation. We generate corr on train data given
# that it still has the SalePrice column
corrmat = train.corr()
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corrmat, vmax=.8, square=True, cmap="PiYG", annot=True, fmt='.1f');


# In[ ]:


# PoolQC : data description says NA means "No Pool"
combined['PoolQC'].fillna('None', inplace=True)

# MiscFeature: data description says NA means "No misc value"
combined['MiscFeature'].fillna('None', inplace=True)

# Alley : data description says NA means "No alley access"
combined['Alley'].fillna('None', inplace=True)

# Fence : data description says NA means "No Fence"
combined['Fence'].fillna('None', inplace=True)

# FireplaceQu : data description says NA means "No Fireplace"
combined['FireplaceQu'].fillna('None', inplace=True)

# we ran .describe() on LotFrontage and foudn the mean = median, we, hence, proceed to filling 
# NA values for LotFrontage with mean of column
combined.LotFrontage.fillna(value=combined['LotFrontage'].mean(), inplace=True)

# Garage-cat : replacing NAs "None"
garage_list_cat = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_list_cat:
    combined[col] = combined[col].fillna('None')

# Garage_num : replacing NAs with 0s
garage_list_num = ['GarageArea', 'GarageCars', 'GarageYrBlt']
for col in garage_list_num:
    combined[col] = combined[col].fillna(0)

# Basement_num : Replacing basement numericals with 0s
basement_list_num = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
for col in basement_list_num:
    combined[col] = combined[col].fillna(0)

# Basement_cat : Replacing basement categories with "None"
basement_list_num = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in basement_list_num:
    combined[col] = combined[col].fillna('None')
    
# Veneer data: replacing with 0s and 'None'
combined['MasVnrArea'].fillna(0, inplace=True)
combined['MasVnrType'].fillna('None', inplace=True)

# Zoning: Replacing with most common value 'RL'
combined['MSZoning'].fillna('RL', inplace=True)

# Utilities: Drop it since it has one type of observation and thus not helpful
combined.drop('Utilities', axis=1 ,inplace=True)

# Functional: Description says NA = 'Typ'
combined['Functional'].fillna('Typ', inplace=True)

# Electrical: Set missing value to 'SBrkr'
combined['Electrical'].fillna('SBrkr', inplace=True)

# KitchenQual: Only one NA value, and same as Electrical, we set 'TA'
combined['KitchenQual'].fillna('TA', inplace=True)

# Exterior1st and Exterior2nd : We will just substitute in the most common string
combined['Exterior1st'].fillna(combined['Exterior1st'].mode()[0], inplace=True)
combined['Exterior2nd'].fillna(combined['Exterior2nd'].mode()[0], inplace=True)

# SaleType : Fill in again with most frequent which is "WD"
combined['SaleType'].fillna(combined['SaleType'].mode()[0], inplace=True)

# SaleType : Fill in again with most frequent which is "WD"
combined['MSSubClass'].fillna('None', inplace=True)


# Fixing categorical values that have some numerical values

# In[ ]:


# MSSubClass=The building class
combined['MSSubClass'] = combined['MSSubClass'].apply(str)


# Changing OverallCond into a categorical variable
combined['OverallCond'] = combined['OverallCond'].astype(str)


# Year and month sold are transformed into categorical features.
combined['YrSold'] = combined['YrSold'].astype(str)
combined['MoSold'] = combined['MoSold'].astype(str)


# We want to check now proceed with dropping variables that are parts of the total (to reduce multicollinearity between variables)

# In[ ]:


# Dropping part of total columns
# Dropping basement and duplicated columns
columns_kept_train = [c for c in combined.columns if c not in ['BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF','GarageCars']]
combined = combined[columns_kept_train]

# Dropping GrLivArea parts
columns_kept_train = [c for c in combined.columns if c not in ['1stFlrSF', '2ndFlrSF','LowQualFinSF',]]
combined = combined[columns_kept_train]


# In[ ]:


combined.shape


# We now check the skewness of other features in the dataset that we think are important

# In[ ]:


subset = combined.copy()
skew_features = subset.skew(axis=0).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 10))
plt.xticks(rotation='vertical')
sns.barplot(x=skew_features.index, y=skew_features);


# The plot above shows that some features are highly skewed. We will not transform all skewed data but focus ONLY on the features that exhibit a skewness > 5. No reason why we chose 5 specifically. This is just arbitrary and seems like a good cut-off based on the bar plot above

# We proceed to plotting the histograms of the 5 features we identified as highly skewed to have a look at their distributions

# In[ ]:


# Plotting histograms of highly skewed numerical variables with SalesPrice to assess underlying distributions
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
sns.distplot(combined.MiscVal, bins=20, kde=True, ax=ax[0,0])
sns.distplot(combined.PoolArea, bins=20, kde=True, ax=ax[0,1])
sns.distplot(combined.LotArea, bins=20, kde=True, ax=ax[1,0])
sns.distplot(combined['3SsnPorch'], bins=20, kde=True, ax=ax[1,1])
fig.suptitle('Histograms of relevant numeric variables', fontsize=20)
plt.show()


# Hmmm... since the above distribution plots have shown that these features exhibit skewness due to the presence of zeros, we will not treat them given that log transformation in order to 'normalize' them would not work.

# Last step is getting dummy variables! We proceed with changing all our categoricals to numeric in order to perform linear regression on them using ```pd.get_dummies()```

# In[ ]:


# Making sure no column has NAs
combined.isnull().sum()


# In[ ]:


# Plotting correlationmap one more time
corrmat_combined = combined.corr()
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corrmat_combined, vmax=.8, square=True, cmap="PiYG", annot=True, fmt='.1f');


# In[ ]:


# Get dummy variables
model_data = combined.copy()
model_data = pd.get_dummies(model_data)
model_data.shape


# # Modeling

# ## Resplitting and Scaling the data (preprocessing)

# In[ ]:


# Resplitting our data back into test and train
X_train = model_data[:train.shape[0]]
X_test = model_data[train.shape[0]:]
X_test.shape, X_train.shape


# In[ ]:


# Scaling the data using StandardScaler

ss = StandardScaler()
X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.transform(X_test)


# ## Ridge regression using RidgeCV

# In[ ]:


# create an array of alpha values
alpha_range = 10.**np.arange(-2, 3)
alpha_range

# select the best alpha with RidgeCV
ridgeregcv = RidgeCV(alphas=alpha_range, cv=5)
ridgeregcv.fit(X_train_ss, y_train)


# In[ ]:


# Printing out RidgeCV results
print('RidgeCV alpha: ', str(ridgeregcv.alpha_))


# In[ ]:


# predict method uses the best alpha value
y_pred_log = ridgeregcv.predict(X_test_ss)
y_pred_log


# In[ ]:


# Scaling back RidgeCV results to original SalePrice scale
y_pred = np.expm1(y_pred_log)
y_pred


# In[ ]:


# printing out CSV file to test
ridgecv_submission = pd.DataFrame({'Id': test_id.Id, 'SalePrice': y_pred})
ridgecv_submission.to_csv('ridgecv_submission.csv', index=False)


# ## LASSO regression using LassoCV

# In[ ]:


# select the best alpha with LassoCV
lassoregcv = LassoCV(n_alphas=100, random_state=87, cv=5)
lassoregcv.fit(X_train, y_train)


# In[ ]:


# examine the coefficients
print('LASSO penalization: ', str(lassoregcv.alpha_))
print('LASSO coefs: ', str(lassoregcv.coef_[:5]))
print('LASSO alphas: ', str(lassoregcv.alphas_[:5]))
print('LASSO MES path: ', str(lassoregcv.mse_path_[:5]))


# In[ ]:


# predict method uses the best alpha value
y_pred_log_lasso = lassoregcv.predict(X_test)
y_pred_log_lasso


# In[ ]:


# Scaling back RidgeCV results to original SalePrice scale
y_pred_lasso = np.expm1(y_pred_log_lasso)
y_pred_lasso


# In[ ]:


# printing out CSV file to test
lassocv_submission = pd.DataFrame({'Id': test_id.Id, 'SalePrice': y_pred_lasso})
lassocv_submission.to_csv('lassocv_submission.csv', index=False)


# We have submitted both results to Kaggle and the output of both models was tested:
# - RidgeCV rmse = **0.13569**
# - LassoCV rmse = **0.19058**
# 
# It seems that **RidgeCV performs better than LassoCV** in this case. Why is that? I'm not quite sure.
# 
# We are not done yet! To test if we can get a better score, we **perform additional feature engineer** focused on the following:
# 
# - Simplifying some of our categorical variables (ones with large number of categories)
# - Summing up ```GrLivArea``` and ```TotalBsmtSF``` to generate a total living space column and drop their parts accordingly
# - Summing up other features like ```Porch features```
# - Using Cox-Box transformation to transform some of our highly skewed data further
# 
# Fingers crossed! Let's do it!
