#!/usr/bin/env python
# coding: utf-8

# # 0. Project Introduction
# 
# This is a data analysis project using Python and several regression models.
# The whole process can be divided into three parts:
# 
# 1. Data inspection and plotting
# 2. Data preprocessing and feature engineering
# 3. Modelling and making predictions (including cross validation and hyper parameter tuning)
# 

# # 1. Data inspection and plotting
# 
# First, let's load the data and take a look at the data distribution and correlation.
# 

# In[ ]:


# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


# Load data sets
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test_ID = df_test['Id']


# In[ ]:


# Print the first 10 rows
df_train.head(10)


# We can see that there are multiple features (Alley, PoolArea, etc) containing invalid data that can not serve as model inputs. These NaNs and NAs will be dealed with later. 

# In[ ]:


# Get descriptive stats
df_train.describe()


# In[ ]:


# Now drop the  'Id' column since it's unnecessary for  the prediction process.
df_train.drop("Id", axis=1, inplace=True)
df_test.drop("Id", axis=1, inplace=True)
print(df_train.columns)
print(df_train['SalePrice'].describe())


# We may not get a good grasp of the distribution of Sale Price column. So it is a good idea to plot a histogram.

# In[ ]:


# histogram
sns.distplot(df_train['SalePrice'])


# Obviously, sale prices do not conform to normal distribution; they are skewed.
# 
# Let's plot other features paired with sale price to see if they are correlated. 

# In[ ]:


# scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


# In[ ]:


# scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


# In[ ]:


# scatter plot LotFrontage/saleprice
var = 'LotFrontage'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


# In[ ]:


# scatter plot 1stFlrSF/saleprice
var = '1stFlrSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


# In[ ]:


# boxplot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data, palette=sns.color_palette("hls", 8))
fig.axis(ymin=0, ymax=800000)


# Overall quality ranking has a strong positive correlation with sale price. 

# In[ ]:


# boxplot YearBuilt/saleprice
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data, palette=sns.color_palette("hls", 8))
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)


# It is common sense that newer houses sell at a premium compared to those with some ages, which is evidenced by the graph. 

# In[ ]:


# correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

# saleprice correlation matrix
k = 10  # number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values,
                 xticklabels=cols.values)


# We create a 10x10 correlation matrix which shows the top 10 factors that are positively related to final sale price.All of these features are variables with regard to quality ranking, size of area, and age of the property, which is consistent with our life experience. 
# 
# # Important findings in Part 1:
# 1. Some features in the dataset have missing and/or invalid data. A feature engineering should be implemented to correct this.
# 2. Sale price, the most important feature as well as our prediction target, is negatively skewed (i.e. there are more low prices than high ones). 

# # 2. Data preprocessing and feature engineering

# In[ ]:


# Combine train and test set for preprocessing
X_all = pd.concat((df_train, df_test)).reset_index(drop=True)
X_all.drop(['SalePrice'], axis=1, inplace=True)
print("X_all size is : {}".format(X_all.shape))


# In[ ]:


# Handle missing values (NA, NaN)
# Check if any missing values
X_all_na = (X_all.isnull().sum() / len(X_all)) * 100
X_all_na = X_all_na.drop(X_all_na[X_all_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': X_all_na})
print(missing_data.head(10))


# The top 5 features have at least half of their data missing. To make our model more robust, we can remove them from our dataset.

# In[ ]:


# Drop features
X_all = X_all.drop(['PoolQC', "MiscFeature", "Alley", 'Fence', 'FireplaceQu'], axis=1)


# In[ ]:


# Fill features
X_all["MasVnrType"] = X_all["MasVnrType"].fillna("None")
X_all["MasVnrArea"] = X_all["MasVnrArea"].fillna(0)
X_all['MSZoning'] = X_all['MSZoning'].fillna(X_all['MSZoning'].mode()[0])
X_all["Functional"] = X_all["Functional"].fillna("Typ")
X_all['Electrical'] = X_all['Electrical'].fillna(X_all['Electrical'].mode()[0])
X_all['KitchenQual'] = X_all['KitchenQual'].fillna(X_all['KitchenQual'].mode()[0])
X_all['Exterior1st'] = X_all['Exterior1st'].fillna(X_all['Exterior1st'].mode()[0])
X_all['Exterior2nd'] = X_all['Exterior2nd'].fillna(X_all['Exterior2nd'].mode()[0])
X_all['SaleType'] = X_all['SaleType'].fillna(X_all['SaleType'].mode()[0])
X_all['MSSubClass'] = X_all['MSSubClass'].fillna("None")
X_all['Utilities'] = X_all['Utilities'].fillna('AllPub')
X_all["LotFrontage"] = X_all.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    X_all[col] = X_all[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    X_all[col] = X_all[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    X_all[col] = X_all[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    X_all[col] = X_all[col].fillna('None')


# In[ ]:


# Tranform numeric feature to categorical 
X_all['MSSubClass'] = X_all['MSSubClass'].apply(str)


# In[ ]:


# Check remaining missing values if any
X_all_na = (X_all.isnull().sum() / len(X_all)) * 100
X_all_na = X_all_na.drop(X_all_na[X_all_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio': X_all_na})
print("remaining missing value:", missing_data)


# After successfully dealing with missing data, we now spend some time processing the skewed features`

# In[ ]:


# Deal with highly skewed features
from scipy.stats import skew

numeric_feats = X_all.dtypes[X_all.dtypes != "object"].index
skewed_feats = X_all[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew': skewed_feats})
print(skewness)


# In[ ]:


skewness = skewness[abs(skewness) > 1]
print("There are {} skewed numerical features to transform".format(skewness.shape[0]))


# In[ ]:


skewed_features = skewness.index
for feat in skewed_features:
    X_all[feat] = np.log1p((X_all[feat]))


# This log1p transformation can be replaced by Box Cox transforming. However, after some testing on lambda value and comparing these two methods on my local computer, I do not see any apparent discrepancy between them. So, in this notebook, I choose the simpler and more intuitive way of log1p transformation. 

# In[ ]:


# Make categorical features dummies
X_all = pd.get_dummies(X_all)


# In[ ]:


# Create train and test set
X_train = X_all[:df_train.shape[0]].values
X_test = X_all[df_train.shape[0]:].values
y_train = np.log1p(df_train['SalePrice'])
print(X_train.shape, y_train.shape)


# # 3. Modelling and making predictions

# In[ ]:


from sklearn.model_selection import KFold, cross_val_score
kfolds = KFold(n_splits=10, shuffle=True, random_state=1)


# In[ ]:


# Define model
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=20000, learning_rate=0.02, max_depth=3, max_features='sqrt',
                                min_samples_leaf=5, min_samples_split=5, loss='ls')


# In[ ]:


# Define model hyper parameters for cross validation
gbr_param_grid = {
    "n_estimators": [40000, 60000, 80000],
    "learning_rate": [0.01, 0.02],
    "max_depth": [3],
    "max_features": ["sqrt"],
    "min_samples_leaf": [5],
    "min_samples_split": [5],
    "loss": ["ls", 'huber']
}


# In[ ]:


from sklearn.model_selection import  GridSearchCV
def grid_search(model, param_grid):
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=kfolds, scoring='r2', verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best: {0:.4f} using {1}".format(grid_search.best_score_, grid_search.best_params_))

    return grid_search


# In[ ]:


grid_search(gbr, gbr_param_grid)


# After doing cross validation to find the best parameter set with highest accuracy (and lowest CV error on the test set), we can predict the outcome of the test set and save the results to csv file. 

# In[ ]:


# Prediction

# Use the below parameters grid:
#              param_grid={'learning_rate': [0.02, 0.01], 'loss': ['ls', 'huber'],
#                          'max_depth': [3], 'max_features': ['sqrt'],
#                          'min_samples_leaf': [5], 'min_samples_split': [5],
#                          'n_estimators': [40000, 60000, 80000]},
    
# model_final = grid_search(knr, knr_param_grid)
# y_pred = np.expm1(np.expm1(model_final.predict(X_test)))
# y_test = pd.DataFrame()
# y_test['Id'] = test_id.values
# y_test['SalePrice'] = y_pred
# y_test.to_csv('submission.csv',index=False)

