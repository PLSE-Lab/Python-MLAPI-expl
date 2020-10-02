#!/usr/bin/env python
# coding: utf-8

# Boston House Price is a very good dataset to analyze Regression problem.
# 
# Start from the EXPLORATION of Multivariate Data Analysis:
# 
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# 
# And refer to the wonderful solution:
# 
# https://www.kaggle.com/jesucristo/1-house-prices-solution-top-1
# 
# Now we begin our Exploratory Data Analysis

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.stats import norm, skew
from scipy import stats
from sklearn.preprocessing import StandardScaler


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# ## Read Data

# In[ ]:


ls ../input


# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# In[ ]:


train.shape, test.shape


# In[ ]:


train.columns


# In[ ]:


# train.info()


# In[ ]:


train['SalePrice'].describe()


# ## Preprocessing

# ### Pearson Correlation Matrix of Features
# 
# Heatmap of correlation is the best way to overview the data features.

# In[ ]:


corr_matrix = train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap=plt.cm.RdBu_r)


# There are 2 highlighted red block get to my first sight. (Red means high correlation here)
# 
# - TotalBsmtSF and 1stFlrSF: the square feet of basement(TotalBsmtSF) area is probably very similiar to the 1st floor(1stFlrSF)
# - GarageCars and GarageArea: the number of cars(GarageCars) in Garage is decided by its size(GarageArea)
# 
# I think we don't care about the most negative correlated features in the highlighted blue blocks here. In current topic, the target is **Linear Regression**, pick out the most significantly related variables is more important.

# ### Correlation Matrix of 'SalePrice'
# 
# Reduce the Scope of Heatmap

# In[ ]:


k = 10
cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
k_corr_matrix = train[cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(k_corr_matrix, annot=True, cmap=plt.cm.RdBu_r)


# The most correlated features with `SalePrice`:
# 
# - OverallQual: **Rates** the overall material and finish of the house
# - GrLivArea: Above grade (ground) living **area square feet**
# - GarageCars and GarageArea: the most strongly correlated features, twin brothers of **Garage**
# - TotalBsmtSf and 1stFlrSF: **square feet** of basement and 1st floor
# - FullBath: Full **bathrooms** above grade, for urgency of ...
# - TotRmsAbvGrd: **Total rooms** above grade (does not include bathrooms)
# - YearBuilt: Original construction date decides the **years** of house
# 
# Rates, area square feet, Garage, bathrooms, total rooms, years, etc. These features is highly related to house price when taken into our real life, the correlation heatmap is really make sence.

# ### Pairplot
# 
# Another wonderful tool in @seaborn

# In[ ]:


cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], height=2)


# What we can find from the `Pairplot` figure above about variables relationship?
# 
# 1. `TotalBsmtSF` and `GrLivArea`: dots drawing a linear line like a upperbound, means that most of the basement area is not bigger than above ground living area, this make sense. It is a house, not a bunker
# 2. `GrLivArea ~ SalePrice` and `TotalBsmtSF ~ SalePrice` is linear related, those 2 continuous variable is both about **area square feet** and they are highly related to house `SalePrice`
# 3. `YearBuilt ~ SalePrice` dot clouds appears to be like an exponential function
# 
# And the categorical varibles `OverallQual`, `GarageCars`, `FullBath` is also positive correlated to `SalePrice`.

# In[ ]:


plt.figure(figsize=(16, 30))
for idx, f in enumerate(['OverallQual', 'GrLivArea', 'GarageArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF', 
                         'FullBath', 'TotRmsAbvGrd','YearBuilt']):
    plt.subplot(9, 2, 2*idx+1)
    sns.distplot(train[f])
    plt.subplot(9, 2, 2*idx+2)
    sns.scatterplot(x=f, y='SalePrice', data=train)


# In[ ]:


plt.figure(figsize=(8, 12))
train.corr()['SalePrice'].sort_values().plot(kind='barh')


# ### SalePrice
# 
# Analyze the target variable first. Does it a normal distribution?

# In[ ]:


# distribution histogram for ourt target: SalePrice
sns.distplot(train['SalePrice'], fit=norm)

(mu, sigma) = norm.fit(train['SalePrice'])
plt.legend(['Normal dist ($\mu=${:.2f}, $\sigma=${:.2f})'.format(mu, sigma)])
plt.ylabel('Frequency')
plt.title('SalePrice Distribution')

# normal probability plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# In[ ]:


#skewness and kurtosis
print('mu: %.2f, sigma: %.2f' % (mu, sigma))
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


# The `SalePrice` is not rightly normal. Shows deflecting to left, positive "skewness", and not follow the diagonal line.
# 
# **Statistics: in case of positive skewness, log transformations works well**

# In[ ]:


# applying log transformation
train['SalePriceLog'] = np.log1p(train['SalePrice'])


# In[ ]:


# distribution histogram and normal probability plot
(mu, sigma) = norm.fit(train['SalePriceLog'])

sns.distplot(train['SalePriceLog'], fit=norm)
plt.legend(['Normal dist ($\mu=${:.2f}, $\sigma=${:.2f})'.format(mu, sigma)])

fig = plt.figure()
stats.probplot(train['SalePriceLog'], plot=plt)
plt.show()


# In[ ]:


#skewness and kurtosis
print('mu: %.2f, sigma: %.2f' % (mu, sigma))
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


# The left skew is corrected and the data appears more normally distributed.

# ## Outliers

# ### SalePrice Distribution
# 
# Standardize the data and see if there're any outlier points. 
# 
# Standardization means converting data values to be with mean of 0 and standard deviation of 1 ($x \sim \mathcal{N}(0, 1)$). 
# 
# $$x=\frac{x-\mu}{\sigma}$$

# In[ ]:


sale_price_scaled = StandardScaler().fit_transform(train['SalePrice'][:, np.newaxis])

sns.distplot(sale_price_scaled, fit=norm)

low_range = sale_price_scaled[sale_price_scaled[:, 0].argsort()[:10]]
high_range = sale_price_scaled[sale_price_scaled[:, 0].argsort()[-10:]]
print(f'outer range (low) of the distribution: \n{low_range}')
print(f'outer range (high) of the distribution: \n{high_range}')


# - Low range is within 2 standard deviations ($-2\sigma$)
# - High range like the 7.x values are really out of range
# 
# At least, the 2 points with value greater than 7 should be considered as an outlier.

# ### Scatter Plot

# In[ ]:


train.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000), xlim=(0, 6000))


# The 2 points in the bottom right are outside of the crowd and definetly outliers.
# 
# The 2 points in the upper right greater than 4000 in x-axis could also be considered as outliers.

# In[ ]:


train = train[train['GrLivArea'] < 4000]


# In[ ]:


train.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000), xlim=(0, 6000))


# ## Feature Engineering

# ### Quantitative and Qualitative

# In[ ]:


quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
quantitative.sort()
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']
qualitative.sort()


# In[ ]:


# continuous variable
quantitative


# In[ ]:


# categorical variable
qualitative


# ### Concat all data
# 
# Concatenate the train and test data to analyze

# In[ ]:


train.reset_index(drop=True, inplace=True)
y_train = train['SalePriceLog']
X_train = train.drop(['SalePrice', 'SalePriceLog'], axis=1)
X_test = test


# In[ ]:


all_data = pd.concat([X_train, test], axis=0, sort=False)
all_data.drop(['Id'], axis=1, inplace=True)
all_data.shape


# In[ ]:


all_data.head(5)


# ### Missing Data

# In[ ]:


na_total = all_data.isnull().sum().sort_values(ascending=False)
na_ratio = (all_data.isnull().sum() / all_data.shape[0]).sort_values(ascending=False)
missing_data = pd.concat([na_total, na_ratio], axis=1, keys=['Total', 'Ratio'])
missing_data.head(50)


# In[ ]:


# Most value of these 4 features are missing and they have no pattern , just delete them
plt.figure(figsize=(16, 12))
for idx, f in enumerate(['PoolQC', 'Utilities', 'Street', 'MiscFeature']):
    plt.subplot(2, 2, idx+1)
    sns.scatterplot(x='SalePrice', y=f, data=train)

all_data.drop(['PoolQC', 'Utilities', 'Street', 'MiscFeature', ], axis=1, inplace=True)


# According to the `data_description`, value NA means "None" for these `categorical features`, Fiil NA with **None** for them

# In[ ]:


all_data['Alley'].fillna('None', inplace=True)
all_data['Fence'].fillna('None', inplace=True)
all_data['FireplaceQu'].fillna('None', inplace=True)

all_data['GarageQual'].fillna('None', inplace=True)
all_data['GarageFinish'].fillna('None', inplace=True)
all_data['GarageCond'].fillna('None', inplace=True)
all_data['GarageType'].fillna('None', inplace=True)

all_data['BsmtExposure'].fillna('None', inplace=True)
all_data['BsmtCond'].fillna('None', inplace=True)
all_data['BsmtQual'].fillna('None', inplace=True)
all_data['BsmtFinType2'].fillna('None', inplace=True)
all_data['BsmtFinType1'].fillna('None', inplace=True)


# For the `categorical features` without what NA means, fill the NA with the **mode**, 
# which means most categorical type of the feature

# In[ ]:


all_data['MasVnrType'].fillna('None', inplace=True)
all_data['HasMasVnr'] = all_data['MasVnrType'].apply(lambda x: 0 if x == 'None' else 1)

all_data['MSZoning'] = all_data.groupby(['MSSubClass'])['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
all_data['Functional'].fillna('Typ', inplace=True)
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])


# In[ ]:


all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


all_data['GarageYrBlt'] = (all_data['YearBuilt'] + all_data['YearRemodAdd']) /2


# In[ ]:


sns.scatterplot(x='SalePrice', y='MasVnrArea',hue='MasVnrType', data=train, legend=None)
all_data['MasVnrArea'] = all_data.groupby(['MasVnrType'])['MasVnrArea'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


print(all_data[all_data['GarageCars'].isnull()][['GarageArea', 'GarageCars', 'GarageType', 'GarageYrBlt', 'GarageQual']])
all_data['GarageArea'].fillna(0, inplace=True)
all_data['GarageCars'].fillna(0, inplace=True)


# In[ ]:


print(all_data[all_data['TotalBsmtSF'].isnull()][
    ['TotalBsmtSF', 'BsmtQual', 'BsmtCond', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFullBath','BsmtHalfBath']])
all_data['TotalBsmtSF'].fillna(0, inplace=True)
all_data['BsmtUnfSF'].fillna(0, inplace=True)
all_data['BsmtFinSF1'].fillna(0, inplace=True)
all_data['BsmtFinSF2'].fillna(0, inplace=True)
all_data['BsmtFullBath'].fillna(0, inplace=True)
all_data['BsmtHalfBath'].fillna(0, inplace=True)


# In[ ]:


# sns scatter plot might be very usefull to see the data distribution with different categories
# sns.scatterplot(x='SalePrice', y='MasVnrArea',hue='MasVnrType', data=train, legend=None)


# Merge mutiple related or same kind of categorical features to creat a new one

# In[ ]:


all_data['YrBltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']
all_data['TotalSF']=all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

all_data['TotalSqrFootage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] +
                                 all_data['1stFlrSF'] + all_data['2ndFlrSF'])

all_data['TotalBathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))

all_data['TotalPorchSF'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +
                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] +
                              all_data['WoodDeckSF'])


# Simplified features

# In[ ]:


all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


# Some of the non-numeric predictors are stored as numbers; we convert them into strings 
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# In[ ]:


all_data = pd.get_dummies(all_data).reset_index(drop=True)


# In[ ]:


all_data.isnull().sum().sort_values(ascending=False)


# ## Models

# In[ ]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import ElasticNet, Lasso, Ridge, ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from mlxtend.regressor import StackingCVRegressor


# In[ ]:


X_train = all_data.iloc[:len(y_train), :]
X_test = all_data.iloc[len(y_train):, :]


# In[ ]:


X_train.shape, y_train.shape, X_test.shape


# ### Define a cross validation strategy
# 
# Use `cross_val_score` to get the **root mean square error**, which is the score method for current regression problem
# 

# In[ ]:


def rmse_cv(model):
    mse = cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv=5)
    rmse = np.sqrt(-mse)
    print(f'{model.__class__.__name__} score: {rmse.mean():.4f}, {rmse.std():.4f}')
    #return(rmse)


# ### GridSearchCV to tune Model parameters
# 
# Use `GridSearchCV` to find the best parameters

# > #### Lasso

# In[ ]:


lasso = Lasso()
lasso_search = GridSearchCV(lasso, {'alpha': np.logspace(-4, -3, 5)}, cv=5, scoring="neg_mean_squared_error")
lasso_search.fit(X_train, y_train)
lasso_search.best_estimator_


# In[ ]:


lasso_model = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
rmse_cv(lasso_model)


# > #### Ridge

# In[ ]:


ridge = Ridge()
ridge_search = GridSearchCV(ridge, {'alpha': np.linspace(10, 30, 10)}, cv=5, scoring="neg_mean_squared_error")
ridge_search.fit(X_train, y_train)
ridge_search.best_estimator_


# In[ ]:


ridge_model = make_pipeline(RobustScaler(), Ridge(alpha=19))
rmse_cv(ridge_model)


# > #### ElasticNet, hybrid of Lasso and Ridge

# In[ ]:


enet = ElasticNet()
enet_search = GridSearchCV(enet, {'alpha': np.linspace(0.0001, 0.001, 10), 'l1_ratio':np.linspace(0.5, 1.5, 10)}, cv=5, scoring="neg_mean_squared_error")
enet_search.fit(X_train, y_train)
enet_search.best_estimator_


# In[ ]:


enet_model = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0004, l1_ratio=1.4, random_state=3))
rmse_cv(enet_model)


# > #### Gradient Boost Regressor

# In[ ]:


gbdt_model = GradientBoostingRegressor(learning_rate=0.05, min_samples_leaf=5, min_samples_split=10, max_depth=4, n_estimators=3000)
rmse_cv(gbdt_model)


# > #### Random Forest

# In[ ]:


rf_model = RandomForestRegressor(min_samples_leaf=4, min_samples_split=8)
rmse_cv(rf_model)


# > #### SVR

# In[ ]:


svr_model = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.005, gamma=0.0003))
rmse_cv(svr_model)


# > #### XGBoost

# In[ ]:


xgb_model = XGBRegressor(learning_rate=0.01, max_depth=5, n_estimators=3000, 
                         n_thread=-1, n_jobs=-1, objective='reg:squarederror')
rmse_cv(xgb_model)


# > #### Light GBM

# In[ ]:


lgb_model = LGBMRegressor(objective='regression',
                    learning_rate=0.01, max_depth=5, num_leaves=4, 
                    n_estimators=3000)
rmse_cv(lgb_model)


# > #### Stacked Models

# In[ ]:


stack_model = StackingCVRegressor([lasso_model, ridge_model, enet_model, gbdt_model, rf_model, svr_model, xgb_model, lgb_model], 
                                  meta_regressor=lgb_model,
                                  use_features_in_secondary=True)
# rmse_cv(stack_model)


# ## Prediction

# In[ ]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[ ]:


lasso_model = lasso_model.fit(X_train, y_train)
ridge_model = ridge_model.fit(X_train, y_train)
enet_model = enet_model.fit(X_train, y_train)
gbdt_model = gbdt_model.fit(X_train, y_train)
rf_model = rf_model.fit(X_train, y_train)
svr_model = svr_model.fit(X_train, y_train)
xgb_model = xgb_model.fit(X_train, y_train)
lgb_model = lgb_model.fit(X_train, y_train)
stack_model = stack_model.fit(np.array(X_train), np.array(y_train))


# In[ ]:


def combine_models_predict(X):
    return ((0.1 * enet_model.predict(X)) +             (0.1 * ridge_model.predict(X)) +             (0.1 * lasso_model.predict(X)) +             (0.15 * gbdt_model.predict(X)) +             (0.15 * xgb_model.predict(X)) +             (0.1 * lgb_model.predict(X)) +             (0.075 * rf_model.predict(X)) +             (0.075 * svr_model.predict(X)) +             (0.15 * stack_model.predict(np.array(X)))
           )


# In[ ]:


print('RMSLE score on train data:')
print(rmsle(y_train, combine_models_predict(X_train)))


# In[ ]:


log_result = combine_models_predict(X_test)
result = np.expm1(log_result)


# ## Submission

# In[ ]:


sub = pd.DataFrame()
sub['Id'] = test['Id']
sub['SalePrice'] = result
sub.head()


# In[ ]:


sub.to_csv('submission.csv',index=False)

