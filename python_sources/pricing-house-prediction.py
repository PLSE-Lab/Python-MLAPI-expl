#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Import Datasets

# In[ ]:


train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print(train_df.info())
print('-*'*20)
print(test_df.info())


# In[ ]:


sample = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
sample.head()


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


train_df['SalePrice'].describe()


# # Data Visualisation

# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt
#histogram
sns.distplot(train_df['SalePrice']);


# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % train_df['SalePrice'].skew())
print("Kurtosis: %f" % train_df['SalePrice'].kurt())

# If the kurtosis is greater than 3, then the dataset has heavier tails than a normal distribution (more in the tails).


# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
sns.regplot(x=var, y='SalePrice', data = data)


# It shows that a linear relationship exists between GrLivArea and SalePrice

# In[ ]:


#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
sns.regplot(x=var, y='SalePrice', data=data);


# It shows that a linear relationship exists between TotalBsmtSF and SalePrice

# In[ ]:


#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# It shows that with increasing OverallQual, SalePrice also increases.

# In[ ]:


var = 'YearBuilt'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# It can be seen that SalePrice has increased with year.

# **Note:**
# * 'GrLivArea' and 'TotalBsmtSF' seem to be linearly related with 'SalePrice'. Both relationships are positive, which means that as one variable increases, the other also increases. In the case of 'TotalBsmtSF', we can see that the slope of the linear relationship is particularly high.
# * 'OverallQual' and 'YearBuilt' also seem to be related with 'SalePrice'. The relationship seems to be stronger in the case of 'OverallQual', where the box plot shows how sales prices increase with the overall quality.

# **Now we shall do the following:**
# 
# * Correlation matrix (heatmap style).
# * 'SalePrice' correlation matrix (zoomed heatmap style).
# * Scatter plots between the most correlated variables (move like Jagger style).

# In[ ]:


#correlation matrix
corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[ ]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# * 'OverallQual', 'GrLivArea' and 'TotalBsmtSF' are strongly correlated with 'SalePrice'.
# * 'GarageCars' and 'GarageArea' are also some of the most strongly correlated variables.in brothers. Therefore, we just need one of these variables in our analysis (we can keep 'GarageCars' since its correlation with 'SalePrice' is higher).
# * 'TotalBsmtSF' and '1stFloor' also seem to be twin brothers. We can keep 'TotalBsmtSF' just to say that our first guess was right 
# * 'FullBath' can be ignored.
# * 'TotRmsAbvGrd' too has very less correlation and can be ignored.
# * 'YearBuilt' is slightly correlated with 'SalePrice'.

# In[ ]:


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df[cols], size = 2.5)
plt.show();


# The Total Missing Data and their Percentage is shown below

# In[ ]:


#missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# The missing data shall be removed now.

# In[ ]:


#dealing with missing data
train_df = train_df.drop((missing_data[missing_data['Total'] > 1]).index,axis =1) #all the variables except Electrical are dropped
train_df = train_df.drop(train_df.loc[train_df['Electrical'].isnull()].index) #droppinf the row containing 1 null value in Electrical variable
train_df.isnull().sum().max() 


# **Univariate analysis**

# In[ ]:


#standardizing data to have mean 0 and standard deviation 1
from sklearn.preprocessing import StandardScaler
saleprice_scaled = StandardScaler().fit_transform(train_df['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# **Normality**

# In[ ]:


from scipy.stats import norm
from scipy import stats
def diagnostic_plots(train_df, variable):## defining a function to plot histogram and Q-Q plot
    plt.figure(figsize = (15,6))
    plt.subplot(1,2,1)
    sns.distplot(train_df[variable], fit=norm);
    plt.subplot(1,2,2)
    stats.probplot(train_df[variable], dist = 'norm', plot = plt)
    plt.show()


# In[ ]:


#histogram and normal probability plot of  SalePrice
diagnostic_plots(train_df, 'SalePrice')


# The graph shown above is not a normally distributed one. Apply Logarithmic Transformation to convert it into a normal distribution

# In[ ]:


#applying log transformation
train_df['SalePrice'] = np.log(train_df['SalePrice'] + 1)# +1 is added in case there is any 0 input to it which would create issue in taking log
diagnostic_plots(train_df, 'SalePrice')


# We shall check the same for GrLivArea and TotalBsmtSF and apply the same procedure if these are not normally distributed.

# In[ ]:


#histogram and normal probability plot of  GrLivArea
diagnostic_plots(train_df, 'GrLivArea')


# Not normally distributed, so we shall apply logarithmic transformation.

# In[ ]:


#applying log transformation
train_df['GrLivArea'] = np.log(train_df['GrLivArea'] + 1)# +1 is added in case there is any 0 input to it which would create issue in taking log
diagnostic_plots(train_df, 'GrLivArea')


# Since GrLivArea is now normally distributed, we shall look into TotalBsmtSF

# In[ ]:


#histogram and normal probability plot of  TotalBsmtSF
diagnostic_plots(train_df, 'TotalBsmtSF')


# In[ ]:


#applying reciprocal transformation
train_df['GrLivArea'] = 1/ (train_df['GrLivArea']+1)
diagnostic_plots(train_df, 'GrLivArea')


# In[ ]:


train_df.head()


# In[ ]:


train_df.columns


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe()


# In[ ]:


#correlation matrix
corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(50, 20))
sns.heatmap(corrmat, vmax=.8, square=True, annot = True);


# The numerical variables having correlation with SalePrice < | 0.2 | are dropped

# In[ ]:


train_df.drop(['MSSubClass', 'OverallCond', 'BsmtFinSF2', 'LowQualFinSF',                'BsmtFullBath', 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'Id'], axis = 1, inplace = True)
train_df.info()


# # Convert Categorical Variables into ordinal numerics

# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ( 'ExterQual', 'ExterCond','HeatingQC','KitchenQual', 
       'Functional',   'LandSlope',
        'LotShape', 'PavedDrive', 'Street','CentralAir')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train_df[c].values)) 
    train_df[c] = lbl.transform(list(train_df[c].values))

# shape        
print('Shape all_data: {}'.format(train_df.shape))


# In[ ]:


# Adding total sqfootage feature 
train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']


# In[ ]:


train_df = pd.get_dummies(train_df)
train_df.head()


# # MODELLING

# In[ ]:


y= train_df['SalePrice']
X = train_df.drop('SalePrice',axis =1)


# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[ ]:


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# LASSO Regression

# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
lasso.fit(X,y)
preds = np.expm1(lasso.predict(X))
lasso_score = rmse_cv(lasso)
print("\nLasso score: {:.4f} \n".format(lasso_score.mean()))


# Elastic Net Regression

# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
ENet.fit(X,y)
preds = np.expm1(ENet.predict(X))
ENet_score = rmse_cv(ENet)
print("\nENet score: {:.4f} \n".format(ENet_score.mean()))


# Kernel Ridge Regression

# In[ ]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
KRR.fit(X,y)
preds = np.expm1(KRR.predict(X))
KRR_score = rmse_cv(KRR)
print("\nKRR score: {:.4f} \n".format(KRR_score.mean()))


# Gradient Boosting Regression

# In[ ]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
GBoost.fit(X,y)
preds = np.expm1(GBoost.predict(X))
GBoost_score = rmse_cv(GBoost)
print("\nGBoost score: {:.4f} \n".format(GBoost_score.mean()))


# XGBoost

# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(X,y)
preds = np.expm1(model_xgb.predict(X))
XGBoost_score = rmse_cv(model_xgb)
print("\XGBoost score: {:.4f} \n".format(XGBoost_score.mean()))


# LightGBM

# In[ ]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(X,y)
preds = np.expm1(model_lgb.predict(X))
lgb_score = rmse_cv(model_lgb)
print("\LightGBM score: {:.4f} \n".format(lgb_score.mean()))


# In[ ]:


solution = pd.DataFrame({"id":test_df.Id, "SalePrice":preds})
solution.to_csv("Submission.csv", index = False)

solution.head(20)


# In[ ]:


sample.head()


# In[ ]:




