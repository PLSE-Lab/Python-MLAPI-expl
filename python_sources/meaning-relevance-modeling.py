#!/usr/bin/env python
# coding: utf-8

# # Meaning & Relevance + Modeling
# [Saurabh Chakrabarty](https://linkedin.com/in/iamsaurabhc) - July 2019
# 
# Inspiration taken from: [Comprehensive Data with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) by [Pedro Marcelino](https://www.kaggle.com/pmarcelino)
# 
# ----------

# 1. <b>Understand the problem</b>. We'll look at each variable and do a philosophical analysis about their meaning and importance for this problem.
# 2. <b>Basic cleaning</b>. We'll clean the dataset and handle the missing data, outliers and categorical variables.
# 3. <b>Univariable study</b>. We'll just focus on the dependent variable ('SalePrice') and try to know a little bit more about it. We'll check if our data meets the assumptions required by most multivariate techniques.
# 4. <b>Multivariate study</b>. We'll try to understand how the dependent variable and independent variables relate. We'll check if our data meets the assumptions required by most multivariate techniques.
# 5. <b>Modeling</b>. Using different models and parameters, will try to find the best possible answer.
# 
# As mentioned by [Pedro Marcelino](https://www.kaggle.com/pmarcelino), for each analysis that we perform, we will test these four assumptions :
# 
# 
# **Normality** - When we talk about normality what we mean is that the data should look like a normal distribution. This is important because several statistic tests rely on this (e.g. t-statistics). In this exercise we'll just check univariate normality for 'SalePrice' (which is a limited approach). Remember that univariate normality doesn't ensure multivariate normality (which is what we would like to have), but it helps. Another detail to take into account is that in big samples (>200 observations) normality is not such an issue. However, if we solve normality, we avoid a lot of other problems (e.g. heteroscedacity) so that's the main reason why we are doing this analysis.
# 
# **Homoscedasticity** - Homoscedasticity refers to the 'assumption that dependent variable(s) exhibit equal levels of variance across the range of predictor variable(s)' (Hair et al., 2013). Homoscedasticity is desirable because we want the error term to be the same across all values of the independent variables.
# 
# **Linearity**- The most common way to assess linearity is to examine scatter plots and search for linear patterns. If patterns are not linear, it would be worthwhile to explore data transformations. However, we'll not get into this because most of the scatter plots we've seen appear to have linear relationships.
# 
# **Absence of correlated errors** - Correlated errors, like the definition suggests, happen when one error is correlated to another. For instance, if one positive error makes a negative error systematically, it means that there's a relationship between these variables. This occurs often in time series, where some patterns are time related. We'll also not get into this. However, if you detect something, try to add a variable that can explain the effect you're getting. That's the most common solution for correlated errors.

# ## Understanding the Problem

# In[ ]:


#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="white", palette="muted", color_codes=True)
sns.despine(left=True)


# In[ ]:


#bring in the six packs
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# Before we start any of our work, let's understand how the independent variables are related to each other.
# We are looking for the features with high correlation with "SalePrice".
# 
# Extract the correlation of all columns with "SalePrice" and pick the top 20 features.

# In[ ]:


highCorrFeatures = train[train.columns[1:]].corr()['SalePrice'].sort_values(ascending=False)[0:15].to_frame().transpose().columns.tolist()
highCorrFeatures.append('Id')
train = train[[c for c in highCorrFeatures]]
test = test[[c for c in highCorrFeatures if c!='SalePrice']]
fig, axs = plt.subplots(1,1,figsize=(12,12))
sns.heatmap(train.corr(), vmax=.8, square=True, ax=axs);
plt.close(2)


# Look deep into the Heatmap, and you'll see that a low of features have high correlation between each other (search for lighter squares) which we don't want because they don't add any value to the model. Find more [here](https://newonlinecourses.science.psu.edu/stat501/node/346/)
# 
# We note the following high correlations between feature vectors : 
# * <b>GrLivArea</b> --> TotRmsAbvGrd
# * <b>TotalBsmtSF</b> --> 1stFlrSF
# * <b>GarageCars</b> --> GarageArea
# * <b>YearBuilt</b> --> YearRemodAdd, GarageYrlt
# 
# Hence we delete the right side names as they do not provide much of uniqueness for us currenty.

# In[ ]:


train.drop(['TotRmsAbvGrd','GarageArea','YearRemodAdd', 'GarageYrBlt','1stFlrSF'],axis=1,inplace=True)
test.drop(['TotRmsAbvGrd','GarageArea','YearRemodAdd', 'GarageYrBlt','1stFlrSF'],axis=1,inplace=True)

sns.heatmap(train.corr(), vmax=.8, square=True);


# So we're left with Eight features.
# 
# Awesome. Let's dig deep into these eight features and let's get to know these features more (statistical analysis)

# ## Basic cleaning

# In[ ]:


#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

total_ = test.isnull().sum().sort_values(ascending=False)
percent_ = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent, total_, percent_], axis=1, keys=['Total', 'Percent', 'Total_','Percent_'])
missing_data.head(20)


# In[ ]:


#dealing with missing data
train['MasVnrArea'].fillna(method="ffill", inplace=True)
print(train.isnull().sum().max()) #just checking that there's no missing data missing...

#dealing with missing data
test['BsmtFinSF1'].fillna(0, inplace=True)
test['GarageCars'].fillna(0, inplace=True)
test['MasVnrArea'].fillna(0, inplace=True)
test['TotalBsmtSF'].fillna(0, inplace=True)

print(test.isnull().sum().max()) #just checking that there's no missing data missing...


# ## Univariate study

# **SalePrice** : Selling Price of the House

# In[ ]:


fig, axs = plt.subplots(1,2,figsize=(16,6))
ax = sns.distplot(train['SalePrice'],fit=norm, ax=axs[0])
res = stats.probplot(train['SalePrice'], plot=plt)
plt.close(2)

#skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


# **Normality** - *SalePrice* is not normal. It shows 'peakedness', positive skewness and does not follow the diagonal line. But in case of positive skewness, log transformations usually works well.

# In[ ]:


# Normality : applying log transformation
train['SalePrice'] = np.log(train['SalePrice'])
fig, axs = plt.subplots(1,2,figsize=(16,6))
ax = sns.distplot(train['SalePrice'],fit=norm, ax=axs[0])
res = stats.probplot(train['SalePrice'], plot=plt)
plt.close(2)

# new skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


# **Normality** - Done
# 
# **Homoscedasticity** - N/A for Univariate analysis
# 
# **Linearity** - N/A for Univariate analysis
# 
# **Absence of correlated errors** - N/A for Univariate analysis

# 
# 
# # Multivariate Study
# 
# ## Relation with Categorical variables
# 
# ### **1. OverallQual** : Rates the overall material and finish of the house

# In[ ]:


fig, axs = plt.subplots(1,2,figsize=(24,6))
fig = sns.boxplot(x="OverallQual", y="SalePrice", data=train, ax=axs[0])
ax = sns.distplot(train['OverallQual'], ax=axs[1]);
plt.close(2)


# Better the quality of the house, more the price rise. Majority of the house have (above?) average quality houses (5-7 quality)

# ### 2. GarageCars : Size of garage in car capacity

# In[ ]:


fig, axs = plt.subplots(1,2,figsize=(16,6))
fig = sns.boxplot(x="GarageCars", y="SalePrice", data=train, ax=axs[0])
ax = sns.distplot(train['GarageCars'], ax=axs[1]);
plt.close(2)


# ### 3. FullBath: Full bathrooms above grade

# In[ ]:


fig, axs = plt.subplots(1,2,figsize=(16,6))
fig = sns.boxplot(x="FullBath", y="SalePrice", data=train, ax=axs[0])
ax = sns.distplot(train['FullBath'], ax=axs[1]);
plt.close(2)


# ### 4. YearBuilt: Original construction date

# In[ ]:


fig, axs = plt.subplots(1,2,figsize=(24,6))
ax1 = sns.boxplot(x="YearBuilt", y="SalePrice", data=train, ax=axs[0])
ax = sns.distplot(train['YearBuilt'], ax=axs[1],fit=norm);
plt.close(2)


# ### 5. Fireplaces: Number of fireplaces

# In[ ]:


fig, axs = plt.subplots(1,2,figsize=(16,6))
fig = sns.boxplot(x="Fireplaces", y="SalePrice", data=train, ax=axs[0])
ax = sns.distplot(train['Fireplaces'], ax=axs[1]);
plt.close(2)


# ## Relation with Numerical variables
# 
# ### 6. GrLivArea : Above grade (ground) living area square feet

# In[ ]:


fig, axs = plt.subplots(1,3,figsize=(24,6))
ax = sns.scatterplot(x="GrLivArea", y="SalePrice", data=train, ax=axs[0])
ax = sns.distplot(train['GrLivArea'], ax=axs[1],fit=norm)
res = stats.probplot(train['GrLivArea'], plot=plt)
plt.close(2)

#skewness and kurtosis
print("Skewness: %f" % train['GrLivArea'].skew())
print("Kurtosis: %f" % train['GrLivArea'].kurt())


# **Normality** - The target variable is right skewed. As (linear) models love normally distributed data , we need to transform this variable and make it more normally distributed.

# In[ ]:


#applying log transformation
train['GrLivArea'] = np.log(train['GrLivArea'])
test['GrLivArea'] = np.log(test['GrLivArea'])
fig, axs = plt.subplots(1,3,figsize=(24,6))
ax = sns.regplot(x="GrLivArea", y="SalePrice", data=train, ax=axs[0])
ax = sns.distplot(train['GrLivArea'], ax=axs[1],fit=norm)
res = stats.probplot(train['GrLivArea'], plot=plt)
plt.close(2)

#skewness and kurtosis
print("Skewness: %f" % train['GrLivArea'].skew())
print("Kurtosis: %f" % train['GrLivArea'].kurt())


# **Normality** - Done
# 
# **Homoscedasticity** - The best approach to test homoscedasticity for two metric variables is graphically. Departures from an equal dispersion are shown by such shapes as cones (small dispersion at one side of the graph, large dispersion at the opposite side) or diamonds (a large number of points at the center of the distribution).
# 
# Older versions of this scatter plot (previous to log transformations), had a conic shape. As you can see, the current scatter plot doesn't have a conic shape anymore. That's the power of normality! Just by ensuring normality in some variables, we solved the homoscedasticity problem.
# 
# **Linearity** - We can see a very well defined linear relationship between SalePrice and GrLivArea, hence linearity is proved.
# 
# **Absence of correlated errors** - Not sure. Need help

# ### 7. TotalBsmtSF : Total basement area in square feet

# In[ ]:


fig, axs = plt.subplots(1,3,figsize=(24,6))
ax = sns.scatterplot(x="TotalBsmtSF", y="SalePrice", data=train, ax=axs[0])
ax = sns.distplot(train['TotalBsmtSF'], ax=axs[1],fit=norm);
res = stats.probplot(train['TotalBsmtSF'], plot=plt)
plt.close(2)

#skewness and kurtosis
print("Skewness: %f" % train['TotalBsmtSF'].skew())
print("Kurtosis: %f" % train['TotalBsmtSF'].kurt())


# **Normality** - The target variable is right skewed. As (linear) models love normally distributed data , we need to transform this variable and make it more normally distributed.
# 
# To apply a log transformation here, we'll create a variable that can get the effect of having or not having basement (binary variable). Then, we'll do a log transformation to all the non-zero observations, ignoring those with value zero. This way we can transform data, without losing the effect of having or not basement.

# In[ ]:


#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['HasBsmt'] = 0 
train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1

test['HasBsmt'] = pd.Series(len(test['TotalBsmtSF']), index=test.index)
test['HasBsmt'] = 0 
test.loc[test['TotalBsmtSF']>0,'HasBsmt'] = 1


# In[ ]:


#transform data
train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])
test.loc[test['HasBsmt']==1,'TotalBsmtSF'] = np.log(test['TotalBsmtSF'])
fig, axs = plt.subplots(1,3,figsize=(24,6))
ax = sns.regplot(x=train[train['TotalBsmtSF']>0]['TotalBsmtSF'], y=train[train['TotalBsmtSF']>0]["SalePrice"], ax=axs[0])
ax = sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], ax=axs[1],fit=norm)
res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
plt.close(2)

#skewness and kurtosis
print("Skewness: %f" % train[train['TotalBsmtSF']>0]['TotalBsmtSF'].skew())
print("Kurtosis: %f" % train[train['TotalBsmtSF']>0]['TotalBsmtSF'].kurt())


# **Normality** - Done
# 
# **Homoscedasticity** - Older versions of this scatter plot (previous to log transformations), had a conic shape. As you can see, the current scatter plot doesn't have a conic shape anymore. That's the power of normality! Just by ensuring normality in some variables, we solved the homoscedasticity problem.
# 
# **Linearity** - We can see a very well defined linear relationship between SalePrice and GrLivArea, hence linearity is proved.
# 
# **Absence of correlated errors** - Not sure. Need help

# ### 8. MasVnrArea : Masonry veneer area in square feet

# In[ ]:


fig, axs = plt.subplots(1,3,figsize=(24,6))
ax = sns.scatterplot(x="MasVnrArea", y="SalePrice", data=train, ax=axs[0])
ax = sns.distplot(train['MasVnrArea'].fillna(method='ffill'), ax=axs[1],fit=norm);
res = stats.probplot(train['MasVnrArea'], plot=plt)
plt.close(2)

#skewness and kurtosis
print("Skewness: %f" % train['MasVnrArea'].skew())
print("Kurtosis: %f" % train['MasVnrArea'].kurt())


# **Normality** - The target variable is right skewed. As (linear) models love normally distributed data , we need to transform this variable and make it more normally distributed.
# 
# To apply a log transformation here, we'll create a variable that can get the effect of having or not having basement (binary variable). Then, we'll do a log transformation to all the non-zero observations, ignoring those with value zero. This way we can transform data, without losing the effect of having or not basement.

# In[ ]:


#create column for new variable (one is enough because it's a binary categorical feature)
#if MasVnrArea>0 it gets 1, for MasVnrArea==0 it gets 0
train['HasMasVnr'] = pd.Series(len(train['MasVnrArea']), index=train.index)
train['HasMasVnr'] = 0 
train.loc[train['MasVnrArea']>0,'HasMasVnr'] = 1

test['HasMasVnr'] = pd.Series(len(test['MasVnrArea']), index=test.index)
test['HasMasVnr'] = 0 
test.loc[test['MasVnrArea']>0,'HasMasVnr'] = 1


# In[ ]:


#transform data
train.loc[train['HasMasVnr']==1,'MasVnrArea'] = np.log(train['MasVnrArea'])
test.loc[test['HasMasVnr']==1,'MasVnrArea'] = np.log(test['MasVnrArea'])
fig, axs = plt.subplots(1,3,figsize=(24,6))
ax = sns.regplot(x=train[train['MasVnrArea']>0]['MasVnrArea'], y=train[train['MasVnrArea']>0]["SalePrice"], ax=axs[0])
ax = sns.distplot(train[train['MasVnrArea']>0]['MasVnrArea'], ax=axs[1],fit=norm)
res = stats.probplot(train[train['MasVnrArea']>0]['MasVnrArea'], plot=plt)
plt.close(2)

#skewness and kurtosis
print("Skewness: %f" % train[train['MasVnrArea']>0]['MasVnrArea'].skew())
print("Kurtosis: %f" % train[train['MasVnrArea']>0]['MasVnrArea'].kurt())


# **Normality** - Done
# 
# **Homoscedasticity** - Older versions of this scatter plot (previous to log transformations), had a conic shape. As you can see, the current scatter plot doesn't have a conic shape anymore. That's the power of normality! Just by ensuring normality in some variables, we solved the homoscedasticity problem.
# 
# **Linearity** - We can see a very well defined linear relationship between SalePrice and GrLivArea, hence linearity is proved.
# 
# **Absence of correlated errors** - Not sure. Need help

# ### 9. BsmtFinSF1 : Type 1 finished square feet

# In[ ]:


fig, axs = plt.subplots(1,3,figsize=(24,6))
ax = sns.scatterplot(x="BsmtFinSF1", y="SalePrice", data=train, ax=axs[0])
ax = sns.distplot(train['BsmtFinSF1'].fillna(method="ffill"), ax=axs[1],fit=norm);
res = stats.probplot(train['BsmtFinSF1'], plot=plt)
plt.close(2)

#skewness and kurtosis
print("Skewness: %f" % train['BsmtFinSF1'].skew())
print("Kurtosis: %f" % train['BsmtFinSF1'].kurt())


# **Normality** - The target variable is right skewed. As (linear) models love normally distributed data , we need to transform this variable and make it more normally distributed.
# 
# To apply a log transformation here, we'll create a variable that can get the effect of having or not having basement (binary variable). Then, we'll do a log transformation to all the non-zero observations, ignoring those with value zero. This way we can transform data, without losing the effect of having or not basement.

# In[ ]:


#create column for new variable (one is enough because it's a binary categorical feature)
#if BsmtFinSF1>0 it gets 1, for BsmtFinSF1==0 it gets 0
train['HasBsmtFinSF1'] = pd.Series(len(train['BsmtFinSF1']), index=train.index)
train['HasBsmtFinSF1'] = 0 
train.loc[train['BsmtFinSF1']>0,'HasBsmtFinSF1'] = 1

test['HasBsmtFinSF1'] = pd.Series(len(test['BsmtFinSF1']), index=test.index)
test['HasBsmtFinSF1'] = 0 
test.loc[test['BsmtFinSF1']>0,'HasBsmtFinSF1'] = 1


# In[ ]:


#transform data
train.loc[train['HasBsmtFinSF1']==1,'BsmtFinSF1'] = np.log(train['BsmtFinSF1'])
test.loc[test['HasBsmtFinSF1']==1,'BsmtFinSF1'] = np.log(test['BsmtFinSF1'])
fig, axs = plt.subplots(1,3,figsize=(24,6))
ax = sns.regplot(x=train[train['BsmtFinSF1']>0]['BsmtFinSF1'], y=train[train['BsmtFinSF1']>0]["SalePrice"], ax=axs[0])
ax = sns.distplot(train[train['BsmtFinSF1']>0]['BsmtFinSF1'], ax=axs[1],fit=norm)
res = stats.probplot(train[train['BsmtFinSF1']>0]['BsmtFinSF1'], plot=plt)
plt.close(2)

#skewness and kurtosis
print("Skewness: %f" % train[train['BsmtFinSF1']>0]['BsmtFinSF1'].skew())
print("Kurtosis: %f" % train[train['BsmtFinSF1']>0]['BsmtFinSF1'].kurt())


# **Normality** - Done
# 
# **Homoscedasticity** - Older versions of this scatter plot (previous to log transformations), had a conic shape. As you can see, the current scatter plot doesn't have a conic shape anymore. That's the power of normality! Just by ensuring normality in some variables, we solved the homoscedasticity problem.
# 
# **Linearity** - We can see a very well defined linear relationship between SalePrice and GrLivArea, hence linearity is proved.
# 
# **Absence of correlated errors** - Not sure. Need help

# ## Modeling

# Inpiration taken from [Stacked Regression](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard) by [Serigne](https://www.kaggle.com/serigne)

# In[ ]:


#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

y_train = train.SalePrice.values
train.drop(['SalePrice'], axis=1, inplace=True)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# This class will be used for Hyperparameter tuning of all Algorithms
class TunedModel:
    def __init__(self, X,y):
        self.X = X
        self.y = y
        self.clf = None
        self.params = None
    
    def run(self,clf, params):
        self.clf = clf
        self.params = params
        # run randomized search
        n_iter_search = 20
        tunedModel = RandomizedSearchCV(self.clf, param_distributions=self.params,n_iter=n_iter_search, cv=5, iid=False)
        tunedModel.fit(self.X, self.y)
        #self.report(tunedModel.cv_results_)
        return tunedModel
        
    # Utility function to report best scores
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
                
tunedObj = TunedModel(train,y_train)


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


# #### Define a cross validation strategy

# In[ ]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# ### Base Models

# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# In[ ]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


# In[ ]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# In[ ]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[ ]:


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# In[ ]:


stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# In[ ]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[ ]:


stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))


# In[ ]:


model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))


# In[ ]:


model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))


# In[ ]:


''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))


# In[ ]:


ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)

