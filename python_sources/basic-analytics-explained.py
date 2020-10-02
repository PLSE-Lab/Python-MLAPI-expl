#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Describe Train Dataset

# In[ ]:


train=pd.read_csv("../input/train.csv")
print(train.head())
test=pd.read_csv("../input/test.csv")
print(test.head())


# In[ ]:


print(train.shape)
print(test.shape)


# Assinging Ids to new series for future reference

# In[ ]:


train_id= train['Id']
test_id=test['Id']   


# Removing Ids from test and train
# 

# In[ ]:


del train['Id']
del test['Id']


# In[ ]:


print(train.shape)
print(test.shape)


# Removing outliers
# 

# In[ ]:


train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
train.shape


# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
print(ntrain)
print(ntest)


# Here the target variable is SalePrice. Storing target variable to y_train
# 

# In[ ]:


y_train = np.log1p(train.SalePrice.values)
print(y_train)


# In[ ]:


print(len(y_train))
print(len(train))


# In[ ]:


df = pd.concat((train, test),sort=False).reset_index(drop=True)
df.drop(['SalePrice'], axis=1, inplace=True)

print(df.shape)


# checking missing values
# 

# In[ ]:


train_na=df.isnull().sum(axis = 0).sort_values(ascending=False)
train_na=train_na[train_na!=0]
print(train_na)


# Take columns of string and numeric datatypes to lists

# In[ ]:


string_col=list(df.dtypes[df.dtypes==object].index)
print(string_col)


# In[ ]:


numeric_col=list(set(list(df.columns))-set(string_col))
print(numeric_col)


# Impute missing values with mode for above columns

# In[ ]:


for col in string_col:
    df[col] = df[col].fillna(str(df[col].mode()[0]))
for col in numeric_col:
    df[col] = df[col].fillna(int(df[col].median()))


# checking missing values again after imputation

# In[ ]:


print(df.isnull().sum(axis = 0).sort_values(ascending=False))


# No missing values in the dataset
# 

# Transforming some numerical columns that are categorical to string type
# 

# In[ ]:


df['OverallCond'] = df['OverallCond'].astype(str)
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
df['MSSubClass'] = df['MSSubClass'].apply(str)


# In[ ]:


#Label encoding the categorical variables


# In[ ]:


print(string_col)


# In[ ]:


print(len(string_col))
string_col.append('OverallCond')
string_col.append('YrSold')
string_col.append('MoSold')
string_col.append('MSSubClass')
print(len(string_col))


# In[ ]:


from sklearn.preprocessing import LabelEncoder
for c in string_col:
    lbl = LabelEncoder() 
    lbl.fit(list(df[c].values)) 
    df[c] = lbl.transform(list(df[c].values))
df


# In[ ]:


df.shape


# Getting dummy variables

# In[ ]:


train = pd.get_dummies(df)
print(df.shape)


# In[ ]:


# Check the skew of all numerical features
from scipy.stats import norm, skew #for some statistics
skewed_feats = df[numeric_col].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness


# In[ ]:


skewness = skewness[abs(skewness) > 0.75]
print(skewness)


# In[ ]:


from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    df[feat] = boxcox1p(df[feat], lam)


# In[ ]:


df


# In[ ]:


df.shape


# In[ ]:


train = df[:ntrain]
test = df[ntrain:]


# In[ ]:


train.shape


# In[ ]:


test.shape


# **Modelling**

# In[ ]:


from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.pipeline import make_pipeline
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


# In[ ]:


from sklearn.model_selection import KFold, cross_val_score, train_test_split
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# Gradient Boosting
# 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
model_gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# XGBoost

# In[ ]:


import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# In[ ]:


import lightgbm as lgb
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# Scoring for Lasso
# 

# In[ ]:


print(rmsle_cv(lasso).mean())


# Scoring for GradientBoost
# 

# In[ ]:


print(rmsle_cv(model_gboost).mean())


# Scoring for XGBoost

# In[ ]:


print(rmsle_cv(model_xgb).mean())


# Scoring for LightGM

# In[ ]:


print(rmsle_cv(model_lgb).mean())


# **Stacking**
# 

# Average based stacking

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


# In[ ]:


averaged_models = AveragingModels(models = (lasso,model_gboost,model_xgb,model_lgb))
score = rmsle_cv(averaged_models)


# In[ ]:


score.mean()


# Using Metamodel using oof
# 

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


# Taking lgm as base model

# In[ ]:


stacked_averaged_models = StackingAveragedModels(base_models = (model_lgb,model_gboost,model_xgb),
                                                 meta_model = lasso)


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# Training and predicting with Stacked Averaged models

# In[ ]:


stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))


# Training with gradient boosting

# In[ ]:


model_gboost.fit(train, y_train)
gb_train_pred = model_gboost.predict(train)
gb_pred = np.expm1(model_gboost.predict(test.values))
print(rmsle(y_train, gb_train_pred))


# Training with XGBoost

# In[ ]:



model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))


# Training with LightGM

# In[ ]:


model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))


# In[ ]:


print(rmsle(y_train,xgb_train_pred ))


# In[ ]:


ensemble = stacked_pred*0.99+ gb_pred*0.01
ensemble


# Submission
# 

# In[ ]:


sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)


# In[ ]:




