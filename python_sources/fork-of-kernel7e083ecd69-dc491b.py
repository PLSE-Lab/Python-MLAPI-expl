#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


train.shape,test.shape


# In[ ]:


train.head(5)


# In[ ]:


train_ID=train['Id']
test_ID=test['Id']
train.drop("Id",axis=1,inplace=True)
test.drop("Id",axis=1,inplace=True)


# In[ ]:


#delete 
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(30,30))
sns.heatmap(train.corr(),annot=True)


# In[ ]:


train=train[train["GrLivArea"]<4500]
train.reset_index(drop=True,inplace=True)


# In[ ]:


train.columns


# In[ ]:


train['SalePrice']=np.log1p(train['SalePrice'])
y=train['SalePrice']
train_features=train.drop('SalePrice',axis=1)
test_features=test


# In[ ]:


train_features.shape,test_features.shape


# In[ ]:


features=pd.concat([train_features,test_features],axis=0)
features.shape


# In[ ]:


numeric_t = [f for f in features.columns if features.dtypes[f] != 'object']
char_t = [f for f in features.columns if features.dtypes[f] == 'object']
numeric_t 


# In[ ]:


char_t


# In[ ]:


features['MoSold'].value_counts()


# In[ ]:


for col in numeric_t:
    if features[col].isnull().sum()>0:
        print("{} is lack of {}".format(col,features[col].isnull().sum()))


# LotFrontage is lack of 486   q
# MasVnrArea is lack of 23
# BsmtFinSF1 is lack of 1
# BsmtFinSF2 is lack of 1
# BsmtUnfSF is lack of 1
# TotalBsmtSF is lack of 1
# BsmtFullBath is lack of 2
# BsmtHalfBath is lack of 2
# GarageYrBlt is lack of 159   q
# GarageCars is lack of 1      q 
# GarageArea is lack of 1      q

# In[ ]:


for col in char_t:
    if features[col].isnull().sum()>0:
        print("{} is lack of {}".format(col,features[col].isnull().sum()))


# MSZoning is lack of 4                q
# Alley is lack of 2719                w
# Utilities is lack of 2               w
# Exterior1st is lack of 1             w
# Exterior2nd is lack of 1             w
# MasVnrType is lack of 24             w
# BsmtQual is lack of 81               q 
# BsmtCond is lack of 82               q
# BsmtExposure is lack of 82           q
# BsmtFinType1 is lack of 79           q
# BsmtFinType2 is lack of 80           q
# Electrical is lack of 1              q
# KitchenQual is lack of 1             q
# Functional is lack of 2              q
# FireplaceQu is lack of 1420          w
# GarageType is lack of 157            q
# GarageFinish is lack of 159          q
# GarageQual is lack of 159            q
# GarageCond is lack of 159            q
# PoolQC is lack of 2908               q
# Fence is lack of 2346                w
# MiscFeature is lack of 2812          w
# SaleType is lack of 1                w

# In[ ]:


features['MSSubClass'] = features['MSSubClass'].astype(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)


# In[ ]:


features['Functional']=features['Functional'].fillna('Typ')
features['Electrical'] = features['Electrical'].fillna("SBrkr")
features['KitchenQual'] = features['KitchenQual'].fillna("TA")


# In[ ]:


features['Exterior1st']=features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
features['Exterior2nd']=features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])


# In[ ]:


features["PoolQC"] = features["PoolQC"].fillna("None")


# In[ ]:


for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    features[col] = features[col].fillna(0)
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    features[col] = features[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('None')


# In[ ]:


features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))


# In[ ]:


objects = []
for i in features.columns:
    if features[i].dtype == object:
        objects.append(i)
features.update(features[objects].fillna('None'))


# In[ ]:


features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics.append(i)
features.update(features[numerics].fillna(0))


# In[ ]:


numerics2 = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics2.append(i)


# In[ ]:


from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))


# In[ ]:


features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)


# In[ ]:


features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']
features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                              features['WoodDeckSF'])


# In[ ]:


features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


print(features.shape)


# In[ ]:


final_features = pd.get_dummies(features).reset_index(drop=True)
print(final_features.shape)


# In[ ]:


X = final_features.iloc[:len(y), :]
X_sub = final_features.iloc[len(X):, :]
print('X', X.shape, 'y', y.shape, 'X_sub', X_sub.shape)


# In[ ]:


outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])


# In[ ]:


overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.94:
        overfit.append(i)

overfit = list(overfit)
overfit.append('MSZoning_C (all)')

X = X.drop(overfit, axis=1).copy()
X_sub = X_sub.drop(overfit, axis=1).copy()


# In[ ]:


overfit


# In[ ]:


print('X', X.shape, 'y', y.shape, 'X_sub', X_sub.shape)


# In[ ]:


import numpy as np  # linear algebra
import pandas as pd  #
from datetime import datetime

from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# In[ ]:


# rmsle
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# build our model scoring function
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y,
                                    scoring="neg_mean_squared_error",
                                    cv=kfolds))
    return (rmse)


# In[ ]:


from sklearn.model_selection import GridSearchCV
kfolds=KFold(n_splits=10,shuffle=True,random_state=42)
scale=RobustScaler().fit(X)
X1=scale.transform(X)
from sklearn.linear_model import Ridge

model=Ridge()
rid_param_grid = {"alpha":[19.8]}
grid_search= GridSearchCV(model,param_grid=rid_param_grid,cv=kfolds,scoring="neg_mean_squared_error",n_jobs= 10, verbose = 1)
grid_search.fit(X1,y)
rid_best=grid_search.best_estimator_
np.sqrt(-grid_search.best_score_)


# In[ ]:


from sklearn.linear_model import Lasso
model=Lasso()
las_param_grid = {"alpha":[0.0005963623316594642]}
grid_search= GridSearchCV(model,param_grid=las_param_grid,cv=kfolds,scoring="neg_mean_squared_error",n_jobs= 10, verbose = 1)
grid_search.fit(X1,y)
las_best=grid_search.best_estimator_
np.sqrt(-grid_search.best_score_)


# In[ ]:


from sklearn.linear_model import ElasticNet
model=ElasticNet()
ela_param_grid = {"alpha":[0.0006951927961775605],
                 "l1_ratio":[0.90]}
grid_search= GridSearchCV(model,param_grid=ela_param_grid,cv=kfolds,scoring="neg_mean_squared_error",n_jobs= 10, verbose = 1)
grid_search.fit(X1,y)
ela_best=grid_search.best_estimator_
np.sqrt(-grid_search.best_score_)


# In[ ]:


from sklearn.svm import SVR
model=SVR()
svr_param_grid = {"C":[66],
                 "gamma":[6.105402296585326e-05]}
grid_search= GridSearchCV(model,param_grid=svr_param_grid,cv=kfolds,scoring="neg_mean_squared_error",n_jobs= 10, verbose = 1)
grid_search.fit(X1,y)
svr_best=grid_search.best_estimator_
np.sqrt(-grid_search.best_score_)


# In[ ]:


grid_search.best_params_


# In[ ]:


model=GradientBoostingRegressor()
gbdt_param_grid = {"n_estimators":[2200],
                 "learning_rate":[0.05],
                   "max_depth":[3],
                   "max_features":["sqrt"],
                   "min_samples_leaf":[5],
                   "min_samples_split":[12],
                   "loss":["huber"]
                  }
                   
                   
                   
grid_search= GridSearchCV(model,param_grid=gbdt_param_grid,cv=kfolds,scoring="neg_mean_squared_error",n_jobs= 10, verbose = 1)
grid_search.fit(X1,y)
gbdt_best=grid_search.best_estimator_
np.sqrt(-grid_search.best_score_)


# In[ ]:


model=LGBMRegressor()
lgbm_param_grid = {
                   'objective':['regression'], 
                   'max_depth':[5],
                   'num_leaves':[12],
                   'learning_rate':[0.005], 
                    'n_estimators':[5500],
                    'max_bin':[190], 
                    'bagging_fraction':[0.2],
                    'feature_fraction':[0.2]                  
                  }
                   
                   
                   
grid_search= GridSearchCV(model,param_grid=lgbm_param_grid,cv=kfolds,scoring="neg_mean_squared_error",n_jobs= 10, verbose = 1)
grid_search.fit(X1,y)
lgbm_best=grid_search.best_estimator_
np.sqrt(-grid_search.best_score_)


# In[ ]:


from sklearn.model_selection import GridSearchCV
kfolds=KFold(n_splits=5,shuffle=True,random_state=42)
scale=RobustScaler().fit(X)
X1=scale.transform(X)
model=XGBRegressor()
xgb_param_grid = {"n_estimators":[3000],
                 "learning_rate":[0.01],
                   "max_depth":[3],
                   "subsample":[0.8],
                "colsample_bytree":[0.8],
                 "gamma":[0],
                "objective":['reg:linear'],
                "min_child_weight":[2], 
                "reg_alpha":[0.1],
                "reg_lambda":[0.5]
                  }
                   
                   
                   
grid_search= GridSearchCV(model,param_grid=xgb_param_grid,cv=kfolds,scoring="neg_mean_squared_error",n_jobs= 10, verbose = 1)
grid_search.fit(X1,y)
xgb_best=grid_search.best_estimator_
np.sqrt(-grid_search.best_score_)


# In[ ]:


stack_gen = StackingCVRegressor(regressors=(rid_best, las_best, ela_best,
                                            gbdt_best, xgb_best, lgbm_best),
                                meta_regressor=xgb_best,
                                use_features_in_secondary=True)
stack_gen.fit(np.array(X1), np.array(y))


# In[ ]:





# In[ ]:


X_sub=scale.transform(X_sub)


# In[ ]:


def blend_models_predict(X):
    return ((0.1 * ela_best.predict(X)) +             (0.1 * las_best.predict(X)) +             (0.1 * rid_best.predict(X)) +             (0.1 * svr_best.predict(X)) +             (0.1 * gbdt_best.predict(X)) +             (0.1 * xgb_best.predict(X)) +             (0.1 * lgbm_best.predict(X)) +             (0.3 * stack_gen.predict(np.array(X))))
            
print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X1)))

print('Predict submission', datetime.now(),)
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))


# def blend_models_predict(X):
#     return ((0.4 * ela_best.predict(X)) + \
#             (0.3 * las_best.predict(X)) + \
#             (0.3 * rid_best.predict(X))) 
#             
#             
# print('RMSLE score on train data:')
# print(rmsle(y, blend_models_predict(X1)))
# 
# print('Predict submission', datetime.now(),)
# submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
# submission.iloc[:,1] = np.expm1(blend_models_predict(X_sub))
# 

# In[ ]:


submission.head()


# # setup models    
# kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
# alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
# alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
# e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
# e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
# 
# ridge = make_pipeline(RobustScaler(),
#                       RidgeCV(alphas=alphas_alt, cv=kfolds,))
# 
# lasso = make_pipeline(RobustScaler(),
#                       LassoCV(max_iter=1e7, alphas=alphas2,
#                               random_state=42, cv=kfolds))
# 
# elasticnet = make_pipeline(RobustScaler(),
#                            ElasticNetCV(max_iter=1e7, alphas=e_alphas,
#                                         cv=kfolds, random_state=42, l1_ratio=e_l1ratio))
#                                         
# svr = make_pipeline(RobustScaler(),
#                       SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
# 
# 
# gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
#                                    max_depth=4, max_features='sqrt',
#                                    min_samples_leaf=15, min_samples_split=10, 
#                                    loss='huber', random_state =42)
#                                    
# 
# lightgbm = LGBMRegressor(objective='regression', 
#                                        num_leaves=4,
#                                        learning_rate=0.01, 
#                                        n_estimators=5000,
#                                        max_bin=200, 
#                                        bagging_fraction=0.75,
#                                        bagging_freq=5, 
#                                        bagging_seed=7,
#                                        feature_fraction=0.2,
#                                        feature_fraction_seed=7,
#                                        verbose=-1,
#                                        #min_data_in_leaf=2,
#                                        #min_sum_hessian_in_leaf=11
#                                        )
#                                        
# 
# xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
#                                      max_depth=3, min_child_weight=0,
#                                      gamma=0, subsample=0.7,
#                                      colsample_bytree=0.7,
#                                      objective='reg:linear', nthread=-1,
#                                      scale_pos_weight=1, seed=27,
#                                      reg_alpha=0.00006, random_state=42)
# 
# # stack
# stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet,
#                                             gbr, xgboost, lightgbm),
#                                 meta_regressor=xgboost,
#                                 use_features_in_secondary=True)

# print('TEST score on CV')
# 
# score = cv_rmse(ridge)
# print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
# 
# score = cv_rmse(lasso)
# print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
# 
# score = cv_rmse(elasticnet)
# print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
# 
# score = cv_rmse(svr)
# print("SVR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
# 
# score = cv_rmse(lightgbm)
# print("Lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
# 
# score = cv_rmse(gbr)
# print("GradientBoosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
# 
# score = cv_rmse(xgboost)
# print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

# print('START Fit')
# print(datetime.now(), 'StackingCVRegressor')
# stack_gen_model = stack_gen.fit(np.array(X), np.array(y))
# print(datetime.now(), 'elasticnet')
# elastic_model_full_data = elasticnet.fit(X, y)
# print(datetime.now(), 'lasso')
# lasso_model_full_data = lasso.fit(X, y)
# print(datetime.now(), 'ridge')
# ridge_model_full_data = ridge.fit(X, y)
# print(datetime.now(), 'svr')
# svr_model_full_data = svr.fit(X, y)
# print(datetime.now(), 'GradientBoosting')
# gbr_model_full_data = gbr.fit(X, y)
# print(datetime.now(), 'xgboost')
# xgb_model_full_data = xgboost.fit(X, y)
# print(datetime.now(), 'lightgbm')
# lgb_model_full_data = lightgbm.fit(X, y)

# def blend_models_predict(X):
#     return ((0.1 * elastic_model_full_data.predict(X)) + \
#             (0.05 * lasso_model_full_data.predict(X)) + \
#             (0.1 * ridge_model_full_data.predict(X)) + \
#             (0.1 * svr_model_full_data.predict(X)) + \
#             (0.1 * gbr_model_full_data.predict(X)) + \
#             (0.15 * xgb_model_full_data.predict(X)) + \
#             (0.1 * lgb_model_full_data.predict(X)) + \
#             (0.3 * stack_gen_model.predict(np.array(X))))
#             
# print('RMSLE score on train data:')
# print(rmsle(y, blend_models_predict(X)))
# 
# print('Predict submission', datetime.now(),)
# submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
# submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))
# 

# print('RMSLE score on train data:')
# print(rmsle(y,las_best.predict(X1)))
# X_sub=scale.transform(X_sub)
# print('Predict submission', datetime.now(),)
# submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
# submission.iloc[:,1] = np.floor(np.expm1(las_best.predict(X_sub)))

# In[ ]:


print('Blend with Top Kernels submissions\n')
sub_1 = pd.read_csv('../input/top-10-0-10943-stacking-mice-and-brutal-force/House_Prices_submit.csv')
sub_2 = pd.read_csv('../input/hybrid-svm-benchmark-approach-0-11180-lb-top-2/hybrid_solution.csv')
sub_3 = pd.read_csv('../input/lasso-model-for-regression-problem/lasso_sol22_Median.csv')
submission.iloc[:,1] = np.floor((0.25 * np.floor(np.expm1(blend_models_predict(X_sub)))) + 
                                (0.25 * sub_1.iloc[:,1]) + 
                                (0.25 * sub_2.iloc[:,1]) + 
                                (0.25 * sub_3.iloc[:,1]))


# In[ ]:


submission.to_csv("submission.csv", index=False)


# In[ ]:


submission.head(5)

