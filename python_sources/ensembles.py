#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#delete 
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


train_ID=train['Id']
test_ID=test['Id']
train.drop("Id",axis=1,inplace=True)
test.drop("Id",axis=1,inplace=True)


# In[ ]:



plt.figure(figsize=(30,30))
sns.heatmap(train.corr(),annot=True)


# In[ ]:


train=train[train["GrLivArea"]<4500]
train.reset_index(drop=True,inplace=True)


# In[ ]:


train['SalePrice']=np.log1p(train['SalePrice'])
y=train['SalePrice']
train_features=train.drop('SalePrice',axis=1)
test_features=test


# In[ ]:


features=pd.concat([train_features,test_features],axis=0)
numeric_t = [f for f in features.columns if features.dtypes[f] != 'object']
char_t = [f for f in features.columns if features.dtypes[f] == 'object']


# In[ ]:


for col in numeric_t:
    if features[col].isnull().sum()>0:
        print("{} is lack of {}".format(col,features[col].isnull().sum()))


# In[ ]:


for col in char_t:
    if features[col].isnull().sum()>0:
        print("{} is lack of {}".format(col,features[col].isnull().sum()))


# In[ ]:


features['MSSubClass'] = features['MSSubClass'].astype(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)
features['Functional']=features['Functional'].fillna('Typ')
features['Electrical'] = features['Electrical'].fillna("SBrkr")
features['KitchenQual'] = features['KitchenQual'].fillna("TA")
features['Exterior1st']=features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
features['Exterior2nd']=features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
features["PoolQC"] = features["PoolQC"].fillna("None")

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    features[col] = features[col].fillna(0)
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    features[col] = features[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('None')
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

objects = []
for i in features.columns:
    if features[i].dtype == object:
        objects.append(i)
features.update(features[objects].fillna('None'))


# In[ ]:


features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics.append(i)
features.update(features[numerics].fillna(0))

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
features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']
features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                              features['WoodDeckSF'])
features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


final_features = pd.get_dummies(features).reset_index(drop=True)
print(final_features.shape)
X = final_features.iloc[:len(y), :]
X_sub = final_features.iloc[len(X):, :]
print('X', X.shape, 'y', y.shape, 'X_sub', X_sub.shape)

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


X_sub=scale.transform(X_sub)


# In[ ]:


def blend_models_predict(X):
    return ((0.1 * ela_best.predict(X)) +             (0.1 * las_best.predict(X)) +             (0.1 * rid_best.predict(X)) +             (0.1 * svr_best.predict(X)) +             (0.1 * gbdt_best.predict(X)) +             (0.1 * xgb_best.predict(X)) +             (0.1 * lgbm_best.predict(X)) +             (0.3 * stack_gen.predict(np.array(X))))
            
print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X1)))

print('Predict submission', datetime.now(),)
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))


# In[ ]:


submission.head()


# In[ ]:


sub_1 = pd.read_csv('../input/top-10-0-10943-stacking-mice-and-brutal-force/House_Prices_submit.csv')
sub_2 = pd.read_csv('../input/hybrid-svm-benchmark-approach-0-11180-lb-top-2/hybrid_solution.csv')
sub_3 = pd.read_csv('../input/lasso-model-for-regression-problem/lasso_sol22_Median.csv')
submission.iloc[:,1] = np.floor((0.25 * np.floor(np.expm1(blend_models_predict(X_sub)))) + 
                                (0.25 * sub_1.iloc[:,1]) + 
                                (0.25 * sub_2.iloc[:,1]) + 
                                (0.25 * sub_3.iloc[:,1]))

submission.to_csv("submission.csv", index=False)


# In[ ]:




