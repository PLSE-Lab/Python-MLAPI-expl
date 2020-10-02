# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype
from scipy import stats
from scipy.stats import norm, skew 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

test = pd.read_csv('../input/test.csv', header=0)
train = pd.read_csv('../input/train.csv', header=0)

# 1st find some correlation between data

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

plt.figure()
sns.jointplot(x='GrLivArea', y='SalePrice', data=train, kind="hex")
sns.plt.title('GrLivArea vs SalePrice')
plt.xticks(rotation=30)
plt.savefig("1_seaborn_jointplot.png")

plt.figure()
sns.jointplot(x='YearBuilt' , y='SalePrice', data=train, kind="hex")
sns.plt.title('YearBuilt vs SalePrice')
plt.xticks(rotation=30)
plt.savefig("2_seaborn_jointplot.png")

plt.figure()
sns.barplot(x='OverallQual' , y='SalePrice', data=train) 
sns.plt.title('OverallQual vs SalePrice')
plt.savefig("3_seaborn_barplot.png")

plt.figure()
sns.barplot(x='Neighborhood' , y='SalePrice', data=train) 
sns.plt.title('Neighborhood vs SalePrice')
plt.xticks(rotation=90)
plt.savefig("4_seaborn_barplot.png")

plt.figure()
sns.barplot(x='BsmtCond' , y='SalePrice', data=train) 
sns.plt.title('BsmtCond vs SalePrice')
plt.savefig("5_seaborn_barplot.png")

train["SalePrice"] = np.log1p(train["SalePrice"])

train_2 = train.loc[:, 'MSSubClass':'SaleCondition']
test_2 = test.loc[:, 'MSSubClass':'SaleCondition']
pred = train['SalePrice']
train_test = pd.concat((train_2,test_2))

train_test['TotalSF'] = train_test['TotalBsmtSF'] + train_test['1stFlrSF'] + train_test['2ndFlrSF']

train_test = train_test.reset_index()

# 2nd preprocessing

# For use LabelEncoder you need the column type as string value and without NaN values

def preprocessing(df):
    le = LabelEncoder()
    for i in df.columns:
        df[i] = df[i].fillna(0)

        if is_numeric_dtype(df[i]) == False:
            df[i] = le.fit_transform(df[i].astype('str'))
        
    print('Preprocessing OK')
    return df

print('Preprocessing begin')
train_test = preprocessing(train_test)


# 3rd create model and find best parameters
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

train_X = train_test.iloc[:len(pred),:]
train_y = pred
test_X = train_test.iloc[len(pred):,:]

def compute_score(X_train, y_train, model):
    model = model.fit(X_train, y_train)
    crossValScore = cross_val_score(model, X_train, y_train, cv=5)
    print('The cross_val_score is :'+str(crossValScore))
    print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(crossValScore),np.std(crossValScore),np.min(crossValScore), np.max(crossValScore)))
    
ridgewp = Ridge()

print('LGBM without param')
lgbwp = lgb.LGBMRegressor()
compute_score(train_X, train_y, lgbwp)

print('LGBM with param')
'''
lgbp = lgb.LGBMRegressor(objective='regression',num_leaves=5,
    learning_rate=0.05, n_estimators=720,
    max_bin = 55, bagging_fraction = 0.8,
    bagging_freq = 5, feature_fraction = 0.2319,
    feature_fraction_seed=9, bagging_seed=9,
    min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
'''
lgbp = lgb.LGBMRegressor(objective='regression', learning_rate=0.25, n_estimators=25, num_leaves=14, max_bin=15
, bagging_fraction=1, bagging_freq=3, feature_fraction=1, feature_fraction_seed=10, bagging_seed=10
, min_data_in_leaf=5, min_sum_hessian_in_leaf=5)
'''
param = {#'learning_rate': [0.01, 1, 0.25, 0.50, 0.75] 
        #, 'n_estimators':[50, 25, 10, 100]
        #, 'num_leaves':[10, 12, 11, 13, 14, 15]
        #, 'max_bin':[10, 15, 14, 13, 12, 11]
        #, 'bagging_fraction':[0.09, 1, 0.1, 0.5]
        #, 'bagging_freq':[3, 6, 1, 10, 2, 4, 5]
        #, 'feature_fraction':[0.05, 0.5, 1]
        #, 'feature_fraction_seed':[10, 15, 6, 7, 8, 9]
        #, 'bagging_seed':[10, 15, 5, 6, 7, 8, 9]
        #, 'min_data_in_leaf':[10, 5, 1, 2, 3, 4, 6, 7, 8, 9]
        #, 'min_sum_hessian_in_leaf':[10, 5, 1, 2, 3, 4, 6, 7, 8, 9]
        }

gsearch_2 = GridSearchCV(estimator = lgbp, param_grid = param, cv=5)
gsearch_2.fit(train_X, train_y)
print(gsearch_2.best_params_, gsearch_2.best_score_)
'''
compute_score(train_X, train_y, lgbp)


print('ENet without param')
enetwp = ElasticNet()
compute_score(train_X, train_y, enetwp)

print('ENet with param')
enetp = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
compute_score(train_X, train_y, enetp)

print('Lasso without param')
lassowp = Lasso()
compute_score(train_X, train_y, lassowp)

print('Lasso with param')
lassop = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
compute_score(train_X, train_y, lassop)

print('GBM without param')
gbmwp = GradientBoostingRegressor()
compute_score(train_X, train_y, gbmwp)

print('GBM with param')
gbmp = GradientBoostingRegressor(learning_rate=0.05, n_estimators=300, max_depth=3
, min_samples_leaf=3, min_samples_split=200, subsample=0.6, random_state=10)

'''
param = {'learning_rate': [0.003, 0.05, 0.04, 0.02] 
        , 'n_estimators':[100, 200, 300, 400]
        , 'min_samples_split':[100, 150, 200, 125]
        , 'min_samples_leaf':[4, 2, 5, 3]
        , 'max_depth':[3, 5, 10]
        , 'subsample':[0.3, 0.6, 1]
        , 'random_state':[10, 15, 20, 50]
        }

gsearch_2 = GridSearchCV(estimator = gbmp, param_grid = param, cv=5)
gsearch_2.fit(train_X, train_y)
print(gsearch_2.best_params_, gsearch_2.best_score_)
'''
compute_score(train_X, train_y, gbmp)

print('XGB without param')
xgbwp = xgb.XGBRegressor()
compute_score(train_X, train_y, xgbwp)

print('XGB with param')
#xgbp = xgb.XGBRegressor(learning_rate=0.1, n_estimators=500
#, max_depth= 3, gamma=0, min_child_weight= 0, subsample=0.6, colsample_bytree=0.6)
#compute_score(train_X, train_y, xgbp)

xgbp = xgb.XGBRegressor(learning_rate=0.004, max_depth=2, n_estimators=13000, min_child_weight=0, gamma=0, subsample=0.6, colsample_bytree=0.6)
'''
xgbp = xgb.XGBRegressor()
param = {'learning_rate': [0, 0.001,0.004, 0.005, 0.1] #0.1
        , 'n_estimators':[100, 500, 1000, 5000, 10000, 11000, 12000, 13000, 15000] #2000
        , 'max_depth':[1, 2, 5, 10]
        , 'min_child_weight':[0, 0.001, 0.002, 0.005]
        , 'gamma':[0, 1, 2]
        , 'subsample':[0, 0.5, 0.6]
        , 'colsample_bytree':[0, 0.5, 0.6]
        }

gsearch_1 = GridSearchCV(estimator = xgbp, param_grid = param, cv=5)
gsearch_1.fit(train_X, train_y)
print(gsearch_1.best_params_, gsearch_1.best_score_)
'''
compute_score(train_X, train_y, xgbp)

print('KR without param')
krwp = KernelRidge()
compute_score(train_X, train_y, krwp)

print('KR with param')
krp = KernelRidge(alpha=0.0005, degree=2, coef0=2.5)
compute_score(train_X, train_y, krp)

print('stacking with Lasso, ENet, GBM, KR')
stregrwp = StackingRegressor(regressors=[enetwp, gbmwp, krwp], 
                           meta_regressor=lassowp)
compute_score(train_X, train_y, stregrwp)

print('stacking with XGB, GBM, KR, Ridge')
stregrwp = StackingRegressor(regressors=[xgbwp, gbmwp, krwp], 
                           meta_regressor=ridgewp)
compute_score(train_X, train_y, stregrwp)

#Score with best param

ridge = make_pipeline(RobustScaler(), Ridge(alpha =0.0005, random_state=1))

print('stacking with XGB, GBM, GBM, KR')
stregr = StackingRegressor(regressors=[xgbp, gbmp, krp], 
                           meta_regressor=ridge)
compute_score(train_X, train_y, stregr)

print('stacking with Lasso, ENet, GBM, KR')
stregr = StackingRegressor(regressors=[enetp, gbmp, krp], 
                           meta_regressor=lassop)
compute_score(train_X, train_y, stregr)

print('Final stack')
stregr = StackingRegressor(regressors=[enetp, gbmp, krp], 
                           meta_regressor=lassop)
compute_score(train_X, train_y, stregr)


# 4th predict and output
df_output = pd.DataFrame()
df_output['Id'] = test['Id']

#stack_pred = np.expm1(stregr.predict(test_X))

xgb_pred = np.expm1(xgbp.predict(test_X))

gbm_pred = np.expm1(gbmp.predict(test_X))

lgb_pred = np.expm1(lgbp.predict(test_X))

#prediction = stack_pred * 0.70 + xgb_pred * 0.15 + gbm_pred * 0.15
prediction = xgb_pred * 0.6 + lgb_pred * 0.2 + gbm_pred * 0.2
prediction = pd.DataFrame(prediction)

df_output['SalePrice'] = prediction
df_output[['Id','SalePrice']].to_csv('output.csv',index=False)
