# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

# RMSLE function
def rmsle(y, pred):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(pred), 2)))

IDcol = 'ID'
target = 'target'

train_y = np.log1p(train["target"].values)
train_X = train.drop(['ID', 'target'], axis=1)
test_X = test.drop(['ID'], axis=1)



MAX_FEATURES = 100   
NGRAMS = 2           
MAXDEPTH = 20 

# Partition datasets into train + validation
X_train, X_test, y_train, y_test = train_test_split(
                                     train_X, train_y,
                                     test_size=0.70,
                                     random_state=42
                                     )

# Lightgbm model

params = {
    'learning_rate': 0.25,
    'application': 'regression',
    'is_enable_sparse' : 'true',
    'max_bin' : 100,
    'max_depth': 5,
    'num_leaves': 60,
    'verbosity': -1,
    'bagging_fraction': 0.5,
    'nthread': 4,
    'metric': 'RMSE'
}

d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label = y_test)
watchlist = [d_train, d_test]

model_lgb = lgb.train(params,
                  train_set=d_train,
                  num_boost_round=240,
                  valid_sets=watchlist,
                  early_stopping_rounds=20,
                  verbose_eval=10)

lgb_pred = model_lgb.predict(X_test)
LGB_RMSLE = rmsle(y_test, lgb_pred)
print('Root Mean Squared Logarithmic Error for L G B: {} '.format(LGB_RMSLE))
#XGB Model

watchlist = [(xgb.DMatrix(X_train, y_train), 'train'), (xgb.DMatrix(X_test, y_test), 'valid')]

params = {'objective': 'reg:linear', 
              'eval_metric': 'rmse', 
              'eta': 0.005, 
              'max_depth': 10, 
              'subsample': 0.7, 
              'colsample_bytree': 0.5, 
              'alpha':0, 'silent': True
              }

xgb_model = xgb.train(params, 
                      xgb.DMatrix(X_train, y_train), 
                      5000,  
                      watchlist, 
                      maximize=False, 
                      verbose_eval=200, 
                      early_stopping_rounds=100
                      )

xgb_pred = xgb_model.predict(xgb.DMatrix(X_test))
XGB_RMSLE = rmsle(y_test, xgb_pred)
print('Root Mean Squared Logarithmic Error for X G B: {} '.format(XGB_RMSLE))

# Using the best RMSLE error algorithm for prediction
submission = pd.read_csv("../input/sample_submission.csv", header = 0)
submission['target'] = 0.0
submission['target'] += np.expm1(xgb_model.predict(xgb.DMatrix(test_X), ntree_limit=xgb_model.best_ntree_limit))
submission[['ID', 'target']].to_csv('submit_xgb.csv', index=False)

# Any results you write to the current directory are saved as output.