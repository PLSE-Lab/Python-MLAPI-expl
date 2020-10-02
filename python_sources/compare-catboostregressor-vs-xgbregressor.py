
import os
print(os.listdir("../input"))

import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import Pool, CatBoostRegressor
from sklearn import model_selection

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    res = np.sqrt(np.mean((y_pred - y) ** 2))
    return res


def preprocess_col(x):
    if x.dtypes == object:
        x = x.fillna('None')
        x = pd.Categorical(x).codes
    else:
        x = x.fillna(0)
        x = np.log1p(x)
    return x


train = pd.read_csv('../input/train.csv', parse_dates=True, index_col=0)
test = pd.read_csv('../input/test.csv', parse_dates=True, index_col=0)

endog = np.log1p(train['SalePrice'].values)
train = train.drop('SalePrice', axis=1)

cat_features = np.where(train.dtypes == object)[0]
print('using caterogicals: ', len(cat_features), ', and numericals: ', len(np.where(train.dtypes != object)[0]))
train = train.apply(preprocess_col, axis=0)
test = test.apply(preprocess_col, axis=0)

est_cat = CatBoostRegressor(verbose=False, random_seed=1, iterations=100, depth=3, learning_rate=0.1)#, one_hot_max_size=50)
est_xgbr = xgb.XGBRegressor(random_state=1, subsample=0.7, colsample_bytree=0.7, n_estimators=500,
                            learning_rate=0.03, max_depth=5, min_child_weight=3)

meta_features = np.zeros((2, train.index.size))
rkfold = model_selection.RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)
for train_index, hold_index in rkfold.split(endog):
    X_train, X_test = train.take(train_index, axis=0), train.take(hold_index, axis=0)
    y_train, y_test = endog.take(train_index, axis=0), endog.take(hold_index, axis=0)

    est_cat.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=cat_features)
    features_set1 = est_cat.predict(X_test)
    meta_features[0, hold_index] += features_set1

    features_set2 = est_xgbr.fit(X_train, y_train).predict(X_test)
    meta_features[1, hold_index] += features_set2

    print('catboost_flow: ', rmsle(y_test, features_set1), 'xgbr_flow: ', rmsle(y_test, features_set2))

meta_features = meta_features / 5
print('MEAN catboost_flow: ', rmsle(endog, meta_features[0]), 'xgbr_flow: ', rmsle(endog, meta_features[1]))

model_cat = est_cat.fit(train, endog, cat_features=cat_features)
model_xgbr = est_xgbr.fit(train, endog)

features_set1 = model_cat.predict(train)
features_set2 = model_xgbr.predict(train)

print('DIRECT catboost_flow: ', rmsle(endog, features_set1), 'xgbr_flow: ', rmsle(endog, features_set2))

features_set1 = model_cat.predict(test)
features_set2 = model_xgbr.predict(test)

submit = pd.DataFrame()
submit['Id'] = test.index
submit['SalePrice'] = np.expm1(features_set1)
submit.to_csv('attemp_cat.csv', index=False)
submit['SalePrice'] = np.expm1(features_set2)
submit.to_csv('attemp_xgbr.csv', index=False)
