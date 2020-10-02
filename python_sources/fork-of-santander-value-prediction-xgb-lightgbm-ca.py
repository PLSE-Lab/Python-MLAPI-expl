#!/usr/bin/env python
# coding: utf-8

# In[29]:


import xgboost as xgb
import lightgbm as lgb
from sklearn import *
import pandas as pd
import numpy as np

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
col = [c for c in train.columns if c not in ['ID', 'target']]
print(train.shape, test.shape)

scl = preprocessing.StandardScaler()
def rmsle(y, pred):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(pred), 2)))

x1, x2, y1, y2 = model_selection.train_test_split(train[col], train.target.values, test_size=0.10, random_state=5)
model = ensemble.RandomForestRegressor(n_jobs = -1, random_state = 7)
model.fit(scl.fit_transform(x1), y1)
print(rmsle(y2, model.predict(scl.transform(x2))))
col = pd.DataFrame({'importance': model.feature_importances_, 'feature': col}).sort_values(by=['importance'], ascending=[False])[:480]['feature'].values

test['target_lgb'] = 0.0
test['target_xgb'] = 0.0
folds = 5
for fold in range(folds):
    x1, x2, y1, y2 = model_selection.train_test_split(train[col], np.log1p(train.target.values), test_size=0.20, random_state=fold)
    #LightGBM
    params = {'learning_rate': 0.02,'max_depth': 13,'reg_alpha':0.04,'reg_lambda':0.073,'boosting': 'gbdt','objective': 'regression','metric': 'rmse','is_training_metric': True, 'num_leaves': 12**2,'feature_fraction': 0.9,'bagging_fraction': 0.8, 'bagging_freq': 5,'min_split_gain':0.0222415,'min_child_weight':40,'subsample':0.8715623,'seed':fold,'silent':-1,'verbose':-1}
    model = lgb.train(params, lgb.Dataset(x1, label=y1), 3000, lgb.Dataset(x2, label=y2), verbose_eval=200, early_stopping_rounds=100)
    test['target_lgb'] += np.expm1(model.predict(test[col], num_iteration=model.best_iteration))
    #XGB
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    #https://www.kaggle.com/samratp/santander-value-prediction-xgb-and-lightgbm
    params = {'objective': 'reg:linear', 'eval_metric': 'rmse', 'eta': 0.005, 'max_depth': 10, 'subsample': 0.7, 'colsample_bytree': 0.5, 'alpha':0, 'silent': True, 'random_state':fold}
    model = xgb.train(params, xgb.DMatrix(x1, y1), 5000,  watchlist, maximize=False, verbose_eval=200, early_stopping_rounds=100)
    test['target_xgb'] += np.expm1(model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit))

test['target_lgb'] /= folds
test['target_xgb'] /= folds
test['target'] = (test['target_lgb'] + test['target_xgb'])/2
test[['ID', 'target']].to_csv('submission1.csv', index=False)


# In[33]:


#https://www.kaggle.com/yekenot/baseline-with-decomposition-components

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

print("Load data...")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')
print("Train shape: {}\nTest shape: {}".format(train.shape, test.shape))

#Added Columns from feature_selection
train = train[['ID', 'target']+list(col)]
test = test[['ID']+list(col)]
print("Train shape: {}\nTest shape: {}".format(train.shape, test.shape))

PERC_TRESHOLD = 0.98   ### Percentage of zeros in each feature ###
N_COMP = 20            ### Number of decomposition components ###

target = np.log1p(train['target']).values
cols_to_drop = [col for col in train.columns[2:]
                    if [i[1] for i in list(train[col].value_counts().items()) 
                    if i[0] == 0][0] >= train.shape[0] * PERC_TRESHOLD]

print("Define training features...")
exclude_other = ['ID', 'target']
train_features = []
for c in train.columns:
    if c not in cols_to_drop     and c not in exclude_other:
        train_features.append(c)
print("Number of featuress for training: %s" % len(train_features))

train, test = train[train_features], test[train_features]
print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))

print("\nStart decomposition process...")
print("PCA")
pca = PCA(n_components=N_COMP, random_state=17)
pca_results_train = pca.fit_transform(train)
pca_results_test = pca.transform(test)

print("tSVD")
tsvd = TruncatedSVD(n_components=N_COMP, random_state=17)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

print("ICA")
ica = FastICA(n_components=N_COMP, random_state=17)
ica_results_train = ica.fit_transform(train)
ica_results_test = ica.transform(test)

print("GRP")
grp = GaussianRandomProjection(n_components=N_COMP, eps=0.1, random_state=17)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

print("SRP")
srp = SparseRandomProjection(n_components=N_COMP, dense_output=True, random_state=17)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)

print("Append decomposition components to datasets...")
for i in range(1, N_COMP + 1):
    train['pca_' + str(i)] = pca_results_train[:, i - 1]
    test['pca_' + str(i)] = pca_results_test[:, i - 1]

    train['ica_' + str(i)] = ica_results_train[:, i - 1]
    test['ica_' + str(i)] = ica_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))

print('\nModelling...')
def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))

folds = KFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train)):
    trn_x, trn_y = train.ix[trn_idx], target[trn_idx]
    val_x, val_y = train.ix[val_idx], target[val_idx]
    cb_model = CatBoostRegressor(iterations=1000, learning_rate=0.01, depth=8, l2_leaf_reg=20, bootstrap_type='Bernoulli',  eval_metric='RMSE', metric_period=50, od_type='Iter', od_wait=45, random_seed=17, allow_writing_files=False)
    cb_model.fit(trn_x, trn_y, eval_set=(val_x, val_y), cat_features=[], use_best_model=True, verbose=True)
    oof_preds[val_idx] = cb_model.predict(val_x)
    sub_preds += cb_model.predict(test) / folds.n_splits
    print("Fold %2d RMSLE : %.6f" % (n_fold+1, rmsle(np.exp(val_y)-1, np.exp(oof_preds[val_idx])-1)))

print("Full RMSLE score %.6f" % rmsle(np.exp(target)-1, np.exp(oof_preds)-1)) 
subm['target'] = np.exp(sub_preds)-1
subm.to_csv('submission2.csv', index=False)


# In[34]:


b1 = pd.read_csv('submission1.csv').rename(columns={'target':'dp1'})
b2 = pd.read_csv('submission2.csv').rename(columns={'target':'dp2'})
b1 = pd.merge(b1, b2, how='left', on='ID')
b1['target'] = (b1['dp1'] * 0.5) + (b1['dp2'] * 0.5)
b1[['ID','target']].to_csv('Submission_Santander.csv', index=False)
#!kaggle competitions submit -c santander-value-prediction-challenge -f blend01.csv -m "z02"


# In[ ]:




