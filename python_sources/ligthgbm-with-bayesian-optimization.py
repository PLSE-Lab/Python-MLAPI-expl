#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import KFold


# In[ ]:


get_ipython().run_cell_magic('time', '', "data = pd.read_csv('../input/santander-value-prediction-challenge/train.csv')\ntarget = np.log1p(data['target'])\ndata.drop(['ID', 'target'], axis=1, inplace=True)")


# ### Add train leak

# In[ ]:


get_ipython().run_cell_magic('time', '', "leak = pd.read_csv('../input/breaking-lb-fresh-start-with-lag-selection/train_leak.csv')\ndata['leak'] = leak['compiled_leak'].values\ndata['log_leak'] = np.log1p(leak['compiled_leak'].values)")


# ### Feature Scoring using XGBoost with the leak feature

# In[ ]:


# %%time
# def rmse(y_true, y_pred):
#     return mean_squared_error(y_true, y_pred) ** .5

# reg = XGBRegressor(n_estimators=1000)
# folds = KFold(4, True, 134259)
# fold_idx = [(trn_, val_) for trn_, val_ in folds.split(data)]
# scores = []

# nb_values = data.nunique(dropna=False)
# nb_zeros = (data == 0).astype(np.uint8).sum(axis=0)

# features = [f for f in data.columns if f not in ['log_leak', 'leak', 'target', 'ID']]
# for _f in features:
#     score = 0
#     for trn_, val_ in fold_idx:
#         reg.fit(
#             data[['log_leak', _f]].iloc[trn_], target.iloc[trn_],
#             eval_set=[(data[['log_leak', _f]].iloc[val_], target.iloc[val_])],
#             eval_metric='rmse',
#             early_stopping_rounds=50,
#             verbose=False
#         )
#         score += rmse(target.iloc[val_], reg.predict(data[['log_leak', _f]].iloc[val_], ntree_limit=reg.best_ntree_limit)) / folds.n_splits
#     scores.append((_f, score))


# ### Create dataframe

# In[ ]:


# report = pd.DataFrame(scores, columns=['feature', 'rmse']).set_index('feature')
# report['nb_zeros'] = nb_zeros
# report['nunique'] = nb_values
# report.sort_values(by='rmse', ascending=True, inplace=True)


# In[ ]:


# report.to_csv('feature_report.csv', index=True)


# In[ ]:


report = pd.read_csv('../input/feature-report/feature_report.csv', index_col='feature')


# In[ ]:


report.head()


# ### Select some features (threshold is not optimized)

# In[ ]:


good_features = report.loc[report['rmse'] <= 0.7955].index
rmses = report.loc[report['rmse'] <= 0.7955, 'rmse'].values
good_features


# In[ ]:


test = pd.read_csv('../input/santander-value-prediction-challenge/test.csv')


# ### Add leak to test

# In[ ]:


get_ipython().run_cell_magic('time', '', "tst_leak = pd.read_csv('../input/breaking-lb-fresh-start-with-lag-selection/test_leak.csv')\ntest['leak'] = tst_leak['compiled_leak']\ntest['log_leak'] = np.log1p(tst_leak['compiled_leak'])")


# ### Train lightgbm

# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb


# In[ ]:


folds = KFold(n_splits=5, shuffle=True, random_state=1)

# Use all features for stats
features = [f for f in data if f not in ['ID', 'leak', 'log_leak', 'target']]
data.replace(0, np.nan, inplace=True)
data['log_of_mean'] = np.log1p(data[features].replace(0, np.nan).mean(axis=1))
data['mean_of_log'] = np.log1p(data[features]).replace(0, np.nan).mean(axis=1)
data['log_of_median'] = np.log1p(data[features].replace(0, np.nan).median(axis=1))
data['nb_nans'] = data[features].isnull().sum(axis=1)
data['the_sum'] = np.log1p(data[features].sum(axis=1))
data['the_std'] = data[features].std(axis=1)
data['the_kur'] = data[features].kurtosis(axis=1)

test.replace(0, np.nan, inplace=True)
test['log_of_mean'] = np.log1p(test[features].replace(0, np.nan).mean(axis=1))
test['mean_of_log'] = np.log1p(test[features]).replace(0, np.nan).mean(axis=1)
test['log_of_median'] = np.log1p(test[features].replace(0, np.nan).median(axis=1))
test['nb_nans'] = test[features].isnull().sum(axis=1)
test['the_sum'] = np.log1p(test[features].sum(axis=1))
test['the_std'] = test[features].std(axis=1)
test['the_kur'] = test[features].kurtosis(axis=1)


# In[ ]:


# Only use good features, log leak and stats for training
features = good_features.tolist()
features = features + ['log_leak', 'log_of_mean', 'mean_of_log', 'log_of_median', 'nb_nans', 'the_sum', 'the_std', 'the_kur']
dtrain = lgb.Dataset(data=data[features], 
                     label=target, free_raw_data=False)
test['target'] = 0

dtrain.construct()
oof_preds = np.zeros(data.shape[0])


# In[ ]:


from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold, KFold


# In[ ]:


def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest MSE: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")


# In[ ]:


import math

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


# In[ ]:


get_ipython().run_cell_magic('time', '', "bayes_cv_tuner = BayesSearchCV(\n    estimator = lgb.LGBMRegressor(objective='regression', boosting_type='gbdt', subsample=0.6143), #colsample_bytree=0.6453, subsample=0.6143\n    search_spaces = {\n        'learning_rate': (0.01, 1.0, 'log-uniform'),\n        'num_leaves': (10, 100),      \n        'max_depth': (0, 50),\n        'min_child_samples': (0, 50),\n        'max_bin': (100, 1000),\n        'subsample_freq': (0, 10),\n        'min_child_weight': (0, 10),\n        'reg_lambda': (1e-9, 1000, 'log-uniform'),\n        'reg_alpha': (1e-9, 1.0, 'log-uniform'),\n        'scale_pos_weight': (1e-6, 500, 'log-uniform'),\n        'n_estimators': (50, 150),\n    },    \n    scoring = 'neg_mean_squared_log_error', #neg_mean_squared_log_error\n    cv = KFold(\n        n_splits=5,\n        shuffle=True,\n        random_state=42\n    ),\n    n_jobs = 1,\n    n_iter = 100,   \n    verbose = 0,\n    refit = True,\n    random_state = 42\n)\n\n# Fit the model\nresult = bayes_cv_tuner.fit(data[features], target, callback=status_print)")


# ### Save submission

# In[ ]:


pred = bayes_cv_tuner.predict(test[features])


# In[ ]:


test['target'] = np.expm1(pred)
test[['ID', 'target']].to_csv('my_submission.csv', index=False, float_format='%.2f')


# In[ ]:


test.loc[test['leak'].notnull(), 'target'] = test.loc[test['leak'].notnull(), 'leak']
test[['ID', 'target']].to_csv('submission.csv', index=False, float_format='%.2f')


# In[ ]:




