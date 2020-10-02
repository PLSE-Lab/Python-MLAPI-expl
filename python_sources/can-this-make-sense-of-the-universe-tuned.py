#!/usr/bin/env python
# coding: utf-8

# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRkYM0u6uIEJ166J76eddNETmwzTBluf2yMdLWFvKn4aABEwb8z)
# 
# ---
# Outline of Notebook
# 
# ---
# * [1.Loading Important Library](#1.Loading-Important-Library)
# * [2.Helping Function](#2.Helping-Function)
# * [3.Feature Extraction](#3.Feature-Extraction)
#     1. [1.Feature Extract from training set file](#1.Feature-Extract-from-training-set-file)
#     2. [2.Feature Extract from training meta file](#2.Feature-Extract-from-training-meta-file)
# * [4.Parameter Tuning](#4.Parameter-Tuning)
# * [5.Model Design](#5.Model-Design)
# * [6.Model Tuning](#6.Model-Tuning)
# * [7.Define Parameter](#7.Define-Parameter)
# * [8.Best Parameter Search Using Bayesian](#8.Best-Parameter-Search-Using-Bayesian)
# * [9.Plotting results](#9.Plotting-results)
# * [10.Training LGB with Best Tuned Parameter](#10.Training-LGB-with-Best-Tuned-Parameter)
# * [11.Predict the results](#11.Predict-the-results)
# * [12.Final Submission](#12.Final-Submission)
# 
# ---
# 
# 
# Inspired Kernel :  
# 1) olivier's excellent [kernel](https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data)  
# 2) Siddhartha Bayesian Apprioach : [Kernel](https://www.kaggle.com/meaninglesslives/lgb-parameter-tuning/notebook?scriptVersionId=6733705)

# ## 1.Loading Important Library
# ---

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns 
import lightgbm as lgb

from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ## 2.Helping Function
# ---

# In[ ]:


def lgb_multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
    
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False


# In[ ]:


def multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


# ## 3.Feature Extraction
# ---
# ### 1.Feature Extract from training set file

# In[ ]:


gc.enable()

train = pd.read_csv('../input/training_set.csv')
train['flux_ratio_sq'] = np.power(train['flux'] / train['flux_err'], 2.0)
train['flux_by_flux_ratio_sq'] = train['flux'] * train['flux_ratio_sq']

aggs = {
    'mjd': ['min', 'max', 'size'],
    'passband': ['min', 'max', 'mean', 'median', 'std','skew'],
    'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
    'detected': ['mean','std'],
    'flux_ratio_sq':['sum','skew'],
    'flux_by_flux_ratio_sq':['sum','skew'],
}

agg_train = train.groupby('object_id').agg(aggs)
new_columns = [
    k + '_' + agg for k in aggs.keys() for agg in aggs[k]
]
agg_train.columns = new_columns
agg_train['mjd_diff'] = agg_train['mjd_max'] - agg_train['mjd_min']
agg_train['flux_diff'] = agg_train['flux_max'] - agg_train['flux_min']
agg_train['flux_dif2'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_mean']
agg_train['flux_w_mean'] = agg_train['flux_by_flux_ratio_sq_sum'] / agg_train['flux_ratio_sq_sum']
agg_train['flux_dif3'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_w_mean']

del agg_train['mjd_max'], agg_train['mjd_min']
agg_train.head()

del train
gc.collect()


# ### 2.Feature Extract from training meta file
# ---

# In[ ]:


meta_train = pd.read_csv('../input/training_set_metadata.csv')
meta_train.head()

full_train = agg_train.reset_index().merge(
    right=meta_train,
    how='outer',
    on='object_id'
)

if 'target' in full_train:
    y = full_train['target']
    del full_train['target']
classes = sorted(y.unique())

# Taken from Giba's topic : https://www.kaggle.com/titericz
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
# with Kyle Boone's post https://www.kaggle.com/kyleboone
class_weight = {
    c: 1 for c in classes
}
for c in [64, 15]:
    class_weight[c] = 2

print('Unique classes : ', classes)


# In[ ]:


if 'object_id' in full_train:
    oof_df = full_train[['object_id']]
    del full_train['object_id'], full_train['distmod'], full_train['hostgal_specz']
    
    
train_mean = full_train.mean(axis=0)
full_train.fillna(train_mean, inplace=True)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
clfs = []
importances = pd.DataFrame()


# ## 4.Parameter Tuning
# ---

# In[ ]:


dim_learning_rate = Real(low=1e-6, high=1e-1, prior='log-uniform',name='learning_rate')
dim_estimators = Integer(low=50, high=2000,name='n_estimators')
dim_max_depth = Integer(low=1, high=6,name='max_depth')

dimensions = [dim_learning_rate,
              dim_estimators,
              dim_max_depth]

default_parameters = [0.01,1500,4]


# ## 5.Model Design

# In[ ]:


def createModel(learning_rate,n_estimators,max_depth):       

    oof_preds = np.zeros((len(full_train), len(classes)))
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_train.iloc[val_], y.iloc[val_]

        clf = lgb.LGBMClassifier(**lgb_params,learning_rate=learning_rate,
                                n_estimators=n_estimators,max_depth=max_depth)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=lgb_multi_weighted_logloss,
            verbose=False,
            early_stopping_rounds=50
        )
        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        print('fold',fold_+1,multi_weighted_logloss(val_y, clf.predict_proba(val_x, num_iteration=clf.best_iteration_)))

        clfs.append(clf)
    
    loss = multi_weighted_logloss(y_true=y, y_preds=oof_preds)
    print('MULTI WEIGHTED LOG LOSS : %.5f ' % loss)
    
    return loss


# ## 6.Model Tuning

# In[ ]:


@use_named_args(dimensions=dimensions)
def fitness(learning_rate,n_estimators,max_depth):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    n_estimators:      Number of estimators.
    max_depth:         Maximum Depth of tree.
    """

    # Print the hyper-parameters.
    print('learning rate: {0:.2e}'.format(learning_rate))
    print('estimators:', n_estimators)
    print('max depth:', max_depth)
    
    lv= createModel(learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    max_depth = max_depth)
    return lv


# ## 7.Define Parameter

# In[ ]:


lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 14,
    'metric': 'multi_logloss',
    'subsample': .93,
    'colsample_bytree': .75,
    'reg_alpha': .01,
    'reg_lambda': .01,
    'min_split_gain': 0.01,
    'min_child_weight': 10,
    'silent':True,
    'verbosity':-1,
    'nthread':-1
}


# In[ ]:


get_ipython().run_cell_magic('time', '', 'error = fitness(default_parameters)')


# ## 8.Best Parameter Search Using Bayesian

# In[ ]:


# use only if you haven't found out the optimal parameters for xgb. else comment this block.
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=20,
                           x0=default_parameters)


# ## 9.Plotting results

# In[ ]:


plot_convergence(search_result)
plt.show()


# In[ ]:


# optimal parameters found using scikit optimize. use these parameter to initialize the 2nd level model.
print(search_result.x)
learning_rate = search_result.x[0]
n_estimators = search_result.x[1]
max_depth = search_result.x[2]


# In[ ]:


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
clfs = []
importances = pd.DataFrame()


# ## 10.Training LGB with Best Tuned Parameter

# In[ ]:


oof_preds = np.zeros((len(full_train), len(classes)))
for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
    val_x, val_y = full_train.iloc[val_], y.iloc[val_]
    
    clf = lgb.LGBMClassifier(**lgb_params,learning_rate=learning_rate,
                                n_estimators=n_estimators,max_depth=max_depth)
    clf.fit(
        trn_x, trn_y,
        eval_set=[(trn_x, trn_y), (val_x, val_y)],
        eval_metric=lgb_multi_weighted_logloss,
        verbose=100,
        early_stopping_rounds=50
    )
    oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
    print(multi_weighted_logloss(val_y, clf.predict_proba(val_x, num_iteration=clf.best_iteration_)))
    
    imp_df = pd.DataFrame()
    imp_df['feature'] = full_train.columns
    imp_df['gain'] = clf.feature_importances_
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    clfs.append(clf)

print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_true=y, y_preds=oof_preds))

mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 12))
sns.barplot(x='gain', y='feature', data=importances.sort_values('mean_gain', ascending=False))
plt.tight_layout()
plt.savefig('importances.png')


# ## 11.Predict the results

# In[ ]:


meta_test = pd.read_csv('../input/test_set_metadata.csv')

import time

start = time.time()
chunks = 5000000
for i_c, df in enumerate(pd.read_csv('../input/test_set.csv', chunksize=chunks, iterator=True)):
    df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
    df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']
    # Group by object id
    agg_test = df.groupby('object_id').agg(aggs)
    agg_test.columns = new_columns
    agg_test['mjd_diff'] = agg_test['mjd_max'] - agg_test['mjd_min']
    agg_test['flux_diff'] = agg_test['flux_max'] - agg_test['flux_min']
    agg_test['flux_dif2'] = (agg_test['flux_max'] - agg_test['flux_min']) / agg_test['flux_mean']
    agg_test['flux_w_mean'] = agg_test['flux_by_flux_ratio_sq_sum'] / agg_test['flux_ratio_sq_sum']
    agg_test['flux_dif3'] = (agg_test['flux_max'] - agg_test['flux_min']) / agg_test['flux_w_mean']

    del agg_test['mjd_max'], agg_test['mjd_min']
#     del df
#     gc.collect()
    
    # Merge with meta data
    full_test = agg_test.reset_index().merge(
        right=meta_test,
        how='left',
        on='object_id'
    )
    full_test = full_test.fillna(train_mean)
    
    # Make predictions
    preds = None
    for clf in clfs:
        if preds is None:
            preds = clf.predict_proba(full_test[full_train.columns]) / folds.n_splits
        else:
            preds += clf.predict_proba(full_test[full_train.columns]) / folds.n_splits
    
   # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    preds_99 = np.ones(preds.shape[0])
    for i in range(preds.shape[1]):
        preds_99 *= (1 - preds[:, i])
    
    # Store predictions
    preds_df = pd.DataFrame(preds, columns=['class_' + str(s) for s in clfs[0].classes_])
    preds_df['object_id'] = full_test['object_id']
    preds_df['class_99'] = 0.14 * preds_99 / np.mean(preds_99) 
    
    if i_c == 0:
        preds_df.to_csv('predictions.csv',  header=True, mode='a', index=False)
    else: 
        preds_df.to_csv('predictions.csv',  header=False, mode='a', index=False)
        
    del agg_test, full_test, preds_df, preds
    gc.collect()
    
    if (i_c + 1) % 10 == 0:
        print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))


# ## 12.Final Submission

# In[ ]:


z = pd.read_csv('predictions.csv')

print(z.groupby('object_id').size().max())
print((z.groupby('object_id').size() > 1).sum())

z = z.groupby('object_id').mean()

z.to_csv('single_predictions.csv', index=True)


# In[ ]:


##DONE

