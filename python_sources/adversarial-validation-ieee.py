#!/usr/bin/env python
# coding: utf-8

# For more details :
# http://fastml.com/adversarial-validation-part-one/
# https://www.kaggle.com/konradb/adversarial-validation-and-other-scary-terms
# https://www.kaggle.com/ogrellier/adversarial-validation-and-lb-shakeup
# 
# Basic data processing from:
# https://www.kaggle.com/artkulak/ieee-fraud-simple-baseline-0-9383-lb
# 
# **Some code snippet below is from other past competition kernels , but now I don't remember the owner, if you are the one please mention in comment and I will add your credit here :)**

# In[ ]:


# Load libraries
import numpy as np
import pandas as pd
import gc
import datetime

from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn import preprocessing
# Params
NFOLD = 5
DATA_PATH = '../input/'


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')\ntest_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')\n\ntrain_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')\ntest_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')\n\nsample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')\n\ntrain = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)\ntest = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)\n\ndel train_transaction, train_identity, test_transaction, test_identity\n\n\n# Label Encoding\nfor f in test.columns:\n    if train[f].dtype=='object' or test[f].dtype=='object': \n        lbl = preprocessing.LabelEncoder()\n        lbl.fit(list(train[f].values) + list(test[f].values))\n        train[f] = lbl.transform(list(train[f].values))\n        test[f] = lbl.transform(list(test[f].values))\n        \nprint(train.shape)\nprint(test.shape)")


# In[ ]:


get_ipython().run_cell_magic('time', '', '# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage\n# WARNING! THIS CAN DAMAGE THE DATA \ndef reduce_mem_usage(df):\n    """ iterate through all the columns of a dataframe and modify the data type\n        to reduce memory usage.        \n    """\n    start_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage of dataframe is {:.2f} MB\'.format(start_mem))\n    \n    for col in df.columns:\n        col_type = df[col].dtype\n        \n        if col_type != object:\n            c_min = df[col].min()\n            c_max = df[col].max()\n            if str(col_type)[:3] == \'int\':\n                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n                    df[col] = df[col].astype(np.int8)\n                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n                    df[col] = df[col].astype(np.int16)\n                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n                    df[col] = df[col].astype(np.int32)\n                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n                    df[col] = df[col].astype(np.int64)  \n            else:\n                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n                    df[col] = df[col].astype(np.float16)\n                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n                    df[col] = df[col].astype(np.float32)\n                else:\n                    df[col] = df[col].astype(np.float64)\n        else:\n            df[col] = df[col].astype(\'category\')\n\n    end_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage after optimization is: {:.2f} MB\'.format(end_mem))\n    print(\'Decreased by {:.1f}%\'.format(100 * (start_mem - end_mem) / start_mem))\n    \n    return df\ntrain = reduce_mem_usage(train)\ntest = reduce_mem_usage(test)')


# In[ ]:


train.drop('isFraud', axis=1,inplace=True)
# Mark train as 1, test as 0
train['target'] = 1
test['target'] = 0

# Concat dataframes
n_train = train.shape[0]
df = pd.concat([train, test], axis = 0)
del train, test
gc.collect()


# In[ ]:


df.head()


# In[ ]:


# Remove columns with only one value in our training set
predictors = list(df.columns.difference(['target']))
df_train = df.iloc[:n_train].copy()
cols_to_remove = [c for c in predictors if df_train[c].nunique() == 1]
df.drop(cols_to_remove, axis=1, inplace=True)

# Update column names
predictors = list(df.columns.difference(['target']))

# Get some basic meta features
df['cols_mean'] = df[predictors].replace(0, np.NaN).mean(axis=1)
df['cols_count'] = df[predictors].replace(0, np.NaN).count(axis=1)
df['cols_sum'] = df[predictors].replace(0, np.NaN).sum(axis=1)
df['cols_std'] = df[predictors].replace(0, np.NaN).std(axis=1)


# In[ ]:


# Prepare for training

# Shuffle dataset
df = df.iloc[np.random.permutation(len(df))]
df.reset_index(drop = True, inplace = True)

# Get target column name
target = 'target'

# lgb params
lgb_params = {
        'boosting': 'gbdt',
        'application': 'binary',
        'metric': 'auc', 
        'learning_rate': 0.1,
        'num_leaves': 32,
        'max_depth': 8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'feature_fraction': 0.7,
}

# Get folds for k-fold CV
folds = KFold(n_splits = NFOLD, shuffle = True, random_state = 0)
fold = folds.split(df)
    
eval_score = 0
n_estimators = 0
eval_preds = np.zeros(df.shape[0])


# In[ ]:


# Run LightGBM for each fold
for i, (train_index, test_index) in enumerate(fold):
    print( "\n[{}] Fold {} of {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i+1, NFOLD))
    train_X, valid_X = df[predictors].values[train_index], df[predictors].values[test_index]
    train_y, valid_y = df[target].values[train_index], df[target].values[test_index]

    dtrain = lgb.Dataset(train_X, label = train_y,
                          feature_name = list(predictors)
                          )
    dvalid = lgb.Dataset(valid_X, label = valid_y,
                          feature_name = list(predictors)
                          )
        
    eval_results = {}
    
    bst = lgb.train(lgb_params, 
                         dtrain, 
                         valid_sets = [dtrain, dvalid], 
                         valid_names = ['train', 'valid'], 
                         evals_result = eval_results, 
                         num_boost_round = 5000,
                         early_stopping_rounds = 100,
                         verbose_eval = 100)
    
    print("\nRounds:", bst.best_iteration)
    print("AUC: ", eval_results['valid']['auc'][bst.best_iteration-1])

    n_estimators += bst.best_iteration
    eval_score += eval_results['valid']['auc'][bst.best_iteration-1]
   
    eval_preds[test_index] += bst.predict(valid_X, num_iteration = bst.best_iteration)
    
n_estimators = int(round(n_estimators/NFOLD,0))
eval_score = round(eval_score/NFOLD,6)

print("\nModel Report")
print("Rounds: ", n_estimators)
print("AUC: ", eval_score)    


# In[ ]:


# Feature importance
lgb.plot_importance(bst, max_num_features = 20)


# As we can see, the separation is almost perfect - which strongly suggests that the train / test rows are very easy to distinguish. **Meaning the distribution of Train and Test is not the same**

# In[ ]:




