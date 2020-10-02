#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
debug = False
warnings.simplefilter('ignore')
plt.style.use('seaborn')
random_state = 333


# In[ ]:


def reverse(train_df, test_df):
    reverse_list = [
        0, 1, 2, 3, 4, 5, 6, 8, 11, 15, 16, 18, 19,
        22, 24, 25, 26, 32, 35, 37, 40, 48,
        49, 51, 52, 53, 55, 60, 62, 66, 67, 
        69, 70, 71, 74, 78, 79, 82, 89, 90, 91, 94,
        95, 97, 99, 105, 106, 110, 111, 112, 118,
        119, 125, 128, 130, 133, 134, 135, 137, 138, 140,
        144, 145, 147, 151, 155, 157, 159, 162, 163,
        164, 167, 168, 170, 171, 173, 175, 179, 180,
        181, 184, 187, 189, 190, 191, 195, 196, 199
    ]
    reverse_list = ['var_%d' % i for i in reverse_list]
    for col in reverse_list:
        train_df[col] = train_df[col] * (-1)
        test_df[col] = test_df[col] * (-1)
    return train_df, test_df


# In[ ]:


train_df = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv', index_col='ID_code')
test_df = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv', index_col='ID_code')
public = np.load('../input/helpingdata/public_LB.npy')
private = np.load('../input/helpingdata/private_LB.npy')
fake = np.load('../input/helpingdata/synthetic_samples_indexes.npy')
target = train_df['target']


# In[ ]:


train_df, test_df = reverse(train_df, test_df)


# In[ ]:


real_idx = np.sort(np.concatenate((public, private)))
real_idx


# In[ ]:


test_real = test_df.iloc[real_idx]
test_real.head()


# In[ ]:


train_df.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "concat_df = pd.concat((train_df, test_real))\nfeatures = train_df.columns[1:]\nif debug:\n    features = features[:3]\nfor i, feature in enumerate(features):\n    print('Calculating var_{}'.format(i), end='\\r')\n    n_dups_dict = concat_df.loc[:, feature].value_counts().to_dict()\n    train_df.loc[:, 'count_{}'.format(feature)] = [n_dups_dict.get(value, 1) * np.sign(value)  for value in train_df.loc[:, feature]]\n    test_df.loc[:, 'count_{}'.format(feature)] = [n_dups_dict.get(value, 1) * np.sign(value) for value in test_df.loc[:, feature]]\n    train_df.loc[:, '{}_x_count_{}'.format(feature, feature)] = train_df.loc[:, feature] * np.absolute(train_df.loc[:, 'count_{}'.format(feature)])\n    test_df.loc[:, '{}_x_count_{}'.format(feature, feature)] = test_df.loc[:, feature] * np.absolute(test_df.loc[:, 'count_{}'.format(feature)])\n    \n    train_df.loc[:, 'count_inverse_{}'.format(feature)] = [(1 / n_dups_dict.get(value, 1)) * np.sign(value)  for value in train_df.loc[:, feature]]\n    test_df.loc[:, 'count_inverse_{}'.format(feature)] = [(1 / n_dups_dict.get(value, 1)) * np.sign(value) for value in test_df.loc[:, feature]]\n    train_df.loc[:, '{}_x_count_inverse_{}'.format(feature, feature)] = train_df.loc[:, feature] * np.absolute(train_df.loc[:, 'count_inverse_{}'.format(feature)])\n    test_df.loc[:, '{}_x_count_inverse_{}'.format(feature, feature)] = test_df.loc[:, feature] * np.absolute(test_df.loc[:, 'count_inverse_{}'.format(feature)])\n\nprint(train_df.shape)")


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


# for i in range(len(features)):
#     sns.distplot(train_df.loc[train_df.target == 0, 'count_var_{}'.format(i)], hist=False)
#     sns.distplot(train_df.loc[train_df.target == 1, 'count_var_{}'.format(i)], hist=False)
#     plt.show()
#     sns.distplot(train_df.loc[train_df.target == 0, 'var_{}_x_count_var_{}'.format(i,i)], hist=False)
#     sns.distplot(train_df.loc[train_df.target == 1, 'var_{}_x_count_var_{}'.format(i,i)], hist=False)
#     plt.show()
#     sns.distplot(train_df.loc[train_df.target == 0, 'count_inverse_var_{}'.format(i)], hist=False)
#     sns.distplot(train_df.loc[train_df.target == 1, 'count_inverse_var_{}'.format(i)], hist=False)
#     plt.show()
#     sns.distplot(train_df.loc[train_df.target == 0, 'var_{}_x_count_inverse_var_{}'.format(i,i)], hist=False)
#     sns.distplot(train_df.loc[train_df.target == 1, 'var_{}_x_count_inverse_var_{}'.format(i,i)], hist=False)
#     plt.show()


# In[ ]:


gc.collect()


# In[ ]:


lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 7,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 1,
    "feature_fraction" : 0.3,
    "min_data_in_leaf": 80,
    "min_sum_hessian_in_leaf" : 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state
}


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_train = train_df.reset_index()\ndf_test = test_df.reset_index()\n\nif debug:\n    df_train = df_train[:1000]\n    df_test = df_test[:1000]\n    target = target[:1000]\n    \nskf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=random_state) # keep splits = 10 for best results\noof = df_train[[\'ID_code\', \'target\']]\noof[\'predict\'] = 0\npredictions = df_test[[\'ID_code\']]\nval_aucs = []\nfeature_importance_df = pd.DataFrame()\nfeatures = [col for col in df_train.columns if col not in [\'target\', \'ID_code\']]\nX_test = df_test[features].values\n\nfor fold, (trn_idx, val_idx) in enumerate(skf.split(df_train, target)):\n    X_train, y_train = df_train.iloc[trn_idx][features], df_train.iloc[trn_idx][\'target\']\n    X_valid, y_valid = df_train.iloc[val_idx][features], df_train.iloc[val_idx][\'target\']\n    print("shape of training data: ", X_train.shape)\n    trn_data = lgb.Dataset(X_train, label=y_train)\n    val_data = lgb.Dataset(X_valid, label=y_valid)\n    evals_result = {}\n    lgb_clf = lgb.train(lgb_params,\n                    trn_data,\n                    100000,\n                    valid_sets = [trn_data, val_data],\n                    early_stopping_rounds=1000,\n                    verbose_eval = 1000,\n                    evals_result=evals_result\n                   )\n    p_valid = lgb_clf.predict(X_valid, num_iteration=lgb_clf.best_iteration)\n    yp = lgb_clf.predict(X_test, num_iteration=lgb_clf.best_iteration)\n    fold_importance_df = pd.DataFrame()\n    fold_importance_df["feature"] =  [col for col in X_train.columns]\n    fold_importance_df["importance"] = lgb_clf.feature_importance()\n    fold_importance_df["fold"] = fold + 1\n    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)\n    oof[\'predict\'][val_idx] = p_valid\n    val_score = roc_auc_score(y_valid, p_valid)\n    val_aucs.append(val_score)\n    predictions[\'fold{}\'.format(fold+1)] = yp')


# In[ ]:


gc.collect()


# In[ ]:


mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print("Mean auc: %.5f, std: %.5f. All auc: %.5f." % (mean_auc, std_auc, all_auc))


# In[ ]:


# submission
predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
predictions.to_csv('lgb_all_predictions.csv', index=None)
sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
sub_df["target"] = predictions['target']
sub_df.to_csv("submission.csv", index=False)
oof.to_csv('lgb_oof.csv', index=False)


# In[ ]:




