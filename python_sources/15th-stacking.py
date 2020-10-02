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
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
print (os.listdir('../input/'))
# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold\nfrom sklearn.linear_model import BayesianRidge\n\ndef group_mean_log_mae(y_true, y_pred, types, floor=1e-9):\n    """\n    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling\n    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric\n    """\n    maes = (y_true-y_pred).abs().groupby(types).mean()\n    return np.log(maes.map(lambda x: max(x, floor))).mean()\n\ntrain = pd.read_csv(\'../input/pmp-oof/final_train_oof_pmp.csv\')\ntest = pd.read_csv(\'../input/pmp-oof/final_test_oof_pmp.csv\')\ndrop_features=[\'id\',\'type\',\'scalar_coupling_constant\',\'molecule_name\'\n              ]#\nfeats = [f for f in train.columns if f not in drop_features]\nprint (\'features:\',feats)\n\nn_splits= 5\nfolds = GroupKFold(n_splits=n_splits)\noof_preds = np.zeros((train.shape[0]))\nsub_preds = np.zeros((test.shape[0]))\n\nfor t in train[\'type\'].unique():\n    train_t = train[train[\'type\']==t].reset_index(drop=True)\n    idx = train[train[\'type\']==t].index\n    test_t = test[test[\'type\']==t].reset_index(drop=True)\n    idx_test = test[test[\'type\']==t].index\n    cv_list = []\n    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_t,groups=train_t[\'molecule_name\'])):\n        train_x, train_y = train_t[feats].iloc[train_idx], train_t[\'scalar_coupling_constant\'].iloc[train_idx]\n        valid_x, valid_y = train_t[feats].iloc[valid_idx], train_t[\'scalar_coupling_constant\'].iloc[valid_idx] \n        valid_type = train_t[\'type\'].iloc[valid_idx] \n        train_x = train_x.values\n        valid_x = valid_x.values\n        \n        clf = BayesianRidge(\n                            n_iter=1000,#50\n                            tol=0.1,#0.01\n                            normalize=False)\n        clf.fit(train_x, train_y)\n    \n        oof_preds[idx[valid_idx]] = clf.predict(valid_x)\n        oof_cv = group_mean_log_mae(y_true=valid_y, \n                              y_pred=oof_preds[idx[valid_idx]], \n                              types=valid_type)\n\n        cv_list.append(oof_cv)\n        sub_preds[idx_test]  += clf.predict(test_t[feats].values) / folds.n_splits\n    print (\'type=\' + str(t),cv_list) \n\ntrain[\'stacking\'] = oof_preds\ntest[\'scalar_coupling_constant\'] = sub_preds\n\nfor t in train[\'type\'].unique():\n    train_t = train[train[\'type\']==t]\n    oof_type = group_mean_log_mae(y_true=train_t[\'scalar_coupling_constant\'], \n                            y_pred=train_t[\'stacking\'], \n                            types=train_t[\'type\'])        \n    print (t,oof_type)\n\noof_full = group_mean_log_mae(y_true=train[\'scalar_coupling_constant\'], \n                            y_pred=oof_preds, \n                            types=train[\'type\'])        \nprint (\'All type\',oof_full)\n\ntrain[[\'id\',\'stacking\']].to_csv(\'train_stacking.csv\',index=False)\ntest[[\'id\',\'scalar_coupling_constant\']].to_csv(\'test_stacking.csv\',index=False)')


# In[ ]:




