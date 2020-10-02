#!/usr/bin/env python
# coding: utf-8

# # Categorical feature Encoding2
# 
# This kernel is a version using one-hot encoding
# 
# I created this kernel by referring to the following kernel:  
# https://www.kaggle.com/peterhurford/why-not-logistic-regression  
# Thanks!
# 

# In[ ]:


# # !pip uninstall sklearn -y
# !pip install -U scikit-learn==0.22.1
# import sklearn
# sklearn.__version__


# In[ ]:


import numpy as np
import pandas as pd
import scipy
import os, gc
from collections import Counter
from sklearn.model_selection import KFold,StratifiedKFold,RepeatedKFold,RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
import category_encoders as ce

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = 50
BIN_COL  = [f'bin_{i}' for i in range(5)]
NOM_COL  = [f'nom_{i}' for i in range(10)]
ORD_COL  = [f'ord_{i}' for i in range(6)]
NOM_5_9  = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
NOM_0_4  = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']
DATE_COL = ['day','month']
# from imblearn.over_sampling import RandomOverSampler,SMOTE
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef read_csv():\n    train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')\n    test  = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')\n\n    train_id = train['id']\n    test_id  = test['id']\n    train.drop('id', axis=1, inplace=True)\n    test.drop('id',  axis=1, inplace=True)\n    return train, test, train_id, test_id\n\ndef preprocessing(df):\n    df.bin_3.replace({'F':0, 'T':1}, inplace=True)\n    df.bin_4.replace({'N':0, 'Y':1}, inplace=True)\n   \n    ord_1_map = {'Novice':1,'Contributor':2,'Expert':3,'Master':4,'Grandmaster':5}\n    ord_2_map = {'Freezing':1, 'Cold':2,'Warm':3,'Hot':4, 'Boiling Hot':5,'Lava Hot':6}\n    df.loc[df['ord_1'].notnull(),'ord_1'] = df.loc[df['ord_1'].notnull(),'ord_1'].map(ord_1_map)\n    df.loc[df['ord_2'].notnull(),'ord_2'] = df.loc[df['ord_2'].notnull(),'ord_2'].map(ord_2_map)\n    df.loc[df['ord_3'].notnull(),'ord_3'] = df.loc[df['ord_3'].notnull(),'ord_3'].apply(\n        lambda c: ord(c) - ord('a') + 1)\n    df.loc[df['ord_4'].notnull(),'ord_4'] = df.loc[df['ord_4'].notnull(),'ord_4'].apply(\n        lambda c: ord(c) - ord('A') + 1)\n    for col in ['ord_1','ord_2','ord_3','ord_4',]:\n        df[col] = df[col].astype(np.float32)\n    \n    df.day = df.day.replace({3:5,2:6,1:7})\n    df.loc[df.ord_5.notnull(), 'ord_5_1'] = df.loc[df.ord_5.notnull(), 'ord_5'].apply(lambda x: x[0])\n    df.loc[df.ord_5.notnull(), 'ord_5_2'] = df.loc[df.ord_5.notnull(), 'ord_5'].apply(lambda x: x[1])\n    df.loc[df['ord_5_1'].notnull(),'ord_5_1'] = df.loc[df['ord_5_1'].notnull(),'ord_5_1'].apply(\n        lambda c: ord(c) - ord('a') + 33).astype(np.float32)\n    df.loc[df['ord_5_2'].notnull(),'ord_5_2'] = df.loc[df['ord_5_2'].notnull(),'ord_5_2'].apply(\n        lambda c: ord(c) - ord('a') + 33)#.astype(float)\n    return df    \n\ndef filling_NaN(df):\n    df.fillna(-1, inplace=True)\n    df.day   = df.day.astype(int)\n    df.month = df.month.astype(int)\n#     print(df.isnull().sum())\n    return df\n\ndef target_encoding(cols, smoothing=1.0, min_samples_leaf=1):\n    for col in cols:\n        encoder = ce.TargetEncoder(cols=col, \n                                   smoothing=smoothing, \n                                   min_samples_leaf=min_samples_leaf)#ce.leave_one_out.LeaveOneOutEncoder()\n        train[f'{col}_mean'] = encoder.fit_transform(train[col], train['target'])[col].astype(np.float32)\n        test[f'{col}_mean']  = encoder.transform(test[col])[col].astype(np.float32)  \n    del encoder\n    gc.collect() ")


# ### preprocessing 

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain, test, train_id, test_id = read_csv()\ntrain = preprocessing(train)\ntest  = preprocessing(test)\nprint(f'train day unique value:{train.day.unique()}')\nprint(f'test  day unique value:{test.day.unique()}')\n\nfor col in test.columns:\n    if len(set(train[col].dropna().unique().tolist())^ set(test[col].dropna().unique().tolist()))>0:\n        train_only = list(set(train[col].dropna().unique().tolist()) - set(test[col].dropna().unique().tolist()))\n        test_only  = list(set(test[col].dropna().unique().tolist()) - set(train[col].dropna().unique().tolist()))\n        print(col, '(train only)', train_only, '(test only)', test_only) \n        train.loc[train[col].isin(train_only), col] = np.NaN\n        test.loc[test[col].isin(test_only), col]    = np.NaN  \n\ntarget_encoding(['ord_5'])\ntarget_encoding(['ord_5_1'], min_samples_leaf=100)#,'ord_5_2'\n# drop_cols = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4','ord_5','ord_5_1']#['ord_5']")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.preprocessing import StandardScaler,RobustScaler,MaxAbsScaler,MinMaxScaler\n\n\nUSE_DUMMIES = BIN_COL+NOM_COL+DATE_COL+['ord_0','ord_1', 'ord_2','ord_3', 'ord_4']#, 'ord_5_2'\nNOT_DUMMIES = [s for s in train.columns if '_mean' in s]\n\nfor col in NOT_DUMMIES:\n    scaler = MaxAbsScaler()#MinMaxScaler()#StandardScaler()\n    train[col] = scaler.fit_transform(train[[col]]).flatten().astype(np.float32)\n    test[col] = scaler.transform(test[[col]]).flatten().astype(np.float32)\ndel scaler;gc.collect()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "print('drop features:',test.columns.drop(USE_DUMMIES+NOT_DUMMIES).tolist())\ntarget = train['target']\ntrain  = train[USE_DUMMIES+NOT_DUMMIES]   \ntest   = test[USE_DUMMIES+NOT_DUMMIES]\n\ntraintest = pd.concat([train, test], sort=False)\ntraintest = traintest.reset_index(drop=True)\ntraintest.head(10)")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntraintest = pd.get_dummies(traintest, \n                           columns=USE_DUMMIES,#traintest.columns,\n                           dummy_na=False,#True,\n                           drop_first=False,#True,\n                           sparse=True, \n                           dtype=np.int8 )\ntraintest.head(10)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef convert_sparse(traintest, train_length):\n#     train_ohe = traintest.iloc[:train_length, :]\n#     test_ohe  = traintest.iloc[train_length:, :]\n    \n#     train_ohe = train_ohe.sparse.to_coo().tocsr().astype(np.float32)\n#     test_ohe  = test_ohe.sparse.to_coo().tocsr().astype(np.float32)\n\n    train_ohe = traintest.iloc[:train_length, :]\n    test_ohe  = traintest.iloc[train_length:, :]\n    \n    train_ohe1 = scipy.sparse.bsr_matrix(train_ohe[NOT_DUMMIES])\n    test_ohe1  = scipy.sparse.bsr_matrix(test_ohe[NOT_DUMMIES])\n    \n    train_ohe2 = scipy.sparse.csr_matrix(train_ohe.drop(columns=NOT_DUMMIES))\n    test_ohe2  = scipy.sparse.csr_matrix(test_ohe.drop(columns=NOT_DUMMIES))\n    \n    train_ohe = scipy.sparse.hstack([train_ohe1,train_ohe2]).tocsr()\n    test_ohe  = scipy.sparse.hstack([test_ohe1, test_ohe2]).tocsr()\n    \n    return train_ohe, test_ohe \n\ntrain_ohe, test_ohe = convert_sparse(traintest, train_length=len(train))\ntrain_ohe.shape, test_ohe.shape')


# In[ ]:


del train, test;gc.collect()


# In[ ]:


train_ohe[0:20].todense()


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef print_cv_scores(label, cv_scores):\n    print(f'{label} cv scores : {cv_scores}')\n    print(f'{label} cv mean score : {np.mean(cv_scores)}')\n    print(f'{label} cv std score : {np.std(cv_scores)}')    \n\ndef run_cv_model(train_ohe, test_ohe, target, model_fn, params={}, \n                 eval_fn=None, label='model', cv=5,  repeats=5):\n    if repeats==1:\n        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)\n#         kf = KFold(n_splits=cv)\n        divide_counts = cv\n    else:\n#         kf = RepeatedKFold(n_splits=cv,n_repeats=repeats, random_state=42)\n        kf = RepeatedStratifiedKFold(n_splits=cv,n_repeats=repeats, random_state=42)\n        divide_counts = kf.get_n_splits()\n        \n    fold_splits = kf.split(train_ohe, target)\n    cv_scores = []\n    pred_full_test = 0\n    pred_train = np.zeros((train_ohe.shape[0]))\n    \n    for fold_id, (train_idx, val_idx) in enumerate(fold_splits):\n        print(f'Started {label} fold:{fold_id} / {divide_counts}')\n        tr_X_ohe, val_X_ohe = train_ohe[train_idx], train_ohe[val_idx]\n#         tr_X_ohe, val_X_ohe = train_ohe.iloc[train_idx], train_ohe.iloc[val_idx]\n        tr_y, val_y = target[train_idx], target[val_idx]\n        print(Counter(tr_y), Counter(val_y))       \n        \n        params2 = params.copy() \n        model, pred_val_y, pred_test_y = model_fn(\n            tr_X_ohe, tr_y, val_X_ohe, val_y, test_ohe, params2)\n        \n        pred_full_test = pred_full_test + pred_test_y\n        pred_train[val_idx] = pred_val_y\n        if eval_fn is not None:\n            cv_score = eval_fn(val_y, pred_val_y)\n            cv_scores.append(cv_score)\n            print(f'{label} cv score {fold_id}: {cv_score}')\n            \n    \n    print_cv_scores(label, cv_scores)    \n    pred_full_test = pred_full_test / divide_counts\n    results = {'label': label, \n               'train': pred_train, \n               'test': pred_full_test, \n               'cv': cv_scores}\n    return results\n\n\ndef runLR(train_X, train_y, val_X, val_y, test_X, params):\n    print('Train LR')\n    model = LogisticRegression(**params)\n    model.fit(train_X, train_y)\n    print('Predict val data')\n    pred_val_y = model.predict_proba(val_X)[:, 1]\n    print('Predict test data')\n    pred_test_y = model.predict_proba(test_X)[:, 1]\n    return model, pred_val_y, pred_test_y\n\nlr_params = {'penalty':'l2', \n             'solver': 'lbfgs', 'C': 0.05,\n#              'class_weight':'balanced', \n             'max_iter':500,#200,#5000\n             'random_state':42,\n            }\n\nresults1 = run_cv_model(\n    train_ohe, test_ohe, target, runLR, \n    lr_params, auc, 'lr', cv=10, repeats=2)#5)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Make submission\nsubmission = pd.DataFrame(\n    {'id': test_id, 'target': results1['test'],})\nsubmission.to_csv('submission.csv', index=False)")


# In[ ]:


plt.figure(figsize=(12,6))
plt.title('distribution of prediction')
sns.distplot(results1['train'])
sns.distplot(results1['test'])
plt.legend(['train','test'])


# In[ ]:


pd.Series(results1['test']).describe()


# In[ ]:


submission[:50]


# In[ ]:


submission[-50:]

