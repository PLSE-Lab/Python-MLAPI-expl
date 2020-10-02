#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import interp
import gc

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from catboost import CatBoostClassifier, Pool, cv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("."))


# Any results you write to the current directory are saved as output.


# **Note** - The score is low from kernel submission because only 50% of training data was used for each fold. I did an median ensemble of 4 such models with random 50% of data. The score will improve if you use full dataset with shuffled k-fold (>0.935). I had to do that because my kernel was dying when I was trying to create *Dataset Pool* for Catboost. The creation of Pool eats up a lot of RAM. Another option for me was to reduce memory of dataframe. But I didn't try that. 
# 
# - Catboost with last 5% as validation set
# - Only null treatment required in pre-processing
# - Categorical variables handled by passing as index
# 
# Only 50% of data has been used in each fold. The score is better if use stratified fold so that it covers all the data. 
# 
# Reduce data memory size by using this kernel - https://www.kaggle.com/mjbahmani/reducing-memory-size-for-ieee/output
# Found this kernel to be useful for experiments and took some code from here - https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt    [](http://)

# In[ ]:


train_trans = pd.read_csv('../input/train_transaction.csv')
test_trans = pd.read_csv('../input/test_transaction.csv')
train_iden = pd.read_csv('../input/train_identity.csv') 
test_iden = pd.read_csv('../input/test_identity.csv')

train_trans.shape, test_trans.shape, train_iden.shape, test_iden.shape


# In[ ]:


df_train = train_trans.merge(train_iden, on="TransactionID", how="left")
df_test = test_trans.merge(test_iden, on="TransactionID", how="left")
del train_iden, train_trans, test_iden, test_trans


# In[ ]:


# df_train = pd.read_csv('train_reduced.csv')
# df_test = pd.read_csv('test_reduced.csv')


# In[ ]:


df_train.head()


# In[ ]:


# filling NAs
df_train.fillna(-999, inplace=True)
df_test.fillna(-999, inplace=True)


# In[ ]:


drop_cols = ["TransactionID", "TransactionDT"]
label_col = "isFraud"


# In[ ]:


y_train = df_train.loc[:,label_col]


# In[ ]:


df_train.drop([label_col]+drop_cols, axis=1, inplace=True)
df_test.drop(["TransactionDT"], axis=1, inplace=True)


# In[ ]:


features = list(df_train)


# In[ ]:


gc.collect()


# Defining categorical features. Defining Card, Email, Device, Product, M, id12-id38, addr as categorical features

# In[ ]:


cat_cols = ["card{}".format(i) for i in range(1,7)] +            ['ProductCD', 'P_emaildomain', 'R_emaildomain',  'DeviceType', 'DeviceInfo'] +            ["M{}".format(i) for i in range(1,10)] +            ["id_{}".format(i) for i in range(12, 39)] +            ["addr{}".format(i) for i in range(1,3)]


# In[ ]:


cat_idx = [df_train.columns.get_loc(c) for c in cat_cols]


# * We create random subsamples from the training data. The reason is that running Pool function exceeds the kernel memory limit and kernel dies. So we create 5 models with 66% bootstrap samples and average predictions from these 5 models for test set. 

# In[ ]:


nfold = 5
# skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
# tscv = TimeSeriesSplit(n_splits=5)
predictions_df = pd.DataFrame()

seeds = [1,12,123,21,423]

mean_fpr = np.linspace(0,1,100)
roc_aucs = []
tprs = []
cms= []
aucs = []
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

i = 1
# for train_idx, valid_idx in skf.split(df_train_con, df_train[label_col].values):
for i in range(nfold):
    np.random.seed(seeds[i])  # to reporoduce
    size_95_per = int(0.95*df_train.shape[0])
    train_idx = np.random.choice(range(size_95_per),
                                 int(0.5*size_95_per), replace=False)  # 70% sampling random
    valid_idx = range(size_95_per, df_train.shape[0]) # last 5% of data
    print("\nfold {}".format(i))
    print("train pool")
    trn_data = Pool(df_train.iloc[train_idx].values,
                     y_train.iloc[train_idx].values,
                     cat_features=cat_idx)
    gc.collect()
    print("valid pool")
    val_data = Pool(df_train.iloc[valid_idx].values,
                    y_train.iloc[valid_idx].values,
                    cat_features=cat_idx) 
    gc.collect()
    clf = CatBoostClassifier(iterations=1500,
                           random_state=10,
                           learning_rate=0.08,
                           task_type = "GPU",
                           eval_metric= 'AUC', 
                           scale_pos_weight = sum(y_train.iloc[train_idx]==0)/sum(y_train.iloc[train_idx]==1)/5.,
#                            one_hot_max_size = 4,
#                            has_time=True,
#                            min_data_in_leaf=5,
                           early_stopping_rounds = 50,
                          )
    print("model")
    clf.fit(trn_data,
          use_best_model=True,
          eval_set=val_data,
          verbose=False,
          plot=True)
    gc.collect()
    print("predict valid")
    oof = clf.predict_proba(df_train.iloc[valid_idx].values)[:,1] 
    print("predict test")
    this_fold_preds = clf.predict_proba(df_test.drop("TransactionID", axis=1))[:,1]
    predictions += this_fold_preds/np.float(nfold)
    predictions_df[str(i)] = this_fold_preds
    
    # Scores 
    roc_aucs.append(roc_auc_score(y_train.iloc[valid_idx].values, oof))
   
    # Roc curve by fold
    fpr, tpr, t = roc_curve(y_train.iloc[valid_idx].values, oof)
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i,roc_auc))
    # don't need them now
    del trn_data, val_data


# In[ ]:


del df_train
gc.collect()


# In[ ]:


h=plt.hist(predictions)  # mean of folds 


# Submitting median of predictions. You can also submit meanm

# In[ ]:


h = plt.hist(predictions_df.median(1))


# Test submission

# In[ ]:


test_submission = pd.DataFrame()
test_submission['TransactionID'] = df_test.iloc[0:506691,0]
test_submission['isFraud'] = predictions_df.median(1)
test_submission.head()


# In[ ]:


test_submission.to_csv('submission_catboost_baseline.csv', index=False)


# ## End
