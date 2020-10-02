#!/usr/bin/env python
# coding: utf-8

# ### XGBoost with Feature Interactions 
# Old but gold: [xgbfir](https://github.com/limexp/xgbfir)

# In[ ]:


get_ipython().system('pip install xgbfir')


# In[ ]:


import numpy as np 
import pandas as pd 
import gc
import os
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import *
import xgboost as xgb
from sklearn import preprocessing
import xgbfir
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        gc.collect()
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    gc.collect()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')\ntest_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')\n\ntrain_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')\ntest_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')\n\nsample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')\ngc.collect()")


# In[ ]:


train_transaction.head()


# In[ ]:


train_identity.head()


# In[ ]:


train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


y_train = train['isFraud'].copy()


# In[ ]:


gc.collect()


# In[ ]:


train.head() 


# In[ ]:


del train_transaction, train_identity, test_transaction, test_identity
gc.collect()


# In[ ]:


train['nulls'] = train.isna().sum(axis=1)
test['nulls'] = test.isna().sum(axis=1)


# In[ ]:


train['D9'] = (train['TransactionDT']%(3600*24)/3600//1)/24.0
test['D9'] = (test['TransactionDT']%(3600*24)/3600//1)/24.0


# #### Dropping columns and mapping emails. Thanks for the code the1owl!
# https://www.kaggle.com/jazivxt/safe-box

# In[ ]:


train_cols = [c for c in train.columns if c not in list(set(['TransactionDT', 'id_30','id_31','id_33', 'id_35', "id_32", "D15", "D10", "D4", "C13", "D11", "C9", "C11", 'V300', 'V309', 'V111', 'C3', 'V124', 'V106', 'V125', 'V315', 'V134', 'V102', 'V123', 'V316', 'V113', 'V136', 'V305', 'V110', 'V299', 'V289', 'V286', 'V318', 'V103', 'V304', 'V116', 'V298', 'V284', 'V293', 'V137', 'V295', 'V301', 'V104', 'V311', 'V115', 'V109', 'V119', 'V321', 'V114', 'V133', 'V122', 'V319', 'V105', 'V112', 'V118', 'V117', 'V121', 'V108', 'V135', 'V320', 'V303', 'V297', 'V120',"isFraud", "id_32", "D15", 'D9','D15', 'V107', 'V28', 'V305', 'V68', 'V117', 'V119', 'V241', 'V27', 'V65', 'V89', 'V122', 'V88']))]


# In[ ]:


emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
us_emails = ['gmail', 'net', 'edu']


# In[ ]:


for c in ['P_emaildomain', 'R_emaildomain']:
    train[c + '_bin'] = train[c].map(emails)
    test[c + '_bin'] = test[c].map(emails)
    
    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
    test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])
    
    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    gc.collect()


# In[ ]:


cat_cols = train.select_dtypes("object").columns.tolist()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for f in cat_cols:\n    lbl = preprocessing.LabelEncoder()\n    lbl.fit(list(train[f].values) + list(test[f].values))\n    train[f] = lbl.transform(list(train[f].values))\n    test[f] = lbl.transform(list(test[f].values))   \n    gc.collect()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = reduce_mem_usage(train)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test = reduce_mem_usage(test)')


# In[ ]:


n_fold = 6
folds = KFold(n_fold, shuffle=False, random_state=11)


# In[ ]:


params = {'tree_method': 'hist',
 'silent': 1,
 'colsample_bytree': 0.8999504397295506,
 'subsample': 0.7446134812140273,
 'learning_rate': 0.05,
 'max_leaves': 72,
 'objective': 'binary:logistic',
 'max_depth': 0,
 'reg_alpha': 0.7726783188295172,
 'min_child_weight': 2,
 'eval_metric': 'auc',
 'grow_policy':'lossguide'}
params


# In[ ]:


get_ipython().run_cell_magic('time', '', '# for loop with a lot of gc \n\noof_preds = np.zeros(train.shape[0])\nsub_preds = np.zeros(test.shape[0])\n\nfeature_importance_df = pd.DataFrame()\n\nfor n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, y_train)):\n    \n    trn_x, trn_y = train[train_cols].iloc[trn_idx], y_train.iloc[trn_idx]\n    val_x, val_y = train[train_cols].iloc[val_idx], y_train.iloc[val_idx]\n    gc.collect()\n    dtrain = xgb.DMatrix(trn_x, trn_y, feature_names=trn_x.columns)\n    dval = xgb.DMatrix(val_x, val_y, feature_names=val_x.columns)\n    gc.collect()\n    \n    clf = xgb.train(params=params, dtrain=dtrain, num_boost_round=1500, evals=[(dtrain, "Train"), (dval, "Val")],\n        verbose_eval= 250, early_stopping_rounds=100) \n    gc.collect()\n    \n    oof_preds[val_idx] = clf.predict(xgb.DMatrix(val_x))\n    sub_preds += clf.predict(xgb.DMatrix(test[train_cols])) / folds.n_splits\n    gc.collect()\n    \n    xgbfir.saveXgbFI(clf, feature_names=trn_x.columns, OutputXlsxFile=\'ieee_xgbfir_%sFold.xlsx\'%str(n_fold+1), MaxInteractionDepth=9, MaxHistograms=15)\n    gc.collect()\n    \n    fold_importance_df = pd.DataFrame()\n    fold_importance_df["feature"] = pd.DataFrame.from_dict(data=clf.get_fscore(), orient="index", columns=["FScore"])["FScore"].index\n    fold_importance_df["fscore"] = pd.DataFrame.from_dict(data=clf.get_fscore(), orient="index", columns=["FScore"])["FScore"].values\n    fold_importance_df["fold"] = n_fold + 1\n    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)\n    gc.collect()\n    \n    print(\'\\nFold %2d AUC %.6f & std %.6f\' %(n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx]), np.std([oof_preds[val_idx]])))\n    gc.collect()\n\nprint(\'\\nCV AUC score %.6f & std %.6f\' % (roc_auc_score(y_train, oof_preds), np.std((oof_preds))))')


# In[ ]:


gc.collect()


# ### XGBoost feature importance types:
# * "weight" is the number of times a feature appears in a tree
# * "gain" is the average gain of splits which use the feature
# * "cover" is the average coverage of splits which use the feature where coverage is defined as the number of samples affected by the split

# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(10,12)) 
xgb.plot_importance(clf, max_num_features=20, ax=ax)  


# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(10,12)) 
xgb.plot_importance(clf, max_num_features=20, ax=ax, importance_type="cover", xlabel="Cover")


# In[ ]:


# Same here V258 - https://www.kaggle.com/c/ieee-fraud-detection/discussion/100754
fig, ax = plt.subplots(1,1,figsize=(10,12)) 
xgb.plot_importance(clf, max_num_features=20, ax=ax, importance_type="gain", xlabel="Gain")


# In[ ]:


feature_importance_df.groupby(["feature"])["fscore",].mean().sort_values("fscore", ascending=False)


# In[ ]:


sample_submission['isFraud'] = sub_preds
gc.collect()


# In[ ]:


sample_submission.head()


# In[ ]:


score = roc_auc_score(y_train, oof_preds)
score


# In[ ]:


sample_submission.to_csv('sub_ieee_sample_xgb_%sfold_%.6f.csv'%(n_fold, score))


# In[ ]:


oof_train = pd.DataFrame(oof_preds, columns=["oof_preds"], index=train.index) 
oof_train.to_csv("ieee_xgb_%sfold_oof_%.6f.csv" %(n_fold, score)) 
oof_train.head()


# In[ ]:


get_ipython().system('ls -sh')


# #### Let's check the 4th fold feature interactions

# ##### Feature Interaction: https://christophm.github.io/interpretable-ml-book/interaction.html

# In[ ]:


xl = pd.ExcelFile("ieee_xgbfir_4Fold.xlsx")


# In[ ]:


xl.sheet_names


# In[ ]:


# Depth 0 - individual effects of variables
xl.parse(xl.sheet_names[0]).head(10)


# In[ ]:


# Depth 1 - 2-way feature interactions
xl.parse(xl.sheet_names[1]).head(10)


# In[ ]:


# Depth 2 - 3-way feature interactions and so on
xl.parse(xl.sheet_names[2]).head(10)


# Based on these interactions you can create some new features...
