#!/usr/bin/env python
# coding: utf-8

# * Model code is from the baseline + [gpu/Xhulu's kernel](https://www.kaggle.com/xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s)
# * I've added a few simple & generic fraud/anomaly features. Surprisingly, they don't seem to have aneffect here, possibly due to artifacts in the anonymization process, or other reasons. the kernel will be updated with additional features.
# * Isolation forest for anomaly detection feature taken from here: https://www.kaggle.com/danofer/anomaly-detection-for-feature-engineering-v2
# 

# In[ ]:


import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb

from sklearn.ensemble import IsolationForest


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID') # ,nrows=42345\ntest_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID') #,nrows=12345\n\ntrain_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')\ntest_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')\n\nsample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')\n\ntrain = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)\ntest = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)\n\nprint(train.shape)\nprint(test.shape)\n\ny_train = train['isFraud'].copy()\ndel train_transaction, train_identity, test_transaction, test_identity\n\n\ntrain.head()")


# In[ ]:


# ## join train+test for easier feature engineering:
# df = pd.concat([train,test],sort=False)
# print(df.shape)


# ### Add some features
# * missing values count
#     * TODO: nans per cattegory/group (e..g V columns)
#     * Could be more efficient with this code, but that's aimed at columnar, not row level summation: https://stackoverflow.com/questions/54207038/groupby-columns-on-column-header-prefix
# * Add some of the time series identified in external platform
# * ToDo: anomaly detection features. 
# * proxy for lack of an identifier, duplicate values. 
#     * TODO: try to understand what could be a proxy for a key/customer/card identifier (then apply features based on that).
#     
#     
# * ToDo: readd feature of identical transactions: this is typically a strong feature, but (surprisingly) gave no signal in this dataset. Both with and without transaction amount (and with transaction time removed ofc).

# In[ ]:


list(train.columns)

# COLUMN_GROUP_PREFIXES = ["card","C","D","M","V","id"]
COLUMN_GROUP_PREFIXES = ["card","D","M","id"] # "C" , "V" # V has many values, slow, 

def column_group_features(df):
    """
    Note: surprisingly slow! 
    TODO: Check speed, e.g. with `$ pip install line_profiler`
    """
    df["total_missing"] = df.isna().sum(axis=1)
    print("total_missing",df["total_missing"].describe(percentiles=[0.5]))
    df["total_unique_values"] = df.nunique(axis=1)
    print("total_unique_values",df["total_unique_values"].describe(percentiles=[0.5]))
    
    for p in COLUMN_GROUP_PREFIXES:
        col_group = [col for col in df if col.startswith(p)]
        print("total cols in subset:", p ,len(col_group))
        df[p+"_missing_count"] = df[col_group].isna().sum(axis=1)
        print(p+"_missing_count", "mean:",df[p+"_missing_count"].describe(percentiles=[]))
        df[p+"_uniques_count"] = df[col_group].nunique(axis=1)
        print(p+"_uniques_count", "mean:",df[p+"_uniques_count"].describe(percentiles=[]))
#         df[p+"_max_val"] = df[col_group].max(axis=1)
#         df[p+"_min_val"] = df[col_group].min(axis=1)
    print("done \n")
    return df


# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
# WARNING! THIS CAN DAMAGE THE DATA 
def reduce_mem_usage(df,do_categoricals=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
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
            if do_categoricals==True:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


# %%time

train = reduce_mem_usage(train,do_categoricals=False)
test = reduce_mem_usage(test,do_categoricals=False)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain = column_group_features(train)\nprint("train features generated")\n\ntest = column_group_features(test)\n\ntrain.head()')


# ## datetime features
# * try to guess date and datetime delta unit, then add features
# * TODO: strong features potential already found offline, need to validate
# * Try 01.12.2017 as start date: https://www.kaggle.com/kevinbonnes/transactiondt-starting-at-2017-12-01

# In[ ]:


import datetime

START_DATE = '2017-12-01'

# Preprocess date column
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
train['time'] = train['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
test['time'] = test['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

## check if time of day is morning/early night, and/or weekend/holiday:
train["hour_of_day"] = train['time'].dt.hour
test["hour_of_day"] = test['time'].dt.hour

## check if time of day is morning/early night, and/or weekend/holiday: (day of the week with Monday=0, Sunday=6.)
train["day_of_week"] = train['time'].dt.dayofweek
test["day_of_week"] = test['time'].dt.dayofweek

print(train['time'].describe())
print(test['time'].describe())


# In[ ]:


## no clear correlation, but we expect any such features to be categorical in nature, not ordinal/continous. the model can findi t
train[["isFraud","hour_of_day","day_of_week"]].sample(frac=0.1).corr()


# ### label-encode & model build
# * TODO: compare to OHE? +- other encoding/embedding methods

# In[ ]:


# Drop target, fill in NaNs ?
# consider dropping the TransactionDT column as well...
X_train = train.drop(['isFraud',"time"], axis=1)
X_test = test.drop(["time"], axis=1).copy()

del train, test

# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))   


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nX_train = reduce_mem_usage(X_train,do_categoricals=True)\nX_test = reduce_mem_usage(X_test,do_categoricals=True)')


# ## Anomaly detection features
# * Isolation forest approach for now, can easily be improved with semisupervised approach, additional models, TSNE etc'
# * Based on this kernel: https://www.kaggle.com/danofer/anomaly-detection-for-feature-engineering-v2
# 
# * Note: potential improvement: train additional model on only positive (non fraud) samples on concatenated train+test. 
# 
# 
# 
# ##### Isolation forest (anomaly detection)
# * https://www.kaggle.io/svf/1100683/56c8356ed1b0a6efccea8371bc791ba7/__results__.html#Tree-based-techniques )
# * contamination = % of anomalies expected  (fraud class % in our case)
# 
# * isolation forest doesn't work on nan values!
#     * TODO: model +- transaction amount. +- nan imputation (at least/especially for important columns)

# In[ ]:


df_all = pd.concat([X_train.dropna(axis=1),X_test.dropna(axis=1)]).drop(["TransactionDT"],axis=1).dropna(axis=1)
TR_ROWS = X_train.shape[0]
NO_NAN_COLS = df_all.columns
print("num of no nan cols",len(NO_NAN_COLS))
print(df_all.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf = IsolationForest(random_state=42,  max_samples=0.7, bootstrap=True,n_jobs=2,\n                          n_estimators=120,max_features=0.9,behaviour="new",contamination= 0.035)\nclf.fit(df_all)\ndel (df_all)')


# In[ ]:


## add anomalous feature.
## Warning! this is brittle! be careful with the columns!!

X_train["isolation_overall_score"] =clf.decision_function(X_train[NO_NAN_COLS])
X_test["isolation_overall_score"] =clf.decision_function(X_test[NO_NAN_COLS])

print("Fraud only mean anomaly score",X_train.loc[y_train==1]["isolation_overall_score"].mean())
print("Non-Fraud only mean anomaly score",X_train.loc[y_train==0]["isolation_overall_score"].mean())


# In[ ]:


# train only on non fraud samples

clf = IsolationForest(random_state=42,  bootstrap=False,  max_samples=0.85,
                          n_estimators=100,max_features=0.8,behaviour="new",n_jobs=1)
clf.fit(X_train[NO_NAN_COLS].loc[y_train==1].values)

X_train["isolation_pos_score"] =clf.decision_function(X_train[NO_NAN_COLS])
X_test["isolation_pos_score"] =clf.decision_function(X_test[NO_NAN_COLS])

del (clf)

print("Fraud only mean pos-anomaly score",X_train.loc[y_train==1]["isolation_pos_score"].mean())
print("Non-Fraud only mean pos-anomaly score",X_train.loc[y_train==0]["isolation_pos_score"].mean())


# ##### Model training
# 
# * todo: do cross_val_predict (sklearn) using sklearn api for convenience
# * Temporal split :  use sklearn's TimeSeriesSplit (or manual) for early stopping/validation + validation

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import KFold, TimeSeriesSplit\nfrom sklearn.metrics import roc_auc_score\nEPOCHS = 4\nkf = KFold(n_splits = EPOCHS, shuffle = True)\n# kf = TimeSeriesSplit(n_splits = EPOCHS) # temporal validation. use this to evaluate performance better , not necessarily as good for OOV ensembling though!\n\ny_preds = np.zeros(sample_submission.shape[0])\ny_oof = np.zeros(X_train.shape[0])\nfor tr_idx, val_idx in kf.split(X_train, y_train):\n    clf = xgb.XGBClassifier(#n_jobs=2,\n        n_estimators=500,  # 500 default\n        max_depth=9, # 9\n        learning_rate=0.05,\n        subsample=0.9,\n        colsample_bytree=0.9,\n#         tree_method=\'gpu_hist\' # #\'gpu_hist\', - faster, less exact , "gpu_exact" - better perf\n#         ,min_child_weight=2 # 1 by default\n    )\n    \n    X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]\n    y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]\n    clf.fit(X_tr, y_tr)\n    y_pred_train = clf.predict_proba(X_vl)[:,1]\n    y_oof[val_idx] = y_pred_train\n    print(\'ROC AUC {}\'.format(roc_auc_score(y_vl, y_pred_train)))\n    \n    y_preds+= clf.predict_proba(X_test)[:,1] / EPOCHS')


# In[ ]:


# make submissions
sample_submission['isFraud'] = y_preds
sample_submission.to_csv('dan_xgboost.csv')


# #### Simple model based feature importance plot
# * TODO: shapley, interactions
# 
# * It looks like our grouped missing values are **valuable**, although the datetime features seemingly didn't (likely, some of the anonymized variables already capture them). They may have some marginal contribution.
#     * toDo: check that run models with and without them

# In[ ]:


import matplotlib.pyplot as plt

# fi = pd.DataFrame(index=clf.feature_names_)
fi = pd.DataFrame(index=X_train.columns)
fi['importance'] = clf.feature_importances_
fi.loc[fi['importance'] > 0.0005].sort_values('importance').head(50).plot(kind='barh', figsize=(14, 28), title='Feature Importance')
plt.show()

