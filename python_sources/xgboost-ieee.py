#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold,GroupKFold,StratifiedShuffleSplit


# In[ ]:


os.listdir('../input/')


# In[ ]:


X=pd.read_csv('../input/ieeefeature1/train.csv')
test=pd.read_csv('../input/ieeefeature1/test.csv')
X1=pd.read_pickle('../input/ieeefeature2/X.pkl')
test1=pd.read_pickle('../input/ieeefeature2/test.pkl')
y=pd.read_csv('../input/ieeefeature1/y.csv')


# In[ ]:


X1=X1.reset_index(drop=True)
test1=test1.reset_index(drop=True)
new_columns=[col for col in X1.columns if col not in X.columns]
X1=X1[new_columns]
test1=test1[new_columns]


# In[ ]:


X=pd.concat([X,X1],axis=1)
test=pd.concat([test,test1],axis=1)


# In[ ]:


X.head()


# In[ ]:


test.dtypes.value_counts()


# In[ ]:


drop_columns=['TransactionID','TransactionDT','isFraud','DT','DT_M','DT_W','DT_D','DT_hour','DT_day_week',
              'DT_day_month','DT_M_total','DT_W_total','DT_D_total','uid','uid2','uid3','uid4','uid5','bank_type']

feature_columns=[col for col in X.columns if col not in drop_columns]
# X=X[feature_columns]
# test=test[feature_columns]
object_columns=['M1','id_35','id_36' ,'id_37', 'id_38']
encoder=LabelEncoder()
for col in object_columns:
    X[col]=encoder.fit_transform(X[col])
    test[col]=encoder.fit_transform(test[col])


# In[ ]:


X.shape


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


X_=reduce_mem_usage(X)
test=reduce_mem_usage(test)
X=X.fillna(-999)
test=test.fillna(-999)
X.to_pickle('X.pkl')
y.to_pickle('y.pkl')
test.to_pickle('test.pkl')


# In[ ]:


test.shape


# In[ ]:


X.shape


# In[ ]:


n_fold = 5
folds = KFold(n_fold)


# In[ ]:


xgb_submission=pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')


# In[ ]:


xgb_submission['isFraud']=0


# In[ ]:


xgb_submission.head()


# In[ ]:


cv_scores=[]


# In[ ]:


# count=0
# for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
#     xgbclf = xgb.XGBClassifier(
#         n_estimators=1200,
#         max_depth=10,
#         learning_rate=0.04,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         missing=-999,
#         objective='binary:logistic',
#         tree_method='gpu_hist',  # THE MAGICAL PARAMETER
#         reg_alpha=0.1,
#         reg_lamdba=0.8,
#     )
    
#     X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
#     y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
#     xgbclf.fit(X_train,y_train)
#     pred=xgbclf.predict_proba(test)[:,1]
#     val=xgbclf.predict_proba(X_valid)[:,1]
#     score=roc_auc_score(y_valid, val)
#     print('Fold {}:ROC accuracy: {}'.format(fold_n,score))
#     cv_scores.append(score)
#     xgb_submission['isFraud'] = xgb_submission['isFraud']+pred/n_fold


    
    


# In[ ]:


np.mean(cv_scores)


# In[ ]:


xgb_submission.to_csv('submission.csv',index=False)

