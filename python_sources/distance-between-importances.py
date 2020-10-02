#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


null_feat_imps=pd.read_csv('../input/feature-selection-ieee/Null_imp.csv')
actual_feat_imp=pd.read_csv('../input/feature-selection-ieee/Actual_imp.csv')


# In[ ]:


train_identity=pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
# test_identity=pd.read_csv('../input/test_identity.csv')
train_transaction=pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
# test_transaction=pd.read_csv('../input/test_transaction.csv')


# In[ ]:


train=pd.merge(train_transaction,train_identity,how='left',on='TransactionID')
# test=pd.merge(test_transaction,test_identity,how='left',on='TransactionID')


# In[ ]:


del train_identity,train_transaction


# In[ ]:


target=train['isFraud']
train=train.drop(['isFraud','TransactionID'],axis=1)
# test=test.drop('TransactionID',axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split

train,val,target,val_y=train_test_split(train,target,test_size=0.5,random_state=5,stratify=target)


# In[ ]:


del val,val_y


# In[ ]:


train=train.fillna(-999)
# test=test.fillna(-999)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

cat_cols=[col for col in train.columns if train[col].dtype=='object']
for col in cat_cols:
    le=LabelEncoder()
    le.fit(list(train[col].values))
    train[col]=le.transform(list(train[col].values))
#     test[col]=le.transform(list(test[col].values))


# In[ ]:


get_ipython().run_cell_magic('time', '', '# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage\n# WARNING! THIS CAN DAMAGE THE DATA \ndef reduce_mem_usage(df):\n    """ iterate through all the columns of a dataframe and modify the data type\n        to reduce memory usage.        \n    """\n    start_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage of dataframe is {:.2f} MB\'.format(start_mem))\n    \n    for col in df.columns:\n        col_type = df[col].dtype\n        \n        if col_type != object:\n            c_min = df[col].min()\n            c_max = df[col].max()\n            if str(col_type)[:3] == \'int\':\n                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n                    df[col] = df[col].astype(np.int8)\n                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n                    df[col] = df[col].astype(np.int16)\n                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n                    df[col] = df[col].astype(np.int32)\n                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n                    df[col] = df[col].astype(np.int64)  \n            else:\n                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n                    df[col] = df[col].astype(np.float16)\n                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n                    df[col] = df[col].astype(np.float32)\n                else:\n                    df[col] = df[col].astype(np.float64)\n        else:\n            df[col] = df[col].astype(\'category\')\n\n    end_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage after optimization is: {:.2f} MB\'.format(end_mem))\n    print(\'Decreased by {:.1f}%\'.format(100 * (start_mem - end_mem) / start_mem))\n    \n    return df')


# In[ ]:


train=reduce_mem_usage(train)
# test=reduce_mem_usage(test)


# In[ ]:


correlation_scores=[]

for feature in train.columns:
    null_imp=null_feat_imps[null_feat_imps['features']==feature]['importances'].values
    actual_imp=actual_feat_imp[actual_feat_imp['features']==feature]['importances'].values
    corr_score=100*(null_imp < actual_imp).sum()/null_imp.size
    correlation_scores.append((feature,corr_score))


# In[ ]:


correlation_df=pd.DataFrame(correlation_scores,columns=['Feature','Score']).sort_values('Score',ascending=False).reset_index(drop=True)
plt.figure(figsize=(10,10))
sns.barplot(x='Score',y='Feature',data=correlation_df.iloc[:50,:])
plt.title('Scores of all features')
plt.show()


# In[ ]:


feature_distdf=[]

for feature in actual_feat_imp['features'].values:
    dist=np.abs(actual_feat_imp[actual_feat_imp['features']==feature]['importances'].values - np.mean(null_feat_imps[null_feat_imps['features']==feature]['importances'].values))
    feature_distdf.append((feature,dist[0]))


# In[ ]:


feature_df=pd.DataFrame(feature_distdf,columns=['Feature','Distance_from_Mean']).sort_values('Distance_from_Mean',ascending=False).reset_index(drop=True)


# In[ ]:


plt.figure(figsize=(10,10))
sns.barplot(x='Distance_from_Mean',y='Feature',data=feature_df[:50])
plt.title('Distance of Actual Importance from Mean of Null Importances')


# In[ ]:


def get_selection_score(data=train,target=target):
    xgb_params=dict(
                    verbosity=0,
                    tree_method='gpu_hist',
                    colsample_bytree=0.8,
               subsample=0.8,
               learning_rate=0.05,
               max_depth=5,
                   objective='binary:logistic',
                   metric='auc')
    
    train_d=xgb.DMatrix(data,label=target)
    result=xgb.cv(xgb_params,train_d,num_boost_round=1000,nfold=3,stratified=True,shuffle=True,early_stopping_rounds=50,verbose_eval=0,
                 seed=5,metrics=('auc'))
    
    
    return (list(result['test-auc-mean'].values)[-1], list(result['test-auc-std'].values)[-1])


# In[ ]:


# for threshold in [0, 10, 20, 30 , 40, 50 ,60 , 70, 80 , 90, 95, 99]:  
#     print('Result for threshold ',threshold)
#     worthy_features=[feature for feature in correlation_df['Feature'].values if correlation_df.loc[correlation_df['Feature']==feature,'Score'].values>=threshold]
#     score=get_selection_score(train[worthy_features],target)
#     print('Test AUC Mean :',score[0])
#     print('Test AUC Std:',score[1])
#     del score


# In[ ]:


for threshold in [0,0.0001,0.0003,0.0005,0.0008,0.001,0.003,0.005,0.008,0.01,0.015,0.025,0.05]:  
    print('Result for threshold ',threshold)
    worthy_features=[feature for feature in feature_df['Feature'].values if feature_df.loc[feature_df['Feature']==feature,'Distance_from_Mean'].values>=threshold]
    score=get_selection_score(train[worthy_features],target)
    print('Test AUC Mean :',score[0])
    print('Test AUC Std:',score[1])
    del score

