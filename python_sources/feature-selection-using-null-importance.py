#!/usr/bin/env python
# coding: utf-8

# ## Feature Selection IEEE Fraud

# This kernel is highly inspired by Oliver's kernel in the Home Credit Default Risk competition : https://www.kaggle.com/ogrellier/feature-selection-with-null-importances

# To get an idea about which features might be important we do the following :
# 1. I fit usual XGB model to the data as it is and get the actual importance of each feature.
# 2. Then I shuffle only the 'isFraud' column and fit XGB model and get the importance of each faeture. We call this as null importance. I did this 50 times. 
# 3. Plot the actual importance and all the null importances.
# 4. Define some way to get score for each feature.

# I have used the same ways which Oliver used in his kernel to score features. But surprisingly, all features are scoring negative. But according to me, if a feature is really important then the distance between the mean of the null importances and actual importance must be more (I know 'more' is very subjective but we can try different thresholds) as compared to that of unimportant features. I have done this in another kernel because of memory issues : https://www.kaggle.com/virajbagal/distance-between-importances  I have used different thresholds and then fit XGB model  in that kernel. 
# 
# So, accoring to me the hypothesis must be:
# 1. The distance between mean of null importances and the actual importance must be more compared to that of unimportant features.
# 2. The null importance of the important features must have high variance. 
# 
# 
# **Please comment if I am wrong anywhere or whatever your thoughts are. 
# Thank you for reading my kernel. All the best for the competitoin. **

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
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import os
print(os.listdir("../input"))
seed=5
# Any results you write to the current directory are saved as output.


# In[ ]:


train_identity=pd.read_csv('../input/train_identity.csv')
# test_identity=pd.read_csv('../input/test_identity.csv')
train_transaction=pd.read_csv('../input/train_transaction.csv')
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


train.info()


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


def get_feature_importance(data,target,shuffle=False):
    y=target.copy()
    if shuffle:
        y=target.copy().sample(frac=1).reset_index(drop=True)
    
    xgb_params=dict(n_estimators=1000,
                verbosity=0,
                tree_method='gpu_hist',
               colsample_bytree=0.8,
               subsample=0.8,
               learning_rate=0.05,
               max_depth=5)
    
    clf=XGBClassifier(**xgb_params)
    clf.fit(train,y)
    
    feat_imp=pd.DataFrame()
    feat_imp['features']=train.columns
    feat_imp['importances']=clf.feature_importances_
    feat_imp['train_score']=roc_auc_score(y,clf.predict(train))
    del clf
    
    return feat_imp


# ## Actual Importance

# In[ ]:


np.random.seed(5)
actual_feat_imp=get_feature_importance(train,target,shuffle=False)


# ## Null Importances

# In[ ]:


import time


null_feat_imps=pd.DataFrame()
start=time.time()
for i in range(50):
    start1=time.time()
    null_imp=get_feature_importance(train,target,shuffle=True)
    null_imp['round']=i+1
    null_feat_imps=pd.concat([null_feat_imps,null_imp],axis=0)
    del null_imp
    end1=time.time()
    epoch_time=(end1-start1)/60
    print(f'Round {i+1} completed in {epoch_time} mins')
    print('-'*100)

    
end=time.time()
total_time=(end-start)/60
print(f'Total time taken : {total_time} mins')


# In[ ]:


def show_null_actual(feature):
    plt.figure(figsize=(10,5))
    a=plt.hist(null_feat_imps[null_feat_imps['features']==feature]['importances'],label='Null Importance')
    plt.vlines(x=actual_feat_imp[actual_feat_imp['features']==feature]['importances'],ymin=0,ymax=np.max(a[0]),color='r',linewidth=10,label='Real Target')
    plt.legend(loc='best')
    plt.title(f'Acutal Importance vs Null Importance of {feature}')
    plt.show()


# In[ ]:


actual_feat_imp.to_csv('Actual_imp.csv',index=False)
null_feat_imps.to_csv('Null_imp.csv',index=False)


# # Plotting null and actual importance.

# You can try this out for all other features. 

# In[ ]:


show_null_actual('V1')


# ## Feature Scores

# **1. Using the log of ratio of actual importance and 75th percentile of null importance. The logic here is, in case of important features, the actual importance must be substantially greater than the 75th percentile of null importances of that feature. So, the log of the ratio is expected to be more positive for more important features. But here, for every feature the score obtained is negative. That means for every feature, actual importance is lesser than (1 + the 75th percentile of null importances). This was shocking to me . **

# In[ ]:


feature_scores=[]

for feature in train.columns:
    null_imp=null_feat_imps[null_feat_imps['features']==feature]['importances'].values
    actual_imp=actual_feat_imp[actual_feat_imp['features']==feature]['importances'].values
    score=np.log((1e-10 + actual_imp/(1+np.percentile(null_imp,75))))[0]
    feature_scores.append((feature,score))


# In[ ]:


feature_score_df=pd.DataFrame(feature_scores,columns=['Feature','Score']).sort_values('Score',ascending=False).reset_index(drop=True)

plt.figure(figsize=(10,10))
sns.barplot(x='Score',y='Feature',data=feature_score_df.iloc[:50,:])
plt.title('Scores of all features')


# **2. Here we just use the counts of how many null importances are lesser than the actual importances. For more important features, we expect the score to be more. **

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

