#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import preprocessing
from wordcloud import WordCloud
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly import tools
from datetime import date
import lightgbm as lgb


# In[ ]:


train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')


# In[ ]:


train_transaction.head()


# In[ ]:


train_transaction.shape


# In[ ]:


import seaborn as sns
ax = train_transaction['isFraud'].value_counts().plot.bar(
    figsize = (12,6),
    color = 'mediumvioletred',
    fontsize = 16
)
ax.set_title('isFraud',fontsize = 20)
sns.despine(bottom = True, left = True)


# In[ ]:


ax = train_transaction['card4'].value_counts().plot.bar(
    figsize = (12,6),
    color = 'mediumvioletred',
    fontsize = 16
)
ax.set_title('Card Type',fontsize = 20)
sns.despine(bottom = True, left = True)


# In[ ]:


ax = train_transaction['card6'].value_counts().plot.bar(
    figsize = (12,6),
    color = 'mediumvioletred',
    fontsize = 16
)
ax.set_title('card6',fontsize = 20)
sns.despine(bottom = True, left = True)


# In[ ]:


ax = train_transaction['P_emaildomain'].value_counts().plot.bar(
    figsize = (12,6),
    color = 'mediumvioletred',
    fontsize = 16
)
ax.set_title('P_emaildomain',fontsize = 20)
sns.despine(bottom = True, left = True)


# In[ ]:


train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(train.shape)
print(test.shape)

y_train = train['isFraud'].copy()
del train_transaction, train_identity, test_transaction, test_identity

# Drop target, fill in NaNs
X_train = train.drop('isFraud', axis=1)
X_test = test.copy()

del train, test

X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)


# In[ ]:


# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))   


# In[ ]:


get_ipython().run_cell_magic('time', '', '# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage\n# WARNING! THIS CAN DAMAGE THE DATA \ndef reduce_mem_usage(df):\n    """ iterate through all the columns of a dataframe and modify the data type\n        to reduce memory usage.        \n    """\n    start_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage of dataframe is {:.2f} MB\'.format(start_mem))\n    \n    for col in df.columns:\n        col_type = df[col].dtype\n        \n        if col_type != object:\n            c_min = df[col].min()\n            c_max = df[col].max()\n            if str(col_type)[:3] == \'int\':\n                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n                    df[col] = df[col].astype(np.int8)\n                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n                    df[col] = df[col].astype(np.int16)\n                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n                    df[col] = df[col].astype(np.int32)\n                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n                    df[col] = df[col].astype(np.int64)  \n            else:\n                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n                    df[col] = df[col].astype(np.float16)\n                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n                    df[col] = df[col].astype(np.float32)\n                else:\n                    df[col] = df[col].astype(np.float64)\n        else:\n            df[col] = df[col].astype(\'category\')\n\n    end_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage after optimization is: {:.2f} MB\'.format(end_mem))\n    print(\'Decreased by {:.1f}%\'.format(100 * (start_mem - end_mem) / start_mem))\n    \n    return df\nX_train = reduce_mem_usage(X_train)\nX_test = reduce_mem_usage(X_test)')


# In[ ]:


X_train.shape,X_test.shape


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=18)
lgb_train = lgb.Dataset(data=x_train, label=y_train)
lgb_eval = lgb.Dataset(data=x_val, label=y_val)


# In[ ]:


params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 
          'learning_rate': 0.05, 'num_leaves': 200, 'num_iteration': 450, 'verbose': 0 ,
          'colsample_bytree':.8, 'subsample':.9, 'max_depth':18, 'reg_alpha':.1, 'reg_lambda':.1, 
          'min_split_gain':.01, 'min_child_weight':1}
model = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=60, verbose_eval=60)


# In[ ]:


preds = model.predict(X_test)


# In[ ]:


sample_submission['isFraud'] = preds
sample_submission.to_csv('deepak.csv')

