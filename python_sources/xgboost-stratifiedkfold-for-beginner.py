#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
import gc
gc.enable()


# Load Data

# In[ ]:


df_train = pd.read_csv('../input/train.csv')


# In[ ]:


df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_sub =  pd.read_csv('../input/sample_submission.csv')


# Basic check

# In[ ]:


df_train.shape,df_test.shape


# In[ ]:


df_train.head()


# Hmm.... ID_code is just a simple index, maybe we should exclude it?

# In[ ]:


df_test.head()


# OK, it appears in test data as well. We will remove it then...
# Now we need to check if any missing data here...

# In[ ]:


total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()


# Good! nothing is missed so we can skip the fillNA part. :D

# Next, let's have a look at correlation of columns and our target. This is very first and important for feature engineering.

# In[ ]:


def plotCorr(df,col,K=10,ascending = False):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    K=10
    corrmat = df.corr()
    cols = corrmat.nsmallest(K,'target')['target'].index if ascending else corrmat.nlargest(K,'target')['target'].index
    cm = np.corrcoef(df[cols].values.T)
    sns.heatmap(cm,cbar=True,annot=True,yticklabels=cols.values,xticklabels=cols.values,annot_kws={'size':10})
    plt.show()


# In[ ]:


plotCorr(df_train,'target')


# Huh...the higest correlation rate is only 0.067!... Thats a bit low..

# Can we simply add them up?

# In[ ]:


def featureEngineering(df,cols,col_name):
    df[col_name] = 0
    for c in cols:
        df[col_name] += df[c]
    return df


# In[ ]:


corr = df_train.corr()
cols = corr.nlargest(12,'target').index.values[1:]


# In[ ]:


df_train = featureEngineering(df_train,cols, 'posCols')


# In[ ]:


plotCorr(df_train,'target')


# Now the highest corr rate is 0.16.. It might help or might not... 
# Let's just do the same thing to test data

# In[ ]:


df_test = featureEngineering(df_test,cols, 'posCols')


# My machine is slow.. so I just split data into three parts

# In[ ]:


val = np.zeros(df_train.shape[0])
pred = np.zeros(df_test.shape[0])
x = df_train.drop(['ID_code','target'],axis=1).values
y = df_train['target'].values
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=1234)


# In[ ]:


model_xgb =  xgb.XGBClassifier(max_depth=2,
                              colsample_bytree=0.7,
                              n_estimators=20000,
                              scale_pos_weight = 9,
                              learning_rate=0.02,
                              objective='binary:logistic', 
                              verbosity =1,
                              eval_metric  = 'auc',
                              tree_method='gpu_hist',
                              n_jobs=-1)


# In[ ]:


for fold_index, (train_index,val_index) in enumerate(folds.split(x,y)):
    print('Batch {} started...'.format(fold_index))
    gc.collect()
    bst = model_xgb.fit(x[train_index],y[train_index],
              eval_set = [(x[val_index],y[val_index])],
              early_stopping_rounds=200,
              verbose= 200, 
              eval_metric ='auc'
              )
    val[val_index] = model_xgb.predict_proba(x[val_index])[:,1]
    print('auc of this val set is {}'.format(roc_auc_score(y[val_index],val[val_index])))
    pred += model_xgb.predict_proba(df_test.drop(['ID_code'],axis=1).values)[:,1]/folds.n_splits


# In[ ]:


df_sub.target = pred


# In[ ]:


df_sub.to_csv('r_prob.csv',index=False)


# In[ ]:




