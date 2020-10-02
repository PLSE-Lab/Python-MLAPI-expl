#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# In[6]:


import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import scikitplot.plotters as skplt
from sklearn.model_selection import StratifiedKFold


# In[7]:


dtype = {  'ip' : 'uint32',
           'app' : 'uint16',
           'device' : 'uint16',
           'os' : 'uint16',
           'channel' : 'uint8',
           'is_attributed' : 'uint8'}

usecol=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']


# In[8]:


df_train = pd.read_csv("../input/train.csv", dtype=dtype, infer_datetime_format=True, usecols=usecol, 
                               low_memory = True,nrows=20000000)
df_test = pd.read_csv("../input/test.csv")


# In[9]:


df_train.head()


# In[10]:


df_test.head()


# In[12]:


df_train['is_attributed'].value_counts()


# In[13]:


cols = ['ip', 'app', 'device', 'os', 'channel']
uniques_train = {col :df_train[col].nunique() for col in cols}
print('Train : Unique Values')
uniques_train


# In[14]:


cols = ['ip', 'app', 'device', 'os', 'channel']
uniques_test = {col :df_test[col].nunique() for col in cols}
print('Test : Unique Values')
uniques_test


# In[15]:


def mean_test_encoding(df_trn, df_tst, cols, target):    
   
    for col in cols:
        df_tst[col + '_mean_encoded'] = np.nan
        
    for col in cols:
        tr_mean = df_trn.groupby(col)[target].mean()
        mean = df_tst[col].map(tr_mean)
        df_tst[col + '_mean_encoded'] = mean

    prior = df_trn[target].mean()

    for col in cols:
        df_tst[col + '_mean_encoded'].fillna(prior, inplace = True) 
        
    return df_tst


# In[16]:


def mean_train_encoding(df, cols, target):
    y_tr = df[target].values
    skf = StratifiedKFold(5, shuffle = True, random_state=123)

    for col in cols:
        df[col + '_mean_encoded'] = np.nan

    for trn_ind , val_ind in skf.split(df,y_tr):
        x_tr, x_val = df.iloc[trn_ind], df.iloc[val_ind]

        for col in cols:
            tr_mean = x_tr.groupby(col)[target].mean()
            mean = x_val[col].map(tr_mean)
            df[col + '_mean_encoded'].iloc[val_ind] = mean

    prior = df[target].mean()

    for col in cols:
        df[col + '_mean_encoded'].fillna(prior, inplace = True) 
        
    return df


# In[17]:


y = df_train['is_attributed']
cols = ['app', 'channel']
target = 'is_attributed'
df_train = mean_train_encoding(df_train, cols, target)
df_test  = mean_test_encoding(df_train, df_test, cols, target)


# In[18]:


df_train.drop(['click_time','is_attributed'], axis = 1, inplace = True)   
df_test.drop(['click_time','click_id'], axis = 1, inplace = True)   


# In[19]:


def print_score(m, df, y):
    print('Accuracy: [Train , Val]')
    res =  [m.score(df, y)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)           
    print(res)
    
    print('Train Confusion Matrix')
    df_train_proba = m.predict_proba(df)
    df_train_pred_indices = np.argmax(df_train_proba, axis=1)
    classes_train = np.unique(y)
    preds_train = classes_train[df_train_pred_indices]    
    skplt.plot_confusion_matrix(y, preds_train)      


# In[36]:


print(df_test.head())
df_train.head()


# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
x1,x2,y1,y2=train_test_split(df_train,y)
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
alg1 = DecisionTreeClassifier()
alg2 = ExtraTreeClassifier()


# In[43]:


alg1.fit(x1,y1)
alg2.fit(x1,y1)
y_pred1 = alg1.predict_proba(x2)[:,1]
y_pred2 = alg2.predict_proba(x2)[:,1]


# In[44]:


print("DecisionTreeClassifier :",roc_auc_score(y2,y_pred1))
print("ExtraTreeClassifier :",roc_auc_score(y2,y_pred2))


# In[48]:


test_submission2 = pd.read_csv("../input/sample_submission.csv")
test_submission1 = pd.read_csv("../input/sample_submission.csv")
test_submission1.head()


# In[24]:


# clf = RandomForestClassifier(n_estimators=12, max_depth=6, min_samples_leaf=100, max_features=0.5, bootstrap=False, n_jobs=-1, random_state=123)
# %time clf.fit(df_train, y)
# print_score(clf, df_train, y)


# In[46]:


cols = df_train.columns
Imp = clf.feature_importances_
feature_imp_dict = {}
for i in range(len(cols)):
    feature_imp_dict[cols[i]] = Imp[i]
print(feature_imp_dict)


# In[52]:


test_submission2.shape


# In[49]:


# #y_pred1 = clf.predict_proba(df_test)
# test_submission1['is_attributed'] = y_pred1
# #y_pred2 = clf.predict_proba(df_test)
# test_submission2['is_attributed'] = y_pred1

# test_submission1.head()


# In[ ]:


# test_submission.to_csv('submission_rf_.csv', index=False)


# In[ ]:




