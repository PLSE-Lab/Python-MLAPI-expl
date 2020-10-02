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


import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, cross_val_score



# In[ ]:


train = pd.read_csv('../input/train.csv').set_index('ID_code')
print(train.shape)
train.head()


# In[ ]:


test = pd.read_csv('../input/test.csv').set_index('ID_code')
print(test.shape)
test.head()


# In[ ]:


RS = 2019


# In[ ]:


X_train,y_train = train.iloc[:,1:],train.iloc[:,0]


# In[ ]:


X_train.shape, y_train.shape,


# In[ ]:


skf = StratifiedKFold(n_splits=5,random_state=RS,shuffle=True,)


# In[ ]:


# lightgbm

model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=4, learning_rate=0.05, n_estimators=4500,
                      objective='binary',min_child_samples=2, subsample=.9, colsample_bytree=.05,
                     reg_alpha=0.0, reg_lambda=0.0, seed=RS,importance_type='gain',)


out = cross_val_score(model, X=X_train, y=y_train, scoring='roc_auc', cv=skf.split(X_train,y_train), verbose=1,)
print(out.mean(),out.std(),out.mean()-out.std())


# In[ ]:


# prediction for test set
model.fit(X_train,y_train)

pred_test = model.predict_proba(test)[:,1]

sub = pd.DataFrame({'ID_code':test.index,'target':pred_test})
sub.to_csv('my_lgb_sub.csv',index=False)

sub.head()


# In[ ]:




