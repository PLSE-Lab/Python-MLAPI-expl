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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv("../input/train_V2.csv")
test=pd.read_csv("../input/test_V2.csv")
sample_submission=pd.read_csv("../input/sample_submission_V2.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


print('train shape before dropping null=',train.shape)
train=train.dropna()
print('train shape after dropping null=',train.shape)


# In[ ]:


y=train['winPlacePerc']
train=train.drop(columns=['winPlacePerc','Id','groupId','matchId'],axis=1)
test=test.drop(columns=['Id','groupId','matchId'],axis=1)


# In[ ]:


train=pd.get_dummies(train)
test=pd.get_dummies(test)
print('train shape:',train.shape,'test shape:',test.shape)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(train,y,test_size=0.2)


# In[ ]:


from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


#xgb=XGBRegressor(max_depth=18,ganmma=0.5,n_jobs=20)
#lgb=LGBMRegressor()
cb=CatBoostRegressor(iterations=2000,depth=7)
#rf=RandomForestRegressor()


# In[ ]:


#xgb.fit(x_train,y_train)
cb.fit(x_train,y_train)
#rf.fit(x_train,y_train)


# In[ ]:


y_pred=cb.predict(x_test)


# In[ ]:


#y_pred=y_pred/(y_pred.max()-y_pred.min())


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


print('r2 score is:',r2_score(y_test,y_pred))


# In[ ]:


y_pred_final=cb.predict(test)


# In[ ]:


sample_submission['winPlacePerc']=y_pred_final


# In[ ]:


sample_submission.head(2)


# In[ ]:


sample_submission.to_csv('submission_pubg.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:




