#!/usr/bin/env python
# coding: utf-8

# Credits to the Experts:
# 
#     inversion and Walter Reade's https://www.kaggle.com/inversion/ieee-simple-xgboost

# **This notebook is a part of the project I did for the paper 1xx333**

# In[ ]:


import numpy as np 
import pandas as pd 
from shutil import copyfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
copyfile(src = "../input/158333a1/a1_wrangle.py", dst = "../working/a1_wrangle.py")
import a1_wrangle as a1w


# ### Data Wrangling 

# In[ ]:


X_train, y_train, X_test, submission = a1w.quick_wrangle()


# In[ ]:


y_train.head()


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# ### Fit and prediction

# In[ ]:


rf = RandomForestClassifier(n_estimators=900, max_depth=2, random_state=0, n_jobs=4)
ab = AdaBoostClassifier(n_estimators=500, random_state=0)
xg = xgb.XGBClassifier(n_estimators=500, n_jobs=4, max_depth=11, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, missing=-999)

ec = VotingClassifier(estimators=[('rf', rf), ('ab', ab), ('xg', xg)], voting='hard')


# In[ ]:


ec.fit(X_train, y_train)


# In[ ]:


submission['isFraud'] = ec.predict(X_test)


# In[ ]:


submission.to_csv('majority_voting_3_in_1.csv')

