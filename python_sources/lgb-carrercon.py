#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import itertools
import xgboost as xgb
import time
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *


# In[20]:


X_train = pd.read_csv("../input/X_train.csv")
X_test = pd.read_csv("../input/X_test.csv")
y_train = pd.read_csv("../input/y_train.csv")
sub = pd.read_csv("../input/sample_submission.csv")


# In[3]:


X_train.head()


# In[4]:


plt.figure(figsize=(15, 5))
sns.countplot(y_train['surface'])
plt.title('Target distribution', size=15)
plt.show()


# In[5]:


# https://www.kaggle.com/jesucristo/1-smart-robots-complete-notebook-0-73/notebook
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(X_train.iloc[:,3:].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# 

# Here we can discover strong correlation between angular_velocity_Z and angular_velocity_Y.
# 

# ### Feature Engineer

# In[21]:


#https://www.kaggle.com/prashantkikani/help-humanity-by-helping-robots

def fe(data):
    
    df = pd.DataFrame()
    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 +
                             data['angular_velocity_Z']**2)** 0.5
    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 +
                             data['linear_acceleration_Z'])**0.5
    data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 +
                             data['orientation_Z'])**0.5
   
    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']
    
    for col in data.columns:
        if col in ['row_id','series_id','measurement_number']:
            continue
        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()
        df[col + '_median'] = data.groupby(['series_id'])[col].median()
        df[col + '_max'] = data.groupby(['series_id'])[col].max()
        df[col + '_min'] = data.groupby(['series_id'])[col].min()
        df[col + '_std'] = data.groupby(['series_id'])[col].std()
        df[col + '_range'] = df[col + '_max'] - df[col + '_min']
        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']
        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))
        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))
        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2
    return df


# In[22]:


get_ipython().run_cell_magic('time', '', 'X_train = fe(X_train)\nX_test = fe(X_test)\nprint(X_train.shape)')


# In[23]:


le = LabelEncoder()
y_train['surface'] = le.fit_transform(y_train['surface'])


# In[24]:


X_train.fillna(0, inplace = True)
X_test.fillna(0, inplace = True)
X_train.replace(-np.inf, 0, inplace = True)
X_train.replace(np.inf, 0, inplace = True)
X_test.replace(-np.inf, 0, inplace = True)
X_test.replace(np.inf, 0, inplace = True)


# In[25]:


#X_train.drop('row_index', axis=1)
#X_test.drop(['row_index'], axis=1)
data = X_train
target = y_train["surface"]
print(X_train.shape)
print(X_test.shape)


# In[26]:


from sklearn.model_selection import GridSearchCV,StratifiedKFold, train_test_split
kfold = StratifiedKFold(n_splits=5)
from sklearn.metrics import roc_auc_score,roc_curve,auc
train_x,val_x,train_y,val_y = train_test_split(data, target, test_size = 0.10, random_state=14)
train_x.shape,val_x.shape,train_y.shape,val_y.shape


# In[27]:


import lightgbm
train_data = lightgbm.Dataset(train_x, label=train_y)
test_data = lightgbm.Dataset(val_x, label=val_y)


# In[28]:


#used tuned parameters after applying gridsearch
para={'boosting_type': 'gbdt',
 'colsample_bytree': 0.85,
 'learning_rate': 0.1,
 'max_bin': 512,
 'max_depth': -1,
 'metric': 'multi_error',
 'min_child_samples': 8,
 'min_child_weight': 1,
 'min_split_gain': 0.5,
 'nthread': 3,
 'num_class': 9,
 'num_leaves': 31,
 'objective': 'multiclass',
 'reg_alpha': 0.8,
 'reg_lambda': 1.2,
 'scale_pos_weight': 1,
 'subsample': 0.7,
 'subsample_for_bin': 200,
 'subsample_freq': 1}

model = lightgbm.train(para,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=50,
                      )


# In[34]:


y_pred = model.predict(X_test)


# In[35]:


class_prediction=pd.DataFrame(y_pred).idxmax(axis=1) 


# In[38]:


submission1 = pd.DataFrame({
        "series_id": X_test.index,
        "surface": class_prediction,
        
    })
submission1.surface.value_counts()


# In[41]:


submission1.surface=le.inverse_transform(submission1.surface)


# In[43]:


submission1.head()


# In[44]:


submission1.to_csv('submissionlgb.csv', index=False)


# In[ ]:




