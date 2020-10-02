#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool, cv
import os
import math
from sklearn.metrics import roc_auc_score
import fastai_structured as fs

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_df = pd.read_csv("/kaggle/input/ps-reducing-memory-size-for-ieee/train.csv")
test_df = pd.read_csv("/kaggle/input/ps-reducing-memory-size-for-ieee/test.csv")
train_df.head()


# ### Handling categorical data

# In[ ]:


fs.train_cats(train_df)
fs.apply_cats(test_df, train_df)


# In[ ]:


nas = {}
df_trn, y_trn, nas = fs.proc_df(train_df, 'isFraud', na_dict=nas)   ## Avoid creating NA columns as total cols may not match later
df_test, _, _ = fs.proc_df(test_df, na_dict=nas)
df_trn.head()


# ### Defining function to calculate the evaluation metric

# In[ ]:


def auc(x,y): 
    return roc_auc_score(x,y)
def print_score(m):
    res = [auc(m.predict(train_X), train_y), auc(m.predict(val_X), val_y)]
    print(res)


# ### Splitting into training & validation sets

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(df_trn, y_trn, test_size=0.7, random_state=42)


# In[ ]:


del train_df, test_df


# In[ ]:


del df_trn, y_trn


# ### Specify & fit models on training set

# In[ ]:


cat_params = {
    'loss_function': 'Logloss',
    'custom_loss':['AUC'],
    'logging_level':'Silent',
    'task_type' : 'CPU',
    'early_stopping_rounds' : 100
}

model1 = CatBoostClassifier(**cat_params)


# In[ ]:


model2 = RandomForestClassifier(n_estimators = 100, min_samples_leaf=10, max_features=0.5,
                           max_depth = 4, n_jobs=-1)


# In[ ]:


Catfeats = ['ProductCD'] +            ["card"+f"{i+1}" for i in range(6)] +            ["addr"+f"{i+1}" for i in range(2)] +            ["P_emaildomain", "R_emaildomain"] +            ["M"+f"{i+1}" for i in range(9)] +            ["DeviceType", "DeviceInfo"] +            ["id_"+f"{i}" for i in range(12, 39)]

# removing columns dropped earlier when we weeded out the empty columns

#Catfeats = list(set(Catfeats)- set(cols_dropped))

model1.fit(train_X, train_y, cat_features = Catfeats)


# In[ ]:


model2.fit(train_X, train_y)


# In[ ]:


print_score(model1)


# In[ ]:


print_score(model2)


# ### Make predictions on validation AND test set

# In[ ]:


preds1 = model1.predict(val_X)


# In[ ]:


preds2 = model2.predict(val_X)


# In[ ]:


test_preds1 = model1.predict(df_test)


# In[ ]:


test_preds2 = model2.predict(df_test)


# ### Form a new dataset for validation & test by stacking the predictions

# In[ ]:


stacked_predictions = np.column_stack((preds1, preds2))
stacked_test_predictions = np.column_stack((test_preds1, test_preds2))


# ### Specify meta model & fit it on stacked validation set predictions

# In[ ]:


meta_model = linear_model.LogisticRegression()


# In[ ]:


meta_model.fit(stacked_predictions, val_y)


# ### Use meta model to make preditions on the stacked predictions of test set

# In[ ]:


final_predictions = meta_model.predict(stacked_test_predictions)


# ### Submit predictions

# In[ ]:


submission = pd.read_csv('/kaggle/input/ps-reducing-memory-size-for-ieee/sample_submission.csv')
submission.head()


# In[ ]:


submission['isFraud'] = final_predictions 
submission.to_csv('stacking_v1.csv', index=False)

