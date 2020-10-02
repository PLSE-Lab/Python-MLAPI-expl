#!/usr/bin/env python
# coding: utf-8

# # ML Challenge 2019
# ### starter code

# ### Imports

# In[ ]:


import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


# ### Load Data

# In[ ]:


train = pd.read_csv('../input/kml2019/train.csv', encoding='cp949')
test  = pd.read_csv('../input/kml2019/test.csv', encoding='cp949')


# In[ ]:


train.head()


# In[ ]:


test.head()


# ### Make Features & Split Data

# In[ ]:


df = train.pivot_table(index='item_id', columns='month', values='count', fill_value=0, aggfunc=np.sum).clip(0,20).reset_index()
df = pd.merge(df, train.drop_duplicates('item_id')[['item_id','category']], on='item_id', how='left')
df


# In[ ]:


TIME_STEP = 6

def make_seq_data(train, test, timestep, intersect):
    df = train.pivot_table(index='item_id', columns='month', values='count', 
                           fill_value=0, aggfunc=np.sum).clip(0,20).reset_index()
    df = pd.merge(df, train.drop_duplicates('item_id')[['item_id','category']], 
                  on='item_id', how='left')    
    if intersect: 
        no_items = set(test.item_id).intersection(set(df.item_id))
        df = df.query('item_id in @no_items')
    X = []
    y = []
    X_cat = []
    for i in range(1, 12-timestep):
        X.append(df.loc[:, range(i,i+timestep)])
        y.append(df[i+timestep])
        X_cat.append(df['category'])
    X_train = np.vstack(X)
    y_train = np.hstack(y)
    X_train_cat = np.hstack(X_cat)
    ind = list(range(12-timestep,12))
    ind.append('item_id')
    X_test = pd.merge(test, df.loc[:, ind], on='item_id', how='left').fillna(0).drop(['item_id','category'], axis=1)
    X_test_cat = pd.merge(test, df.loc[:, ind], on='item_id', how='left').fillna(0)['category']
    return X_train, X_test, y_train, X_train_cat, X_test_cat

X_train, X_test, y_train, X_train_cat, X_test_cat = make_seq_data(train, test, TIME_STEP, True)
X_train.shape, X_test.shape, y_train.shape, X_train_cat.shape, X_test_cat.shape


# ### Preprocess Data

# In[ ]:


# Encoding for categorical features
le = LabelEncoder()
X_train_cat = le.fit_transform(X_train_cat)
X_test_cat = le.transform(X_test_cat)


# In[ ]:


# Concatenate numerical features and categorical features
X_train = np.hstack([X_train, X_train_cat.reshape(-1,1)])
X_test = np.hstack([X_test, X_test_cat.reshape(-1,1)])


# ### Build XGB Models

# In[ ]:


model = XGBRegressor()
model.fit(X_train, y_train, eval_metric='rmse')


# ### Make Submissions

# In[ ]:


y_test = model.predict(X_test).clip(0, 20)
submission = pd.DataFrame({
    "item_id": test.item_id, 
    "item_cnt_month": y_test
})
t = pd.Timestamp.now()
fname = f"xgb_submission_{t.month:02}{t.day:02}_{t.hour:02}{t.minute:02}.csv"
submission.to_csv(fname, index=False)


# # End
