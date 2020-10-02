#!/usr/bin/env python
# coding: utf-8

# ### This kernel is based on the excellent lectures by [@kazanova](https://www.kaggle.com/kazanova) from the [Competitive Data Science](https://www.coursera.org/learn/competitive-data-science) course on Coursera
# We'll create a simple meta model by stacking a linear model and a RF model

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import os
import math

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train_df.head()


# The evaluation criteria is RMSE of log of Sales Price. So first, let's change the target variable to log

# In[ ]:


train_df['SalePrice'] = np.log(train_df['SalePrice'])


# ### Handling categorical data

# In[ ]:


import fastai_structured as fs
fs.train_cats(train_df)
fs.apply_cats(test_df, train_df)


# In[ ]:


nas = {}
df_trn, y_trn, nas = fs.proc_df(train_df, 'SalePrice', na_dict=nas)   ## Avoid creating NA columns as total cols may not match later
df_test, _, _ = fs.proc_df(test_df, na_dict=nas)
df_trn.head()


# ### Defining function to calculate the evaluation metric

# In[ ]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(train_X), train_y), rmse(m.predict(val_X), val_y),     ## RMSE of log of prices
                m.score(train_X, train_y), m.score(val_X, val_y)]
    #if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# ### Splitting into training & validation sets

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(df_trn, y_trn, test_size=0.5, random_state=42)


# ### Specify & fit models on training set

# In[ ]:


model1 = linear_model.LinearRegression()
model2 = RandomForestRegressor()


# In[ ]:


model1.fit(train_X, train_y)
model2.fit(train_X, train_y)


# In[ ]:


print_score(model1)


# In[ ]:


print_score(model2)


# ### Make predictions on validation AND test set

# In[ ]:


preds1 = model1.predict(val_X)
preds2 = model2.predict(val_X)


# In[ ]:


test_preds1 = model1.predict(df_test)
test_preds2 = model2.predict(df_test)


# ### Form a new dataset for validation & test by stacking the predictions

# In[ ]:


stacked_predictions = np.column_stack((preds1, preds2))
stacked_test_predictions = np.column_stack((test_preds1, test_preds2))


# ### Specify meta model & fit it on stacked validation set predictions

# In[ ]:


meta_model = linear_model.LinearRegression()


# In[ ]:


meta_model.fit(stacked_predictions, val_y)


# ### Use meta model to make preditions on the stacked predictions of test set

# In[ ]:


final_predictions = meta_model.predict(stacked_test_predictions)


# ### Submit predictions

# In[ ]:


submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission.head()


# In[ ]:


submission['SalePrice'] = np.exp(final_predictions)   ## Convert log back 
submission.to_csv('stacking_example.csv', index=False)

