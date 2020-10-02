#!/usr/bin/env python
# coding: utf-8

# # Ben's baseline model with Log Transform
# Last updated: 04/01/2018
# ***

# In[1]:


import numpy as np
import pandas as pd
from datetime import datetime
import os
import gc
import re
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
pal = sns.color_palette()


# In[2]:


FILE_DIR = '../input/hawaiiml-data'

for f in os.listdir(FILE_DIR):
    print('{0:<30}{1:0.2f}MB'.format(f, 1e-6*os.path.getsize(f'{FILE_DIR}/{f}')))


# In[26]:


train = pd.read_csv(f'{FILE_DIR}/train.csv', encoding='ISO-8859-1')
test = pd.read_csv(f'{FILE_DIR}/test.csv', encoding='ISO-8859-1')
submission = pd.read_csv(f'{FILE_DIR}/sample_submission.csv', encoding='ISO-8859-1')


# In[27]:


train.head()


# In[28]:


train = train[train['unit_price'] > 0]


# In[33]:



train["log1p_unit_price"] = np.log1p(train.unit_price)
test["log1p_unit_price"] = np.log1p(test.unit_price)
train.head()


# In[ ]:


train.columns


# In[ ]:


train.describe()


# Doing a baseline model based on this tutorial: https://www.kaggle.com/dansbecker/how-models-work

# In[ ]:





# In[39]:


from sklearn.model_selection import train_test_split

y = np.log1p(train.quantity)
X = train
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

predictors_log_price = ['customer_id', 'log1p_unit_price']
predictors = ['customer_id', 'unit_price']


# In[44]:


from sklearn.tree import DecisionTreeRegressor
model_log_price = DecisionTreeRegressor()
model = DecisionTreeRegressor()

model.fit(train_X[predictors],train_y)
model_log_price.fit(train_X[predictors_log_price],train_y)

# get predicted prices on validation data
val_predictions = model.predict(val_X[predictors])
val_predictions_log_price = model_log_price.predict(val_X[predictors_log_price])


# In[45]:


# RMLSE error calculator - Model 1
import numpy as np
np.random.seed(0)

def rmsle(val_y, train_y):
    return np.sqrt(np.mean((np.log1p(val_y) - np.log1p(train_y))**2))


# In[48]:


from sklearn.metrics import mean_absolute_error

exp_y = np.expm1(val_y)
exp_predictions = np.expm1(val_predictions)
exp_predictions_log_price = np.expm1(val_predictions_log_price)

print(mean_absolute_error(exp_y, exp_predictions))
print(mean_absolute_error(exp_y, exp_predictions_log_price))

print(f'RMSLE: {rmsle(exp_y, exp_predictions):0.5f}' )
print(f'RMSLE: {rmsle(exp_y, exp_predictions_log_price):0.5f}' )


# In[52]:


final_model = DecisionTreeRegressor()
final_model.fit(train[predictors], y)
preds = np.expm1(final_model.predict(test[predictors]))


# In[53]:


my_submission = pd.DataFrame({'id': test.id, 'quantity': preds})
my_submission.to_csv('submission.csv', index=False)

