#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import h2o
import pandas as pd
import datetime as dt


# In[ ]:


from h2o.automl import H2OAutoML
h2o.init()


# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


df.describe()


# In[ ]:


df.head()


# In[ ]:


df['date'] = pd.to_datetime(df['date'])
print(df['date'].dtype)


# In[ ]:


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday
df['quoter'] = df['year'] * 4 + divmod(df['month'], 3)[0] - 8051
df.head()


# In[ ]:


df['item_store_month_sales'] = df.groupby(['item', 'store', 'month'])['sales'].transform('mean')
df['store_item_weekday_sales'] = df.groupby(['store', 'item', 'weekday'])['sales'].transform('mean')
df['round_item_store_month_sales'] = round(df['item_store_month_sales'])
df['round_store_item_weekday_sales'] = round(df['store_item_weekday_sales'])
df.head()


# In[ ]:


df_select = df[['sales',                 'month',                 'quoter',                 'item_store_month_sales',                 'store_item_weekday_sales',                 'round_item_store_month_sales',                 'round_store_item_weekday_sales']]


# In[ ]:


hf = h2o.H2OFrame(df_select)


# In[ ]:


hf.describe


# In[ ]:


y = 'sales'


# In[ ]:


splits = hf.split_frame(ratios = [0.8], seed = 1)
train = splits[0]
test = splits[1]


# In[ ]:


train.head()


# In[ ]:


aml = H2OAutoML(max_runtime_secs = 930001, seed = 1, project_name = 'lb_frame')
aml.train(y = y, training_frame = train, leaderboard_frame = test)


# In[ ]:


aml.leaderboard.head()


# In[ ]:


pred = aml.predict(test)
pred.head()


# In[ ]:


perf = aml.leader.model_performance(test)
perf


# In[ ]:


ptest = round(pred.as_data_frame())


# In[ ]:


sample_submission = pd.read_csv(os.path.expanduser('../input/sample_submission.csv'))
sample_submission['sales'] = ptest.astype('int')
sample_submission['id'] = sample_submission['id'].astype('str')


# In[ ]:


sample_submission.to_csv('submission.csv', index=False)

