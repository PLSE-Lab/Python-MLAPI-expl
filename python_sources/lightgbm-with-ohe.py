#!/usr/bin/env python
# coding: utf-8

# ### Libraries

# In[ ]:


import pandas as pd
import numpy as np
import os


# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# In[ ]:


print("train data size:",train.shape)
print("test data size:",test.shape)


# In[ ]:


print("train:\n",train.head())
print("test:\n",test.head())


# In[ ]:


train['train/test']='train'
test['train/test']='test'
target=train.sales
train.drop('sales',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)


# In[ ]:


df=pd.concat([train,test])


# In[ ]:


df.head()


# In[ ]:


print(df.shape,"\n",df.dtypes)


# In[ ]:


df['date']=pd.to_datetime(df['date'])


# In[ ]:


df['dayofmonth'] = df.date.dt.day.astype(str)
df['dayofyear'] = df.date.dt.dayofyear.astype(str)
df['dayofweek'] = df.date.dt.dayofweek.astype(str)
df['month'] = df.date.dt.month.astype(str)
#df['year'] = df.date.dt.year.astype(str)
df['weekofyear'] = df.date.dt.weekofyear.astype(str)
#df['is_month_start'] = (df.date.dt.is_month_start).astype(int)
#df['is_month_end'] = (df.date.dt.is_month_end).astype(int)
df[360:370]


# In[ ]:


df.dtypes


# In[ ]:


trntst=df['train/test']
df.drop('train/test',axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df=pd.get_dummies(df)')


# In[ ]:


df.shape


# In[ ]:


df['train_or_test']=trntst
del trntst


# In[ ]:


df.shape


# In[ ]:


df.drop('date',axis=1,inplace=True)


# In[ ]:


x=df[df['train_or_test']=='train']
test=df[df['train_or_test']=='test']


# In[ ]:


x.drop('train_or_test',axis=1,inplace=True)
test.drop('train_or_test',axis=1,inplace=True)


# In[ ]:


print(x.shape,test.shape)


# ### LightGBM

# In[ ]:


import lightgbm as lgb


# In[ ]:


lg=lgb.LGBMRegressor()
lg


# In[ ]:


get_ipython().run_cell_magic('time', '', 'lg.fit(x,target)')


# In[ ]:


lg.score(x,target)


# In[ ]:


get_ipython().run_cell_magic('time', '', "predict=pd.DataFrame(lg.predict(test),columns=['sales'])")


# In[ ]:


predict.head()


# In[ ]:


ids=pd.read_csv("../input/test.csv",usecols=['id'])
sub=ids.join(predict)


# In[ ]:


sub.to_csv("lgbm_get_dummies.csv",index=False)

