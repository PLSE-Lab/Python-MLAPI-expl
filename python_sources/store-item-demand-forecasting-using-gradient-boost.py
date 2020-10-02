#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os


# In[ ]:


#Import Train and Test Dataset
train_df = pd.read_csv("../input/demand-forecasting-kernels-only/train.csv")
test_df = pd.read_csv("../input/demand-forecasting-kernels-only/test.csv")


# In[ ]:


train_df.head()


# In[ ]:


train_df['trsin_or_test'], test_df['trsin_or_test'] = 'train', 'test'
data_df = pd.concat([train_df, test_df])
data_df.drop("id",axis=1,inplace=True)


# In[ ]:


#Change Data type of Date Column
data_df['date']=pd.to_datetime(data_df['date'])


# In[ ]:


data_df['year'] = data_df['date'].dt.year
data_df['quarter'] = data_df['date'].dt.quarter
data_df['month'] = data_df['date'].dt.month
data_df['weekofyear'] = data_df['date'].dt.weekofyear
data_df['weekday'] = data_df['date'].dt.weekday
data_df['dayofweek'] = data_df['date'].dt.dayofweek


# In[ ]:


data_df.head()


# In[ ]:


data_df['item_quarter_mean'] = data_df.groupby(['quarter', 'item'])['sales'].transform('mean')


# In[ ]:


data_df.head()


# In[ ]:


data_df['store_quarter_mean'] = data_df.groupby(['quarter', 'store'])['sales'].transform('mean')
data_df['store_item_quarter_mean'] = data_df.groupby(['quarter', 'item', 'store'])['sales'].transform('mean')


# In[ ]:


data_df['item_month_mean'] = data_df.groupby(['month', 'item'])['sales'].transform('mean')
data_df['store_month_mean'] = data_df.groupby(['month', 'store'])['sales'].transform('mean')
data_df['store_item_month_mean'] = data_df.groupby(['month', 'item', 'store'])['sales'].transform('mean')


# In[ ]:


data_df['item_weekofyear_mean'] = data_df.groupby(['weekofyear', 'item'])['sales'].transform('mean')
data_df['store_weekofyear_mean'] = data_df.groupby(['weekofyear', 'store'])['sales'].transform('mean')
data_df['store_item_weekofyear_mean'] = data_df.groupby(['weekofyear', 'item', 'store'])['sales'].transform('mean')


# In[ ]:


data_df['itemweekday_mean'] = data_df.groupby(['weekday', 'item'])['sales'].transform('mean')
data_df['storeweekday_mean'] = data_df.groupby(['weekday', 'store'])['sales'].transform('mean')
data_df['storeitemweekday_mean'] = data_df.groupby(['weekday', 'item', 'store'])['sales'].transform('mean')


# In[ ]:


data_df.isnull().sum().sum()


# In[ ]:


#Drop Date And Sales Column 
data_df.drop(['date','sales'],axis=1,inplace=True)


# In[ ]:


X= data_df[data_df['trsin_or_test'] == 'train']#.dropna().drop(['id', 'sales', 'trsin_or_test', 'date'], axis=1)
test = data_df[data_df['trsin_or_test'] == 'test']#.dropna()['sales']


# In[ ]:


X.head()


# In[ ]:


test.head()


# In[ ]:


X.drop(['trsin_or_test'],axis=1,inplace=True)
test.drop(['trsin_or_test'],axis=1,inplace=True)


# In[ ]:


Y=train_df["sales"]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=0.3,random_state=0)


# In[ ]:


#Use Gradient Boost Algorithm from Sklearn 
from sklearn.ensemble import GradientBoostingRegressor
XG=GradientBoostingRegressor()
model_XG=XG.fit(X_train, y_train)
pred_XG=model_XG.predict(X_test)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2_score(y_test,pred_XG)*100


# In[ ]:


predict=pd.DataFrame(model_XG.predict(test),columns=['sales'])


# In[ ]:


predict


# In[ ]:


subs = pd.read_csv('../input/demand-forecasting-kernels-only/sample_submission.csv',usecols=['id'])
sub=subs.join(predict)
sub.head(20)

