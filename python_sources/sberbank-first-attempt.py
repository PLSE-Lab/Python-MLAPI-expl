#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import train_test_split


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
macro_df=pd.read_csv('../input/macro.csv')
train_df.shape


# In[ ]:


train_df = pd.merge_ordered(train_df, macro_df, on='timestamp', how='left')
result_df = pd.merge_ordered(test_df, macro_df, on='timestamp', how='left')
train_df.head()


# In[ ]:


train_df[(train_df.price_doc<40000000)].hist(column='price_doc',bins=200)


# In[ ]:


train_df['month']=train_df['timestamp'].map(lambda x:(int(x[0:4])-2011)*12+int(x[5:7])-8)
result_df['month']=result_df['timestamp'].map(lambda x:(int(x[0:4])-2011)*12+int(x[5:7])-8)


# In[ ]:


train_df_numeric = train_df.select_dtypes(exclude=['object'])
train_df_obj = train_df.select_dtypes(include=['object']).copy()

for column in train_df_obj:
    train_df_obj[column] = pd.factorize(train_df_obj[column])[0]

train_df_values = pd.concat([train_df_numeric, train_df_obj], axis=1)[:]
test_df_values = pd.concat([train_df_numeric, train_df_obj], axis=1)[25001:]


# In[ ]:


result_df_numeric = result_df.select_dtypes(exclude=['object'])
result_df_obj = result_df.select_dtypes(include=['object']).copy()

for column in result_df_obj:
    result_df_obj[column] = pd.factorize(result_df_obj[column])[0]

result_df_values = pd.concat([result_df_numeric, result_df_obj], axis=1)


# In[ ]:


result_df_values[(result_df_values.build_year<1800)&(result_df_values.build_year>1)].head(20)


# In[ ]:


X_train = train_df_values[(train_df_values.full_sq<1000)&(train_df_values.price_doc!=1000000)&(train_df_values.price_doc!=2000000)].drop(['price_doc','id','timestamp'],axis=1)
Y_train = np.log1p(train_df_values[(train_df_values.full_sq<1000)&(train_df_values.price_doc!=1000000)&(train_df_values.price_doc!=2000000)]['price_doc'].values.reshape(-1,1))
X_train.shape


# In[ ]:


X_test = test_df_values.drop(['price_doc','id','timestamp'],axis=1)
Y_test = np.log1p(test_df_values['price_doc'].values.reshape(-1,1))
X_test.shape


# In[ ]:


X_result = result_df_values.drop(['id','timestamp'],axis=1)
id_test = result_df_values['id']
X_result.shape


# In[ ]:


dtrain = xgb.DMatrix(X_train, Y_train)
dtest = xgb.DMatrix(X_test, Y_test)
dresult=xgb.DMatrix(X_result)


# In[ ]:


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 1,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
# Uncomment to tune XGB `num_boost_rounds`
#model = xgb.cv(xgb_params, dtrain, num_boost_round=200,
                  #early_stopping_rounds=30, verbose_eval=10)

model = xgb.train(xgb_params, dtrain, num_boost_round=160)


# In[ ]:


y_pred=model.predict(dresult)
output=pd.DataFrame(data={'price_doc':np.exp(y_pred)-1},index=id_test)
output.head()


# In[ ]:


output.to_csv('output.csv',header=True)


# In[ ]:




