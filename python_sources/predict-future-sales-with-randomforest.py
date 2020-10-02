#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


items = pd.read_csv('../input/items.csv')
shops = pd.read_csv('../input/shops.csv')
cat = pd.read_csv('../input/item_categories.csv')
train = pd.read_csv('../input/sales_train.csv')
test  = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train.info()


# In[ ]:


type(train['date'])


# In[ ]:


train.head()


# In[ ]:


test.tail()


# In[ ]:


print(sorted(train['shop_id'].unique()))


# In[ ]:


print(sorted(test['shop_id'].unique()))


# In[ ]:


gp = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['sum']})


# In[ ]:


X = np.array(list(map(list, gp.index.values)))
y_train = gp.values


# In[ ]:


test['date_block_num'] = train['date_block_num'].max() + 1
X_test = test[['date_block_num', 'shop_id', 'item_id']].values


# In[ ]:


samp = np.random.permutation(np.arange(len(y_train)))[:int(len(y_train)*.2)]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[samp,1], X[samp,2], y_train[samp].ravel(), c=X[samp,0], marker='.')


# In[ ]:


oh0 = OneHotEncoder(categories='auto').fit(X[:,0].reshape(-1, 1))
x0 = oh0.transform(X[:,0].reshape(-1, 1))


# In[ ]:


oh1 = OneHotEncoder(categories='auto').fit(X[:,1].reshape(-1, 1))
x1 = oh1.transform(X[:,1].reshape(-1, 1))
x1_t = oh1.transform(X_test[:,1].reshape(-1, 1))


# In[ ]:


print(X[:, :1].shape, x1.toarray().shape, X[:, 2:].shape)
X_train = np.concatenate((X[:, :1], x1.toarray(), X[:, 2:]), axis=1)
X_test = np.concatenate((X_test[:, :1], x1_t.toarray(), X_test[:, 2:]), axis=1)


# In[ ]:


dmy = DummyRegressor().fit(X_train, y_train)


# In[ ]:


reg = LinearRegression().fit(X_train, y_train)


# In[ ]:


rfr = RandomForestRegressor().fit(X_train, y_train.ravel())


# In[ ]:


rmse_dmy = np.sqrt(mean_squared_error(y_train, dmy.predict(X_train)))
print('Dummy RMSE: %.4f' % rmse_dmy)
rmse_reg = np.sqrt(mean_squared_error(y_train, reg.predict(X_train)))
print('LR RMSE: %.4f' % rmse_reg)
rmse_rfr = np.sqrt(mean_squared_error(y_train, rfr.predict(X_train)))
print('RFR RMSE: %.4f' % rmse_rfr)


# In[ ]:


y_test = rfr.predict(X_test)


# In[ ]:


submission['item_cnt_month'] = y_test


# In[ ]:


submission.to_csv('xgb_submission.csv', index=False)

