#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgbm
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


train.loc[49000:50010,:]


# In[ ]:


train.shape


# In[ ]:


train['open_channels'].min()


# In[ ]:


train_time = train['time'].values


# In[ ]:


train_time_0 = train_time[:50000]


# In[ ]:


for i in range(1,100):
    train_time_0 = np.hstack([train_time_0, train_time[i*50000:(i+1)*50000]])


# In[ ]:


train_time_0.shape


# In[ ]:


train['time'] = train_time_0


# In[ ]:


test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
test.head()


# In[ ]:


test.tail()


# In[ ]:


test.shape


# In[ ]:


train_time_0 = train_time[:50000]
for i in range(1,40):
    train_time_0 = np.hstack([train_time_0, train_time[i*50000:(i+1)*50000]])
test['time'] = train_time_0


# In[ ]:


X = train[['time', 'signal']].values
y = train['open_channels'].values


# In[ ]:


model = lgbm.LGBMRegressor(n_estimators=100)
model.fit(X, y)


# In[ ]:


train_preds = model.predict(X)

train_preds
# In[ ]:


train_preds = np.clip(train_preds, 0, 10)
train_preds = train_preds.astype(int)
X_test = test[['time', 'signal']].values


# In[ ]:


submission.head()


# In[ ]:


submission.shape


# In[ ]:


X_test.shape


# In[ ]:


test_preds = model.predict(X_test)
test_preds = np.clip(test_preds, 0, 10)
test_preds = test_preds.astype(int)
submission['open_channels'] = test_preds


# In[ ]:


submission.head()


# In[ ]:


np.set_printoptions(precision=4)


# In[ ]:


submission.time.values[:20]


# In[ ]:


submission['time'] = [format(submission.time.values[x], '.4f') for x in range(2000000)]


# In[ ]:


submission.time.values[:20]


# In[ ]:


submission['open_channels'].mean()


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)

