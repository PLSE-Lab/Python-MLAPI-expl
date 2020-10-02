#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


from catboost import CatBoostRegressor, Pool


# In[ ]:


train = pd.read_csv('../input/train.csv', iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})


# In[ ]:


def gen_features(X):
    strain = []
    strain.append(X.mean())
    strain.append(X.std())
    strain.append(X.min())
    strain.append(X.max())
    strain.append(X.mad())
    strain.append(X.kurtosis())
    strain.append(X.skew())
    strain.append(np.quantile(X,0.01))
    strain.append(np.quantile(X,0.05))
    strain.append(np.quantile(X,0.95))
    strain.append(np.quantile(X,0.99))
    strain.append(np.abs(X).max())
    strain.append(np.abs(X).mean())
    strain.append(np.abs(X).std())
    return pd.Series(strain)


# In[ ]:


X_train = pd.DataFrame()
y_train = pd.Series()
for df in train:
    ch = gen_features(df['acoustic_data'])
    X_train = X_train.append(ch, ignore_index=True)
    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))


# In[ ]:


X_train.describe()


# In[ ]:


train_pool = Pool(X_train, y_train)


# In[ ]:


m = CatBoostRegressor(iterations=10000, loss_function='MAE', boosting_type='Ordered')
m.fit(X_train, y_train, silent=True)
m.best_score_


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')


# In[ ]:


X_test = pd.DataFrame()


# In[ ]:


for seg_id in submission.index:
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    ch = gen_features(seg['acoustic_data'])
    X_test = X_test.append(ch, ignore_index=True)


# In[ ]:


X_test.describe()


# In[ ]:


submission['time_to_failure'] = m.predict(X_test).clip(0, 16)
submission.to_csv('submission.csv')

