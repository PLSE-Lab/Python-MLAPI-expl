#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import math
from sklearn.linear_model import LogisticRegression


# Thanks to Bojan for the starter code 
# - https://www.kaggle.com/tunguz/simple-ion-ridge-regression-starter

# # Check files in location

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Import train test and Submission file

# In[ ]:


train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')


# ## Process Train Data 

# In[ ]:


train_time = train['time'].values
train_time_0 = train_time[:50000]


# In[ ]:


for i in range(1,100):
    train_time_0 = np.hstack([train_time_0, train_time[i*50000:(i+1)*50000]])


# In[ ]:


train['time'] = train_time_0


# ## Process Test Data 

# In[ ]:


train_time_0 = train_time[:50000] 

for i in range(1,40):
    train_time_0 = np.hstack([train_time_0, train_time[i*50000:(i+1)*50000]])
    
test['time'] = train_time_0


# # Signal Processing 

# The following signal processing parts are taken from the following Khoi Nguyen kernel: 
# - https://www.kaggle.com/suicaokhoailang/an-embarrassingly-simple-baseline-0-960-lb

# In[ ]:


n_groups = 100
train["group"] = 0
for i in range(n_groups):
    ids = np.arange(i*50000, (i+1)*50000)
    train.loc[ids,"group"] = i


# In[ ]:


n_groups = 40
test["group"] = 0
for i in range(n_groups):
    ids = np.arange(i*50000, (i+1)*50000)
    test.loc[ids,"group"] = i


# In[ ]:


train['signal_2'] = 0
test['signal_2'] = 0


# In[ ]:


n_groups = 100
for i in range(n_groups):
    sub = train[train.group == i]
    signals = sub.signal.values
    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))
    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))
    signals = signals*(imax-imin)
    train.loc[sub.index,"signal_2"] = [0,] +list(np.array(signals[:-1]))


# In[ ]:


n_groups = 40
for i in range(n_groups):
    sub = test[test.group == i]
    signals = sub.signal.values
    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))
    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))
    signals = signals*(imax-imin)
    test.loc[sub.index,"signal_2"] = [0,] +list(np.array(signals[:-1]))


# # Build Model

# In[ ]:


X = train[['signal_2']].values
y = train['open_channels'].values


# In[ ]:


model = LogisticRegression(max_iter=1000)


# In[ ]:


model.fit(X,y)


# In[ ]:


train_preds = model.predict(X)


# In[ ]:


train_preds = np.clip(train_preds, 0, 10)


# In[ ]:


train_preds = train_preds.astype(int)


# In[ ]:


X_test = test[['signal_2']].values


# In[ ]:


test_preds = model.predict(X_test)
test_preds = np.clip(test_preds, 0, 10)
test_preds = test_preds.astype(int)
submission['open_channels'] = test_preds


# In[ ]:


np.set_printoptions(precision=4)


# In[ ]:


submission['time'] = [format(submission.time.values[x], '.4f') for x in range(2000000)]


# In[ ]:


submission.to_csv('submission.csv', index=False)

