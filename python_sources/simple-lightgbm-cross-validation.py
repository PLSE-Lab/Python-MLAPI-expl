#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# In[ ]:


#load train.csv data
train = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_train.csv')
test = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_test.csv')


# In[ ]:


train_x = train.drop(["Price","Date"],axis=1)
train_y = train["Price"]
test_x = test.drop(["Date"],axis=1)


# In[ ]:


param = {
 'metric': 'rmse'   
}
num_round = 1000


# In[ ]:


kf = KFold(n_splits=5, shuffle=True, random_state=71)

cv = []
preds = []

for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    
    train_data = lgb.Dataset(tr_x, tr_y)
    valid_data = lgb.Dataset(va_x, va_y)
    model = lgb.train(param, train_data, num_round, valid_sets=[train_data, valid_data], verbose_eval=50, early_stopping_rounds=10)
    va_pred = model.predict(va_x, num_iteration=model.best_iteration)
    cv.append(np.sqrt(mean_squared_error(va_y, va_pred)))
    
    test_pred = model.predict(test_x, num_iteration=model.best_iteration)
    preds.append(test_pred)


# In[ ]:


np.mean(cv)


# In[ ]:


submission = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/sampleSubmission.csv')


# In[ ]:


submission['Price'] = np.mean(preds, axis=0)


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




