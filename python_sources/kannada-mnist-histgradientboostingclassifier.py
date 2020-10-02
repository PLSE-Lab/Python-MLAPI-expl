#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import gc
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/Kannada-MNIST/train.csv')
test = pd.read_csv('../input/Kannada-MNIST/test.csv')
submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
submission.head()


# In[ ]:


train.head(20)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X = train[train.columns[1:]].values\nY = train.label.values\n\ntrain_oof = np.zeros((X.shape[0], 10))\ntest_preds = 0\ntrain_oof.shape')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'n_splits = 5\nkf = KFold(n_splits=n_splits, random_state=137)\n\nfor jj, (train_index, val_index) in enumerate(kf.split(X)):\n    print("Fitting fold", jj+1)\n    train_features = X[train_index]\n    train_target = Y[train_index]\n    \n    val_features = X[val_index]\n    val_target = Y[val_index]\n    \n    model = HistGradientBoostingClassifier(max_iter=250)\n    model.fit(train_features, train_target)\n    val_pred = model.predict_proba(val_features)\n    train_oof[val_index] = val_pred\n    print("Fold accuracy:", accuracy_score(val_target, np.argmax(val_pred, axis=1)))\n    test_preds += model.predict_proba(test[test.columns[1:]].values)/n_splits\n    del train_features, train_target, val_features, val_target\n    gc.collect()')


# In[ ]:


print(accuracy_score(Y, np.argmax(train_oof, axis=1)))


# In[ ]:


preds = np.argmax(test_preds, axis=1)
submission['label'] = preds
submission.to_csv('submission.csv', index=False)
submission.head(20)


# In[ ]:


np.save('test_preds', test_preds)
np.save('train_oof', train_oof)


# In[ ]:


test_preds


# In[ ]:




