#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import gc
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = np.load('../input/multi-cat-encodings/X_train_le.npy')
test = np.load('../input/multi-cat-encodings/X_test_le.npy')
sample_submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')
target = np.load('../input/multi-cat-encodings/target.npy')


# In[ ]:


np.unique(target)


# In[ ]:


sample_submission.head()


# In[ ]:


train_oof = np.zeros((train.shape[0],))
test_preds = 0
train_oof.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', 'n_splits = 5\nkf = KFold(n_splits=n_splits, random_state=137)\n\nfor jj, (train_index, val_index) in enumerate(kf.split(train)):\n    print("Fitting fold", jj+1)\n    train_features = train[train_index]\n    train_target = target[train_index]\n    \n    val_features = train[val_index]\n    val_target = target[val_index]\n    \n    model = HistGradientBoostingClassifier(max_iter=10000, learning_rate=0.01)\n    model.fit(train_features, train_target)\n    val_pred = model.predict_proba(val_features)\n    train_oof[val_index] = val_pred[:,1]\n    print("Fold AUC:", roc_auc_score(val_target, val_pred[:,1]))\n    test_preds += model.predict_proba(test)[:,1]/n_splits\n    del train_features, train_target, val_features, val_target\n    gc.collect()')


# In[ ]:


print(roc_auc_score(target, train_oof))


# In[ ]:


sample_submission['target'] = test_preds
sample_submission.to_csv('submission.csv', index=False)


# In[ ]:


np.save('test_preds', test_preds)
np.save('train_oof', train_oof)


# In[ ]:




