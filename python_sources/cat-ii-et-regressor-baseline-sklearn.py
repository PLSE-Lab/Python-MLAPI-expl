#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import ExtraTreesRegressor
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



train = pd.read_csv("/kaggle/input/multi-cat-encodings/X_train_te.csv")
test = pd.read_csv("/kaggle/input/multi-cat-encodings/X_test_te.csv")
sample_submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')
target = np.load('../input/multi-cat-encodings/target.npy')


# In[ ]:


np.unique(target)


# In[ ]:


sample_submission.head()


# In[ ]:





# In[ ]:


train_oof = np.zeros((600000,))
test_preds = 0
train_oof.shape


# In[ ]:


#test = te_0.tocsr()


# In[ ]:


features = test.columns


# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', 'n_splits = 5\nkf = KFold(n_splits=n_splits, random_state=137)\nscores = []\n\nfor jj, (train_index, val_index) in enumerate(kf.split(train)):\n    print("Fitting fold", jj+1)\n    train_features = train.loc[train[\'fold_column\'] != jj][features]\n    train_target = train.loc[train[\'fold_column\'] != jj][\'target\'].values.astype(float)\n    \n    val_features = train.loc[train[\'fold_column\'] == jj][features]\n    val_target = train.loc[train[\'fold_column\'] == jj][\'target\'].values.astype(float)\n    \n    model = ExtraTreesRegressor(n_estimators=150)\n    model.fit(train_features, train_target)\n    val_pred = model.predict(val_features)\n    train_oof[val_index] = val_pred\n    score = roc_auc_score(val_target, val_pred)\n    scores.append(score)\n    print("Fold AUC:", score)\n    test_preds += model.predict(test)/n_splits\n    del train_features, train_target, val_features, val_target\n    gc.collect()\n    \nprint("Mean AUC:", np.mean(scores))')


# In[ ]:


model


# In[ ]:



roc_auc_score(target, train_oof)


# In[ ]:


sample_submission['target'] = test_preds
sample_submission.to_csv('submission.csv', index=False)


# In[ ]:


np.save('test_preds', test_preds)
np.save('train_oof', train_oof)


# In[ ]:




