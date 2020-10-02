#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('time', '', '# INSTALL RAPIDS WITH CONDA. TAKES 6 MINUTES :-(\nimport sys\n!conda create -n rapids -c rapidsai/label/xgboost -c rapidsai -c nvidia -c conda-forge rapids=0.11 python=3.6 cudatoolkit=10.1 --yes\nsys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import cudf, cuml
import cupy as cp
from cuml.linear_model import LogisticRegression
import numpy as np
#from cuml.metrics import accuracy_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import gc
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


from cuml.neighbors import KNeighborsClassifier, NearestNeighbors


# In[ ]:


train = cudf.read_csv('../input/multi-cat-encodings/X_train_te.csv')
test = cudf.read_csv('../input/multi-cat-encodings/X_test_te.csv')
sample_submission = cudf.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')


# In[ ]:


sample_submission.head()


# In[ ]:


train_oof = cp.zeros((train.shape[0],))
test_preds = 0
train_oof.shape


# In[ ]:


import cupy as cp


# In[ ]:


def auc_cp(y_true,y_pred):
  y_true = y_true.astype('float32')
  ids = np.argsort(-y_pred) # we want descedning order
  y_true = y_true[ids.values]
  y_pred = y_pred[ids.values]
  zero = 1 - y_true
  acc_one = cp.cumsum(y_true)
  acc_zero = cp.cumsum(zero)
  sum_one = cp.sum(y_true)
  sum_zero = cp.sum(zero)
  tpr = acc_one/sum_one
  fpr = acc_zero/sum_zero
  return calculate_area(fpr,tpr)

def calculate_area(fpr,tpr):
  return cp.sum((fpr[1:]-fpr[:-1])*(tpr[1:]+tpr[:-1]))/2


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


features = test.columns


# In[ ]:


get_ipython().run_cell_magic('time', '', 'n_splits = 5\nkf = KFold(n_splits=n_splits, random_state=137)\n\nfor jj, (train_index, val_index) in enumerate(kf.split(train)):\n    print("Fitting fold", jj+1)\n    train_features = train.loc[train[\'fold_column\'] != jj][features]\n    train_target = train.loc[train[\'fold_column\'] != jj][\'target\'].values.astype(float)\n    \n    val_features = train.loc[train[\'fold_column\'] == jj][features]\n    val_target = train.loc[train[\'fold_column\'] == jj][\'target\'].values.astype(float)\n    \n    model = KNeighborsClassifier(n_neighbors=400)\n    model.fit(train_features, train_target)\n    val_pred = model.predict_proba(val_features)[1]\n    train_oof[val_index] = val_pred\n    val_target = cp.asarray(val_target)\n    print("Fold AUC:", auc_cp(val_target, val_pred))\n    test_preds += model.predict_proba(test)[1].values/n_splits\n    del train_features, train_target, val_features, val_target\n    gc.collect()')


# In[ ]:


sample_submission['target'] = test_preds
sample_submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:


cp.save('test_preds', test_preds)
cp.save('train_oof', train_oof)


# In[ ]:




