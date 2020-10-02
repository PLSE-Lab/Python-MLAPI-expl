#!/usr/bin/env python
# coding: utf-8

# [Rapids](https://rapids.ai) is an open-source GPU accelerated Data Sceince and Machien Learnign library, developed adn mainatained by [Nvidia](https://www.nvidia.com). It is designed to be compatible with many existing CPU tools, such as Pandas, scikit-learn, numpy, etc. It enables **massive** acceleration of many data-science and machine learning tasks, oftentimes by a factor fo 100X, or even more. 
# 
# Rapids is still undergoing developemnt, and as of right now it's not availabel in the Kaggle Docker environment. If you are interested in installing and riunning Rapids locally on your own machine, then you shoudl [refer to the followong instructions](https://rapids.ai/start.html).
# 
# The first successful install of a Rapids library on kaggle was done by [Chris Deotte](https://www.kaggle.com/cdeotte) in the follwiong [Digit Recognizer kernel](https://www.kaggle.com/cdeotte/rapids-gpu-knn-mnist-0-97). An improved install version that uses a Kaggle Dataset for install can be found [here](https://www.kaggle.com/cdeotte/rapids-data-augmentation-mnist-0-985).  In this kerenl we'll follow that approach.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)\nimport sys\n!cp ../input/rapids/rapids.0.11.0 /opt/conda/envs/rapids.tar.gz\n!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz\nsys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


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


# In this kernels we'll use Ridge regression as our model. Ridge regression is actually a very good choice for **classification** problem when the evaluation metric is AUC - it often performs nearly as well, or even better, than Logistic Reression. 

# In[ ]:


from cuml.linear_model import Ridge


# In[ ]:


train = cudf.read_csv('../input/multi-cat-encodings/X_train_te.csv')
test = cudf.read_csv('../input/multi-cat-encodings/X_test_te.csv')
sample_submission = cudf.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')


# In[ ]:


train_oof = cp.zeros((train.shape[0],))
test_preds = 0
train_oof.shape


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


features = test.columns


# In[ ]:


get_ipython().run_cell_magic('time', '', 'n_splits = 5\nkf = KFold(n_splits=n_splits, random_state=137)\nscores = []\n\nfor jj, (train_index, val_index) in enumerate(kf.split(train)):\n    print("Fitting fold", jj+1)\n    train_features = train.loc[train[\'fold_column\'] != jj][features]\n    train_target = train.loc[train[\'fold_column\'] != jj][\'target\'].values.astype(float)\n    \n    val_features = train.loc[train[\'fold_column\'] == jj][features]\n    val_target = train.loc[train[\'fold_column\'] == jj][\'target\'].values.astype(float)\n    \n    model = Ridge(alpha = 5)\n    model.fit(train_features, train_target)\n    val_pred = model.predict(val_features)\n    train_oof[val_index] = val_pred\n    val_target = cp.asarray(val_target)\n    score = auc_cp(val_target, val_pred)\n    print("Fold AUC:", score)\n    scores.append(cp.asnumpy(score))\n    test_preds += model.predict(test).values/n_splits\n    del train_features, train_target, val_features, val_target\n    gc.collect()\n    \nprint("Mean AUC:", np.mean(scores))')


# In[ ]:


sample_submission['target'] = test_preds
sample_submission.to_csv('submission.csv', index=False)


# In[ ]:


cp.save('test_preds', test_preds)
cp.save('train_oof', train_oof)

