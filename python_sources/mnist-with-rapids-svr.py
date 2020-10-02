#!/usr/bin/env python
# coding: utf-8

# [Rapids](https://rapids.ai) is an open-source GPU accelerated Data Science and Machine Learning library, developed and mainatained by [Nvidia](https://www.nvidia.com). It is designed to be compatible with many existing CPU tools, such as Pandas, scikit-learn, numpy, etc. It enables **massive** acceleration of many data-science and machine learning tasks, oftentimes by a factor fo 100X, or even more. 
# 
# Rapids is still undergoing developemnt, and as of right now it's not availabel in the Kaggle Docker environment. If you are interested in installing and riunning Rapids locally on your own machine, then you shoudl [refer to the followong instructions](https://rapids.ai/start.html).
# 
# In this kernel we'll try to use Rapids SVR algorithm in order to showcase its speed. Unfortuantely, in the present version (0.13) of rapids multi-label classification is still not implemented, so we'll have to using regression. The results are far from being even passable in terms of accuracy, but still an improvement over 22% accuracy that you get from simpel linear regression, and 10% random chance.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)\nimport sys\n!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz\n!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz\nsys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


import cupy as cp # linear algebra
import cudf as cd # data processing, CSV file I/O (e.g. pd.read_csv)

from cuml.svm import SVR
from cuml.decomposition import PCA

from sklearn.model_selection import KFold
from cuml.metrics import accuracy_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import gc
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = cd.read_csv('../input/digit-recognizer/train.csv')
test = cd.read_csv('../input/digit-recognizer/test.csv')
submission = cd.read_csv('../input/digit-recognizer/sample_submission.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', "X = train[train.columns[1:]].values.astype('float32')\ntest = test.values.astype('float32')\nY = train.label.values.astype('float32')")


# In[ ]:


train_oof = cp.zeros((X.shape[0], ))
test_preds = 0
train_oof.shape


# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', 'n_splits = 5\nkf = KFold(n_splits=n_splits, random_state=137)\n\nfor jj, (train_index, val_index) in enumerate(kf.split(X)):\n    print("Fitting fold", jj+1)\n    train_features = X[train_index]\n    train_target = Y[train_index]\n    val_features = X[val_index]\n    val_target = Y[val_index]\n    \n    model =  SVR(kernel=\'rbf\',C=20)\n    model.fit(train_features, train_target)\n    val_pred = model.predict(val_features)\n    val_pred = cp.clip(val_pred.values, 0, 9)\n    train_oof[val_index] = val_pred.astype(\'int\')\n    print("Fold accuracy:", accuracy_score(val_target, val_pred.astype(\'int\')))\n    test_preds += model.predict(test)/n_splits\n          \n    del train_features, train_target, val_features, val_target\n    gc.collect()')


# In[ ]:





# In[ ]:


test_preds = cp.clip(test_preds.values, 0, 9)

submission['Label'] = test_preds.astype('int')
submission.to_csv('submission.csv', index=False)


# In[ ]:



