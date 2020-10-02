#!/usr/bin/env python
# coding: utf-8

# This kernel demonstrates a way of using LightGBM with GPU support in Kaggle kernels.
# 
# Basically to avoid the following error which we get when we give `device='gpu'` in LightGBM parameters.
# 
# `LightGBMError: GPU Tree Learner was not enabled in this build.
# Please recompile with CMake option -DUSE_GPU=1`
# 
# The code & idea are heavily inspired from the following:
# * https://www.kaggle.com/inversion/ieee-simple-xgboost
# * https://www.kaggle.com/xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s
# * https://www.kaggle.com/vinhnguyen/gpu-acceleration-for-lightgbm
# * https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html
# * https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html

# ## LightGBM GPU Installation

# In[ ]:


get_ipython().system('rm -r /opt/conda/lib/python3.6/site-packages/lightgbm')


# In[ ]:


get_ipython().system('git clone --recursive https://github.com/Microsoft/LightGBM')


# In[ ]:


get_ipython().system('apt-get install -y -qq libboost-all-dev')


# ### Build and re-install LightGBM with GPU support

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'cd LightGBM\nrm -r build\nmkdir build\ncd build\ncmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..\nmake -j$(nproc)')


# In[ ]:


get_ipython().system('cd LightGBM/python-package/;python3 setup.py install --precompile')


# In[ ]:


get_ipython().system('mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd')
get_ipython().system('rm -r LightGBM')


# In[ ]:


# Latest Pandas version
get_ipython().system("pip install -q 'pandas==0.25' --force-reinstall")


# ## Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


print("Pandas version:", pd.__version__)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import gc
gc.enable()


# In[ ]:


import lightgbm as lgb
print("LightGBM version:", lgb.__version__)


# ## Preprocessing

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')


# In[ ]:


train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(train.shape)
print(test.shape)


# In[ ]:


y_train = train['isFraud'].copy()
del train_transaction, train_identity, test_transaction, test_identity
gc.collect()


# In[ ]:


# Drop target, fill in NaNs
X_train = train.drop('isFraud', axis=1)
X_test = test.copy()
del train, test
gc.collect()


# In[ ]:


X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)


# In[ ]:


# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))


# ## Modeling

# In[ ]:


# LGBMClassifier with GPU
clf = lgb.LGBMClassifier(
    max_bin = 63,
    num_leaves = 255,
    num_iterations = 500,
    learning_rate = 0.01,
    tree_learner = 'serial',
    task = 'train',
    is_training_metric = False,
    min_data_in_leaf = 1,
    min_sum_hessian_in_leaf = 100,
    sparse_threshold=1.0,
    device = 'gpu',
    num_thread = -1,
    save_binary= True,
    seed= 42,
    feature_fraction_seed = 42,
    bagging_seed = 42,
    drop_seed = 42,
    data_random_seed = 42,
    objective = 'binary',
    boosting_type = 'gbdt',
    verbose = 1,
    metric = 'auc',
    is_unbalance = True,
    boost_from_average = False,
)


# In[ ]:


get_ipython().run_line_magic('time', 'clf.fit(X_train, y_train)')


# In[ ]:


gc.collect()


# ## Feature Importances

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,X_train.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:20])
plt.title('LightGBM Feature Importance - Top 20')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances.png')


# ## Submission

# In[ ]:


sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
sample_submission.to_csv('simple_lightgbm_gpu.csv')


# With `device='gpu'` parameter commented, it takes ~ 22 minutes to fit on CPU.
# 
# `CPU times: user 42min 4s, sys: 13 s, total: 42min 17s
# Wall time: 21min 47s`
# 
# With `device='gpu'`, it takes ~ 3 minutes to fit on GPU.
# 
# `CPU times: user 3min 59s, sys: 46 s, total: 4min 45s
# Wall time: 2min 34s`
# 
# *Note: The CPU provided in Kaggle GPU kernel is 2 core, so the time to fit with above parameters might take half the time(~11 minutes) on a CPU only kernel(4 core CPU), which is still slower than LightGBM GPU implementation.*
