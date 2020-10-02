#!/usr/bin/env python
# coding: utf-8

# ## StackNet Installation

# This kernel demonstrates a way of using [StackNet](https://github.com/h2oai/pystacknet) with LightGBM, XGBoost & Catboost with GPU support in Kaggle kernels.

# In[ ]:


import sys
package_dir = "../input/pystacknet/repository/h2oai-pystacknet-af571e0"
sys.path.append(package_dir)


# ### LightGBM GPU Installation
# Full implementation and tutorial at https://www.kaggle.com/kirankunapuli/ieee-fraud-lightgbm-with-gpu

# In[ ]:


get_ipython().system('rm -r /opt/conda/lib/python3.6/site-packages/lightgbm')
get_ipython().system('git clone --recursive https://github.com/Microsoft/LightGBM')
get_ipython().system('apt-get install -y -qq libboost-all-dev')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'cd LightGBM\nrm -r build\nmkdir build\ncd build\ncmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..\nmake -j$(nproc)')


# In[ ]:


get_ipython().system('cd LightGBM/python-package/;python3 setup.py install --precompile')


# In[ ]:


get_ipython().system('mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd')
get_ipython().system('rm -r LightGBM')


# ## Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import gc
gc.enable()


# In[ ]:


from pathlib import Path


# In[ ]:


from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from pystacknet.pystacknet import StackNetClassifier


# ## Preprocessing

# In[ ]:


input_path = Path('../input/ieee-fraud-detection')


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


train_transaction = pd.read_csv(input_path/'train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv(input_path/'test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv(input_path/'train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv(input_path/'test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv(input_path/'sample_submission.csv', index_col='TransactionID')


# In[ ]:


train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(train.shape)
print(test.shape)


# In[ ]:


y_train = train['isFraud'].copy()
y_train = y_train.astype('category')
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


# In[ ]:


# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
# WARNING! THIS CAN DAMAGE THE DATA 
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_train = reduce_mem_usage(X_train)\nX_test = reduce_mem_usage(X_test)')


# In[ ]:


gc.collect()


# ## StackNet Model

# In[ ]:


# LGBMClassifier with GPU
# Params from https://www.kaggle.com/kirankunapuli/ieee-fraud-lightgbm-with-gpu

clf_lgb = LGBMClassifier(
    max_bin=63,
    num_leaves=255,
    num_iterations=1000,
    learning_rate=0.01,
    tree_learner="serial",
    task="train",
    is_training_metric=False,
    min_data_in_leaf=1,
    min_sum_hessian_in_leaf=100,
    sparse_threshold=1.0,
    device="gpu",
    num_thread=-1,
    save_binary=True,
    seed=42,
    feature_fraction_seed=42,
    bagging_seed=42,
    drop_seed=42,
    data_random_seed=42,
    objective="binary",
    boosting_type="gbdt",
    verbose=1,
    metric="auc",
    is_unbalance=True,
    boost_from_average=False,
)


# In[ ]:


# XGBClassifier with GPU
# Params from https://www.kaggle.com/xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s

clf_xgb = XGBClassifier(
    n_estimators=1000,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    n_jobs=-1,
    random_state=42,
    tree_method="gpu_hist",
)


# In[ ]:


# CatBoostClassifier with GPU
# Params from https://www.kaggle.com/vincentlugat/ieee-catboost-gpu-baseline-5-kfold

param_cb = {
        'learning_rate': 0.2,
        'bagging_temperature': 0.1, 
        'l2_leaf_reg': 30,
        'depth': 12, 
        'max_leaves': 48,
        'max_bin':255,
        'iterations' : 1000,
        'task_type':'GPU',
        'loss_function' : "Logloss",
        'objective':'CrossEntropy',
        'eval_metric' : "AUC",
        'bootstrap_type' : 'Bayesian',
        'random_seed':42,
        'early_stopping_rounds' : 100,
}
clf_ctb = CatBoostClassifier(silent=True, **param_cb)


# In[ ]:


gc.collect()


# In[ ]:


models = [  ######## First level ########
            [clf_lgb, clf_xgb, clf_ctb],
            ######## Second level ########
            [clf_lgb],
]


# In[ ]:


# StackNetClassifier with GPU

model = StackNetClassifier(
    models,
    metric="auc",
    folds=2,
    restacking=False,
    use_retraining=False,
    use_proba=True,
    random_state=42,
    verbose=1,
)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


gc.collect()


# ## Submission

# In[ ]:


features = [c for c in X_train.columns]


# In[ ]:


sample_submission['isFraud'] = model.predict_proba(X_test[features].values)[:,1]
sample_submission.to_csv('submission_stacknet.csv')

