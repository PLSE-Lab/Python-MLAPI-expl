#!/usr/bin/env python
# coding: utf-8

# # GPU-accelerated LightGBM
# 
# This kernel explores a GPU-accelerated LGBM model to predict customer transaction.
# 
# ## Notebook  Content
# 1. [Re-compile LGBM with GPU support](#1)
# 1. [Loading the data](#2)
# 1. [Training the model on CPU](#3)
# 1. [Training the model on GPU](#4)
# 1. [Submission](#5)

# <a id="1"></a> 
# ## 1. Re-compile LGBM with GPU support
# In Kaggle notebook setting, set the `Internet` option to `Internet connected`, and `GPU` to `GPU on`.

# In[ ]:


get_ipython().system('rm -r /opt/conda/lib/python3.6/site-packages/lightgbm')
get_ipython().system('git clone --recursive https://github.com/Microsoft/LightGBM')


# In[ ]:


get_ipython().system('apt-get install -y -qq libboost-all-dev')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'cd LightGBM\nrm -r build\nmkdir build\ncd build\ncmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..\nmake -j$(nproc)')


# In[ ]:


get_ipython().system('cd LightGBM/python-package/;python3 setup.py install --precompile')


# In[ ]:


get_ipython().system('mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd')
get_ipython().system('rm -r LightGBM')


# <a id="2"></a> 
# ## 2. Loading the data

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn import metrics
import gc

pd.set_option('display.max_columns', 200)


# In[ ]:


import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

#logger
def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
    
logger = get_logger()
logger.info('Input data')
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

logger.info('Features engineering')
idx = [c for c in train_df.columns if c not in ['ID_code', 'target']]
for df in [test_df, train_df]:
    df['sum'] = df[idx].sum(axis=1)  
    df['min'] = df[idx].min(axis=1)
    df['max'] = df[idx].max(axis=1)
    df['mean'] = df[idx].mean(axis=1)
    df['std'] = df[idx].std(axis=1)
    df['skew'] = df[idx].skew(axis=1)
    df['kurt'] = df[idx].kurtosis(axis=1)
    df['med'] = df[idx].median(axis=1)

logger.info('Prepare the model')
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']


# <a id="4"></a>
# ## 4. Train model on GPU

# In[ ]:


get_ipython().system('nvidia-smi')


# In order to leverage the GPU, we need to set the following parameters: 
# 
#         'device': 'gpu',
#         'gpu_platform_id': 0,
#         'gpu_device_id': 0
#         
#         

# In[ ]:


param = {
        'num_leaves': 7,
        'max_bin': 119,
        'min_data_in_leaf': 6,
        'learning_rate': 0.03,
        'min_sum_hessian_in_leaf': 0.00245,
        'bagging_fraction': 1.0, 
        'bagging_freq': 5, 
        'feature_fraction': 0.05,
        'lambda_l1': 4.972,
        'lambda_l2': 2.276,
        'min_gain_to_split': 0.65,
        'max_depth': 14,
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }


# In[ ]:


get_ipython().run_cell_magic('time', '', 'nfold = 5\n\ntarget = \'target\'\npredictors = train_df.columns.values.tolist()[2:]\n\nskf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)\n\noof = np.zeros(len(train_df))\npredictions = np.zeros(len(test_df))\n\ni = 1\nfor train_index, valid_index in skf.split(train_df, train_df.target.values):\n    print("\\nfold {}".format(i))\n    xg_train = lgb.Dataset(train_df.iloc[train_index][predictors].values,\n                           label=train_df.iloc[train_index][target].values,\n                           feature_name=predictors,\n                           free_raw_data = False\n                           )\n    xg_valid = lgb.Dataset(train_df.iloc[valid_index][predictors].values,\n                           label=train_df.iloc[valid_index][target].values,\n                           feature_name=predictors,\n                           free_raw_data = False\n                           )   \n\n    \n    clf = lgb.train(param, xg_train, 5000, valid_sets = [xg_valid], verbose_eval=500, early_stopping_rounds = 250)\n    oof[valid_index] = clf.predict(train_df.iloc[valid_index][predictors].values, num_iteration=clf.best_iteration) \n    \n    predictions += clf.predict(test_df[predictors], num_iteration=clf.best_iteration) / nfold\n    i = i + 1\n\nprint("\\n\\nCV AUC: {:<0.2f}".format(metrics.roc_auc_score(train_df.target.values, oof)))')


# <a id="5"></a>
# ## 5. Submission

# In[ ]:


sub_df = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub_df["target"] = predictions
sub_df[:10]


# In[ ]:


sub_df.to_csv("lightgbm_gpu.csv", index=False)


# In[ ]:




