#!/usr/bin/env python
# coding: utf-8

# ## Install RAPIDS for faster feature engineering on GPU
# https://www.kaggle.com/cdeotte/rapids

# In[ ]:


import sys
get_ipython().system('cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz')
get_ipython().system('cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz')
sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
get_ipython().system('cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# ## Get rapids-kaggle-utils

# In[ ]:


get_ipython().system('git clone https://github.com/aerdem4/rapids-kaggle-utils.git')
get_ipython().run_line_magic('cd', 'rapids-kaggle-utils/')


# ## Install LOFO

# In[ ]:


get_ipython().system('pip install lofo-importance')


# ## Install the latest Xgboost for GPU acceleration
# #### 2 times faster than Lightgbm on CPU (4 cores)

# In[ ]:


get_ipython().system('pip install -U xgboost')


# ## Get the current best features and model from:
# 
# https://www.kaggle.com/jazivxt/physically-possible

# In[ ]:


import numpy as np 
import pandas as pd
from sklearn import *
import lightgbm as lgb
import cudf

train = cudf.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')


# In[ ]:


# rapids-kaggle-utils
from cu_utils.transform import cu_min_transform, cu_max_transform, cu_mean_transform


CU_FUNC = {"min": cu_min_transform, "max": cu_max_transform, "mean": cu_mean_transform}


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n\ndef features(df):\n    df = df.sort_values(by=[\'time\']).reset_index(drop=True)\n    df["index"] = (df["time"] * 10_000) - 1\n    df[\'batch\'] = df["index"] // 50_000\n    df[\'batch_index\'] = df["index"]  - (df["batch"] * 50_000)\n    df[\'batch_slices\'] = df[\'batch_index\']  // 5_000\n    df[\'batch_slices2\'] = df[\'batch\'].astype(str) + "_" + df[\'batch_slices\'].astype(str)\n    \n    for c in [\'batch\',\'batch_slices2\']:\n\n        df["abs_signal"] = df["signal"].abs()\n        for abs_val in [True, False]:\n            for func in ["min", "max", "mean"]:\n                output_col = func + c\n                input_col = "signal"\n                if abs_val:\n                    output_col = "abs_" + output_col\n                    input_col = "abs_" + input_col\n                df = df.groupby([c], method=\'cudf\').apply_grouped(CU_FUNC[func],\n                                                                  incols={input_col: \'x\'},\n                                                                  outcols=dict(y_out=np.float32),\n                                                                  tpb=32).rename({\'y_out\': output_col})\n        \n        df[\'range\'+c] = df[\'max\'+c] - df[\'min\'+c]\n        df[\'maxtomin\'+c] = df[\'max\'+c] / df[\'min\'+c]\n        df[\'abs_avg\'+c] = (df[\'abs_min\'+c] + df[\'abs_max\'+c]) / 2\n\n    for c in [c1 for c1 in df.columns if c1 not in [\'time\', \'signal\', \'open_channels\', \'batch\', \'batch_index\', \'batch_slices\', \'batch_slices2\']]:\n        df[c+\'_msignal\'] = df[c] - df[\'signal\']\n        \n    return df\n\ntrain = features(train)\ntrain.shape')


# ## Get Feature Importances on Time Split Mean Squared Error

# In[ ]:


from lofo import LOFOImportance, Dataset, plot_importance
from sklearn.model_selection import KFold
import xgboost

# Convert to pandas for now. Xgboost supports cudf but LOFO doesn't support yet
sample_df = train.to_pandas().sample(frac=0.1, random_state=0)
sample_df.sort_values("time", inplace=True)

# define the validation scheme
cv = KFold(n_splits=5, shuffle=False, random_state=0)

# define the binary target and the features
features = [c for c in train.columns if c not in ['time', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]
dataset = Dataset(df=sample_df, target="open_channels", features=features)

# define the validation scheme and scorer
params ={'learning_rate': 0.8, 'max_depth': 4, "n_estimators ": 100, "tree_method": 'gpu_hist', "gpu_id": 0}
xgb_reg = xgboost.XGBRegressor(**params)
lofo_imp = LOFOImportance(dataset, cv=cv, scoring="neg_mean_squared_error", model=xgb_reg)

# get the mean and standard deviation of the importances in pandas format
importance_df = lofo_imp.get_importance()


# In[ ]:


plot_importance(importance_df, figsize=(12, 20))


# In[ ]:




