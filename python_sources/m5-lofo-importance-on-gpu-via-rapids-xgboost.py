#!/usr/bin/env python
# coding: utf-8

# ## Install RAPIDS for faster feature engineering on GPU
# https://www.kaggle.com/cdeotte/rapids

# In[ ]:


import sys
get_ipython().system('cp ../input/rapids/rapids.0.14.0 /opt/conda/envs/rapids.tar.gz')
get_ipython().system('cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null')
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
get_ipython().system('cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# ## Get rapids-kaggle-utils

# In[ ]:


get_ipython().system('git clone https://github.com/aerdem4/rapids-kaggle-utils.git')
get_ipython().run_line_magic('cd', 'rapids-kaggle-utils/')


# ## Get the latest Xgboost with GPU support

# In[ ]:


get_ipython().system('pip install -U xgboost')


# ## Get the current best public kernel features and parameters
# Adapted from https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50
# 
# Switched from pandas to **cudf** for the speed boost.

# In[ ]:


from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb

import cudf
import cu_utils.transform as cutran


# In[ ]:


h = 28 
max_lags = 57
tr_last = 1913
fday = datetime(2016,4, 25) 
FIRST_DAY = 1000
fday


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef create_df(start_day):\n    prices = cudf.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")\n            \n    cal = cudf.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")\n    cal["date"] = cal["date"].astype("datetime64[ms]")\n    \n    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]\n    catcols = [\'id\', \'item_id\', \'dept_id\',\'store_id\', \'cat_id\', \'state_id\']\n    dt = cudf.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv", usecols = catcols + numcols)\n    \n    dt = cudf.melt(dt,\n                  id_vars = catcols,\n                  value_vars = [col for col in dt.columns if col.startswith("d_")],\n                  var_name = "d",\n                  value_name = "sales")\n    \n    dt = dt.merge(cal, on= "d")\n    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"])\n    \n    return dt\n\n\ndf = create_df(FIRST_DAY)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef transform(data):\n    \n    nan_features = [\'event_name_1\', \'event_type_1\', \'event_name_2\', \'event_type_2\']\n    for feature in nan_features:\n        data[feature].fillna(\'unknown\', inplace = True)\n    \n    data[\'id_encode\'], _ = data["id"].factorize()\n    \n    cat = [\'item_id\', \'dept_id\', \'cat_id\', \'store_id\', \'state_id\', \'event_name_1\', \'event_type_1\', \'event_name_2\', \'event_type_2\']\n    for feature in cat:\n        data[feature], _ = data[feature].factorize()\n    \n    return data\n        \n        \ndf = transform(df)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef create_fea(data):\n\n    for lag in [7, 28]:\n        out_col = "lag_{}".format(str(lag))\n        data[out_col] = data[["id", "sales"]].groupby("id", method=\'cudf\').apply_grouped(cutran.get_cu_shift_transform(shift_by=lag),\n                                                                      incols={"sales": \'x\'},\n                                                                      outcols=dict(y_out=np.float32),\n                                                                      tpb=32)["y_out"]\n    \n        for window in [7, 28]:\n            out_col = "rmean_{lag}_{window}".format(lag=lag, window=window)\n            data[out_col] = data[["id", "lag_{}".format(lag)]].groupby("id", method=\'cudf\').apply_grouped(cutran.get_cu_rolling_mean_transform(window),\n                                                                          incols={"lag_{}".format(lag): \'x\'},\n                                                                          outcols=dict(y_out=np.float32),\n                                                                          tpb=32)["y_out"]\n\n    # time features\n    data[\'date\'] = data[\'date\'].astype("datetime64[ms]")\n    data[\'year\'] = data[\'date\'].dt.year\n    data[\'month\'] = data[\'date\'].dt.month\n    data[\'day\'] = data[\'date\'].dt.day\n    data[\'dayofweek\'] = data[\'date\'].dt.weekday\n    \n    \n    return data\n\n\n    \n\n# define list of features\nfeatures = [\'item_id\', \'dept_id\', \'cat_id\', \'store_id\', \'state_id\',\n            \'event_name_1\', \'event_type_1\', \'event_name_2\', \'event_type_2\', \n            \'snap_CA\', \'snap_TX\', \'snap_WI\', \'sell_price\', \n            \'year\', \'month\', \'day\', \'dayofweek\',\n            \'lag_7\', \'lag_28\', \'rmean_7_7\', \'rmean_7_28\', \'rmean_28_7\', \'rmean_28_28\'\n           ]\n\n\ndf = create_fea(df)\ndf.tail()')


# ## Install LOFO and get the feature importances

# In[ ]:


get_ipython().system('pip install lofo-importance')


# In[ ]:


from lofo import LOFOImportance, Dataset, plot_importance
from sklearn.model_selection import KFold
import xgboost

sample_df = df.to_pandas().sample(frac=0.2, random_state=0)
sample_df.sort_values("date", inplace=True)

cv = KFold(n_splits=7, shuffle=False, random_state=0)

dataset = Dataset(df=sample_df, target="sales", features=features)

# define the validation scheme and scorer
params = {"objective": "count:poisson",
          "learning_rate" : 0.075,
          "max_depth": 8,
          'n_estimators': 200,
          'min_child_weight': 50,
          "tree_method": 'gpu_hist', "gpu_id": 0}
xgb_reg = xgboost.XGBRegressor(**params)
lofo_imp = LOFOImportance(dataset, cv=cv, scoring="neg_mean_squared_error", model=xgb_reg)

# get the mean and standard deviation of the importances in pandas format
importance_df = lofo_imp.get_importance()


# In[ ]:


plot_importance(importance_df, figsize=(12, 12))


# In[ ]:




