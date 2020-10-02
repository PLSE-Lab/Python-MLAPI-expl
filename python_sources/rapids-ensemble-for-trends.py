#!/usr/bin/env python
# coding: utf-8

# This is an ensemble version of @aerdem4's excellent SVM notebook that can be found here: https://www.kaggle.com/aerdem4/rapids-svm-on-trends-neuroimaging
# 
# [Rapids](https://rapids.ai) is an open-source GPU accelerated Data Sceince and Machine Learning library, developed and mainatained by [Nvidia](https://www.nvidia.com). It is designed to be compatible with many existing CPU tools, such as Pandas, scikit-learn, numpy, etc. It enables **massive** acceleration of many data-science and machine learning tasks, oftentimes by a factor fo 100X, or even more. 
# 
# Rapids is still undergoing developemnt, and as of right now it's not availabel in the Kaggle Docker environment. If you are interested in installing and riunning Rapids locally on your own machine, then you should [refer to the followong instructions](https://rapids.ai/start.html).

# ## Install Rapids for 500x faster kNN on GPUs

# In[ ]:


import sys
get_ipython().system('cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz')
get_ipython().system('cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null')
sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path
get_ipython().system('cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


import numpy as np
# import pandas as pd
import cudf
import cupy as cp
from cuml.neighbors import KNeighborsRegressor
from cuml import SVR
from cuml.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from cuml.metrics import mean_absolute_error, mean_squared_error


def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))


# In[ ]:


fnc_df = cudf.read_csv("../input/trends-assessment-prediction/fnc.csv")
loading_df = cudf.read_csv("../input/trends-assessment-prediction/loading.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")


labels_df = cudf.read_csv("../input/trends-assessment-prediction/train_scores.csv")
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()
df = df[df["is_train"] == True].copy()

df.shape, test_df.shape


# In[ ]:


# Giving less importance to FNC features since they are easier to overfit due to high dimensionality.
FNC_SCALE = 1/600

df[fnc_features] *= FNC_SCALE
test_df[fnc_features] *= FNC_SCALE


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nNUM_FOLDS = 7\nkf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)\n\n\nfeatures = loading_features + fnc_features\n\noveral_score = 0\nfor target, c, w, ff, pp in [("age", 60, 0.3, 0.55, 0.6), ("domain1_var1", 12, 0.175, 0.2, 0.4), ("domain1_var2", 8, 0.175, 0.2, 0.5), ("domain2_var1", 9, 0.175, 0.22, 0.5), ("domain2_var2", 12, 0.175, 0.22, 0.5)]:    \n    y_oof = np.zeros(df.shape[0])\n    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))\n\n    \n    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):\n        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]\n        train_df = train_df[train_df[target].notnull()]\n\n        model_1 = SVR(C=c, cache_size=3000.0)\n        model_1.fit(train_df[features].values, train_df[target].values)\n        model_2 = Ridge(alpha = 0.0001)\n        model_2.fit(train_df[features].values, train_df[target].values)\n        val_pred_1 = model_1.predict(val_df[features])\n        val_pred_2 = model_2.predict(val_df[features])\n        \n        test_pred_1 = model_1.predict(test_df[features])\n        test_pred_2 = model_2.predict(test_df[features])\n        \n        val_pred = (1-ff)*val_pred_1+ff*val_pred_2\n        val_pred = cp.asnumpy(val_pred.values.flatten())\n        \n        test_pred = (1-ff)*test_pred_1+ff*test_pred_2\n        test_pred = cp.asnumpy(test_pred.values.flatten())\n\n        y_oof[val_ind] = val_pred\n        y_test[:, f] = test_pred\n        \n    df["pred_{}".format(target)] = y_oof\n    test_df[target] = y_test.mean(axis=1)+pp\n    \n    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)\n    mae = mean_absolute_error(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)\n    rmse = np.sqrt(mean_squared_error(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values))\n    overal_score += w*score\n    print(target, np.round(score, 8))\n    print(target, np.round(score, 4))\n    print(target, \'mean absolute error:\', np.round(mae, 8))\n    print(target, \' root mean square error:\', np.round(rmse, 8))\n    print()\n    \nprint("Overal score:", np.round(overal_score, 8))\nprint("Overal score:", np.round(overal_score, 4))')


# In[ ]:


sub_df = cudf.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
assert sub_df.shape[0] == test_df.shape[0]*5
sub_df.head(10)


# In[ ]:


sub_df.to_csv("submission_rapids_ensemble.csv", index=False)


# In[ ]:



