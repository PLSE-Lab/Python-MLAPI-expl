#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this competition a significant delta between local CV and LB scores has been reported in some cases (https://www.kaggle.com/c/trends-assessment-prediction/discussion/153256). We have many features to work with... maybe too many. Reducing variance would seem to be a good thing here and I wanted to investigate the BaggingRegressor for that. The idea is to use the BaggingRegressor to build multiple models, each considering only a fraction of the features, then combine their outputs. From the scikit-learn docs:
# 
# "A Bagging regressor is an ensemble meta-estimator that fits base regressors each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it."
# 
# Ridge regression is known to work well on this dataset, so is used as the base regressor here. The use of the BaggingRegressor is considered as part of a high-performing ensemble, combining SVM and Ridge regression.
# 
# This notebook is heavily based on @aerdem4's excellent SVM notebook and @tunguz's notebook that adds Ridge regression. Those original notebooks can be found here:
# https://www.kaggle.com/aerdem4/rapids-svm-on-trends-neuroimaging
# https://www.kaggle.com/tunguz/rapids-ensemble-for-trends-neuroimaging/
# 
# ## Results
# 
# After doing an offline sweep of blending weights, the final weights show that for the best local CV, the BaggingRegressor was hardly used for the "age" target. However, the BaggingRegressor provided more benefits for the domain variables. In particular for "domain1_var2" and "domain2_var2" the BaggingRegressor almost completely replaces the basic Ridge regression method.
# 
# In terms of local CV, the result is almost identical to Bojan's notebook referenced above. On the leaderboard, adding the BaggingRegressor into the ensemble scores 0.1593, an improvement over Bojan's 0.1595. So the local CV to LB delta is successfully reduced, albeit by a little.
# 
# I find it particularly interesting that only considering small subsets of the features, the BaggingRegressor is competitive for the domain variables but not at all for age.
# 

# # Load the data

# In[ ]:


# Install Rapids for faster SVM on GPUs

import sys
get_ipython().system('cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz')
get_ipython().system('cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null')
sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path
get_ipython().system('cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


import numpy as np
import pandas as pd
import cudf
import cupy as cp
import warnings
from cuml.neighbors import KNeighborsRegressor
from cuml import SVR
from cuml.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingRegressor

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


# In[ ]:


# Giving less importance to FNC features since they are easier to overfit due to high dimensionality.
FNC_SCALE = 1/600

df[fnc_features] *= FNC_SCALE
test_df[fnc_features] *= FNC_SCALE


# # BaggingRegressor + RAPIDS Ensemble

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# To suppress the "Expected column (\'F\') major order, but got the opposite." warnings from cudf. It should be fixed properly,\n# although as the only impact is additional memory usage, I\'ll supress it for now.\nwarnings.filterwarnings("ignore", message="Expected column")\n\n# Take a copy of the main dataframe, to report on per-target scores for each model.\n# TODO Copy less to make this more efficient.\ndf_model1 = df.copy()\ndf_model2 = df.copy()\ndf_model3 = df.copy()\n\nNUM_FOLDS = 7\nkf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)\n\nfeatures = loading_features + fnc_features\n\n# Blending weights between the three models are specified separately for the 5 targets. \n#                                 SVR,  Ridge, BaggingRegressor\nblend_weights = {"age":          [0.4,  0.55,  0.05],\n                 "domain1_var1": [0.55, 0.15,  0.3],\n                 "domain1_var2": [0.45, 0.0,   0.55],\n                 "domain2_var1": [0.55, 0.15,  0.3],\n                 "domain2_var2": [0.5,  0.05,  0.45]}\n\noverall_score = 0\nfor target, c, w in [("age", 60, 0.3), ("domain1_var1", 12, 0.175), ("domain1_var2", 8, 0.175), ("domain2_var1", 9, 0.175), ("domain2_var2", 12, 0.175)]:    \n    y_oof = np.zeros(df.shape[0])\n    y_oof_model_1 = np.zeros(df.shape[0])\n    y_oof_model_2 = np.zeros(df.shape[0])\n    y_oof_model_3 = np.zeros(df.shape[0])\n    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))\n    \n    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):\n        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]\n        train_df = train_df[train_df[target].notnull()]\n\n        model_1 = SVR(C=c, cache_size=3000.0)\n        model_1.fit(train_df[features].values, train_df[target].values)\n        model_2 = Ridge(alpha = 0.0001)\n        model_2.fit(train_df[features].values, train_df[target].values)\n        \n        ### The BaggingRegressor, using the Ridge regression method as a base, is added here. The BaggingRegressor\n        # is from sklearn, not RAPIDS, so dataframes need converting to Pandas.\n        model_3 = BaggingRegressor(Ridge(alpha = 0.0001), n_estimators=30, random_state=42, max_samples=0.3, max_features=0.3)\n        model_3.fit(train_df.to_pandas()[features].values, train_df.to_pandas()[target].values)\n\n        val_pred_1 = model_1.predict(val_df[features])\n        val_pred_2 = model_2.predict(val_df[features])\n        val_pred_3 = model_3.predict(val_df.to_pandas()[features])\n        val_pred_3 = cudf.from_pandas(pd.Series(val_pred_3))\n        \n        test_pred_1 = model_1.predict(test_df[features])\n        test_pred_2 = model_2.predict(test_df[features])\n        test_pred_3 = model_3.predict(test_df.to_pandas()[features])\n        test_pred_3 = cudf.from_pandas(pd.Series(test_pred_3))\n        \n        val_pred = blend_weights[target][0]*val_pred_1+blend_weights[target][1]*val_pred_2+blend_weights[target][2]*val_pred_3\n        val_pred = cp.asnumpy(val_pred.values.flatten())\n        \n        test_pred = blend_weights[target][0]*test_pred_1+blend_weights[target][1]*test_pred_2+blend_weights[target][2]*test_pred_3\n        test_pred = cp.asnumpy(test_pred.values.flatten())\n        \n        y_oof[val_ind] = val_pred\n        y_oof_model_1[val_ind] = val_pred_1\n        y_oof_model_2[val_ind] = val_pred_2\n        y_oof_model_3[val_ind] = val_pred_3\n        y_test[:, f] = test_pred\n        \n    df["pred_{}".format(target)] = y_oof\n    df_model1["pred_{}".format(target)] = y_oof_model_1\n    df_model2["pred_{}".format(target)] = y_oof_model_2\n    df_model3["pred_{}".format(target)] = y_oof_model_3\n    test_df[target] = y_test.mean(axis=1)\n    \n    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)\n    overall_score += w*score\n    \n    score_model1 = metric(df_model1[df_model1[target].notnull()][target].values, df_model1[df_model1[target].notnull()]["pred_{}".format(target)].values)\n    score_model2 = metric(df_model2[df_model2[target].notnull()][target].values, df_model2[df_model1[target].notnull()]["pred_{}".format(target)].values)\n    score_model3 = metric(df_model3[df_model3[target].notnull()][target].values, df_model3[df_model1[target].notnull()]["pred_{}".format(target)].values)\n\n    print(f"For {target}:")\n    print("SVR:", np.round(score_model1, 6))\n    print("Ridge:", np.round(score_model2, 6))\n    print("BaggingRegressor:", np.round(score_model3, 6))\n    print("Ensemble:", np.round(score, 6))\n    print()\n    \nprint("Overall score:", np.round(overall_score, 6))')


# In[ ]:


sub_df = cudf.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
assert sub_df.shape[0] == test_df.shape[0]*5
sub_df.head(10)


# In[ ]:


sub_df.to_csv("submission_rapids_ensemble_with_baggingregressor.csv", index=False)

