#!/usr/bin/env python
# coding: utf-8

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
import pandas as pd
import cudf 
import cupy as cp

from cuml.neighbors import KNeighborsRegressor
from cuml import SVR
from cuml.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingRegressor
from cuml.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA


def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))


# In[ ]:


import cuml


# In[ ]:


fnc = cudf.read_csv('../input/trends-assessment-prediction/fnc.csv')
loading = cudf.read_csv('../input/trends-assessment-prediction/loading.csv')
labels = cudf.read_csv('../input/trends-assessment-prediction/train_scores.csv')
sites = cudf.read_csv('../input/trends-assessment-prediction/reveal_ID_site2.csv')


# In[ ]:


loading = loading.drop(['IC_20'], axis=1)

fnc_features, loading_features = list(fnc.columns[1:]), list(loading.columns[1:])
df = fnc.merge(loading, on="Id")


# In[ ]:


#sites = np.array(sites).reshape(sites.shape[0])
site1 = df[~df['Id'].isin(set(sites))]
site2 = df[df['Id'].isin(set(sites))]

site1['Label'] = 0
site2['Label'] = 1

df = cudf.concat([site1, site2], axis=0)
X = df.iloc[:, :-1]
y = df.loc[:, 'Label']


# In[ ]:


labels["is_train"] = True 
df = df.merge(labels, on="Id", how="left")

test_df = df[df["is_train"] != True].copy() 
df = df[df["is_train"] == True].copy()


# In[ ]:


test_df.head()


# In[ ]:


# Take Id of train set:
id_df = cudf.DataFrame()
id_df["Id"] = df["Id"]

# Take Id of test set:
id_test_df = cudf.DataFrame()
id_test_df["Id"] = test_df["Id"]


df.shape, test_df.shape, id_df.shape, id_test_df.shape


# In[ ]:


data_fnc = df.loc[:, df.columns[1:1378]]
#data_loading = df.loc[:, df.columns[1379:1405]]

n_components_fnc = 20

reducer_fnc = cuml.UMAP(
    n_neighbors=15,
    n_components=n_components_fnc,
    n_epochs=500,
    min_dist=0.1
)
emb_fnc = reducer_fnc.fit_transform(data_fnc)


# n_components_loading = 20
# 
# reducer_loading = cuml.UMAP(
#     n_neighbors=15,
#     n_components=n_components_loading,
#     n_epochs=500,
#     min_dist=0.1
# )
# emb_loading = reducer_loading.fit_transform(data_loading)

# In[ ]:


id_df["key"] = emb_fnc["key"] = [x for x in range(len(emb_fnc))]
df = id_df.merge(emb_fnc, on="key")
#df = df.merge(emb_loading, on="key")

df = df.drop("key")
df = df.merge(loading, on="Id")
df = df.merge(labels, on="Id")

#df = cudf.concat([site1, site2], axis=0)


# In[ ]:


df.shape


# In[ ]:


data_fnc = test_df.loc[:, test_df.columns[1:1378]]
data_loading = test_df.loc[:, test_df.columns[1379:1405]]

#emb_fnc = reducer_fnc.transform(data_fnc)
#emb_loading = reducer_loading.transform(data_loading)


# In[ ]:


id_test_df["key"] = emb_fnc["key"] = [x for x in range(len(test_df))]
test_df = id_test_df.merge(emb_fnc, on="key")
#test_df = test_df.merge(loading,)
test_df = test_df.drop("key")


# In[ ]:


test_df.head()


# In[ ]:


df = df.astype("float32").copy()
test_df = test_df.astype("float32").copy()

df["Id"] = df["Id"].astype(int)
test_df["Id"] = test_df["Id"].astype(int)


# In[ ]:


df.columns[1:-6]


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# To suppress the "Expected column (\'F\') major order, but got the opposite." warnings from cudf. It should be fixed properly,\n# although as the only impact is additional memory usage, I\'ll supress it for now.\n\n# Take a copy of the main dataframe, to report on per-target scores for each model.\n# TODO Copy less to make this more efficient.\ndf_model1 = df.copy()\ndf_model2 = df.copy()\ndf_model3 = df.copy()\n\nNUM_FOLDS = 7\nkf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)\n\nfeatures = [col for col in df.columns[1:-6]]\n\n# Blending weights between the three models are specified separately for the 5 targets. \n#                                 SVR,  Ridge, BaggingRegressor\nblend_weights = {"age":          [0.4,  0.55,  0.05],\n                 "domain1_var1": [0.55, 0.15,  0.3],\n                 "domain1_var2": [0.45, 0.0,   0.55],\n                 "domain2_var1": [0.55, 0.15,  0.3],\n                 "domain2_var2": [0.5,  0.05,  0.45]}\n\noverall_score = 0\nfor target, c, w in [("age", 60, 0.3), ("domain1_var1", 12, 0.175), ("domain1_var2", 8, 0.175), ("domain2_var1", 9, 0.175), ("domain2_var2", 12, 0.175)]:    \n    y_oof = np.zeros(df.shape[0])\n    y_oof_model_1 = np.zeros(df.shape[0])\n    y_oof_model_2 = np.zeros(df.shape[0])\n    y_oof_model_3 = np.zeros(df.shape[0])\n    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))\n    \n    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):\n        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]\n        train_df = train_df[train_df[target].notnull()]\n\n        model_1 = SVR(C=c, cache_size=3000.0)\n        model_1.fit(train_df[features].values, train_df[target].values)\n        model_2 = Ridge(alpha = 0.0001)\n        model_2.fit(train_df[features].values, train_df[target].values)\n        \n        ### The BaggingRegressor, using the Ridge regression method as a base, is added here. The BaggingRegressor\n        # is from sklearn, not RAPIDS, so dataframes need converting to Pandas.\n        model_3 = BaggingRegressor(Ridge(alpha = 0.0001), n_estimators=30, random_state=42, max_samples=0.3, max_features=0.3)\n        model_3.fit(train_df.to_pandas()[features].values, train_df.to_pandas()[target].values)\n\n        val_pred_1 = model_1.predict(val_df[features])\n        val_pred_2 = model_2.predict(val_df[features])\n        val_pred_3 = model_3.predict(val_df.to_pandas()[features])\n        val_pred_3 = cudf.from_pandas(pd.Series(val_pred_3))\n        \n        test_pred_1 = model_1.predict(test_df[features])\n        test_pred_2 = model_2.predict(test_df[features])\n        test_pred_3 = model_3.predict(test_df.to_pandas()[features])\n        test_pred_3 = cudf.from_pandas(pd.Series(test_pred_3))\n        \n        val_pred = blend_weights[target][0]*val_pred_1+blend_weights[target][1]*val_pred_2+blend_weights[target][2]*val_pred_3\n        val_pred = cp.asnumpy(val_pred.values.flatten())\n        \n        test_pred = blend_weights[target][0]*test_pred_1+blend_weights[target][1]*test_pred_2+blend_weights[target][2]*test_pred_3\n        test_pred = cp.asnumpy(test_pred.values.flatten())\n        \n        y_oof[val_ind] = val_pred\n        y_oof_model_1[val_ind] = val_pred_1\n        y_oof_model_2[val_ind] = val_pred_2\n        y_oof_model_3[val_ind] = val_pred_3\n        y_test[:, f] = test_pred\n        \n    df["pred_{}".format(target)] = y_oof\n    df_model1["pred_{}".format(target)] = y_oof_model_1\n    df_model2["pred_{}".format(target)] = y_oof_model_2\n    df_model3["pred_{}".format(target)] = y_oof_model_3\n    test_df[target] = y_test.mean(axis=1)\n    \n    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)\n    overall_score += w*score\n    \n    score_model1 = metric(df_model1[df_model1[target].notnull()][target].values, df_model1[df_model1[target].notnull()]["pred_{}".format(target)].values)\n    score_model2 = metric(df_model2[df_model2[target].notnull()][target].values, df_model2[df_model1[target].notnull()]["pred_{}".format(target)].values)\n    score_model3 = metric(df_model3[df_model3[target].notnull()][target].values, df_model3[df_model1[target].notnull()]["pred_{}".format(target)].values)\n\n    print(f"For {target}:")\n    print("SVR:", np.round(score_model1, 6))\n    print("Ridge:", np.round(score_model2, 6))\n    print("BaggingRegressor:", np.round(score_model3, 6))\n    print("Ensemble:", np.round(score, 6))\n    print()\n    \nprint("Overall score:", np.round(overall_score, 6))')


# %%time
# 
# NUM_FOLDS = 7
# kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)
# 
# 
# features = [col for col in df.columns[1:-6]]
# 
# overal_score = 0
# for target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:    
#     y_oof = np.zeros(df.shape[0])
#     y_test = np.zeros((test_df.shape[0], NUM_FOLDS))
#     
#     for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):
#         train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]
#         train_df = train_df[train_df[target].notnull()]
# 
#         model = SVR(C=c, cache_size=3000.0)
#         model.fit(train_df[features], train_df[target])
# 
#         y_oof[val_ind] = model.predict(val_df[features])
#         y_test[:, f] = model.predict(test_df[features])
#         
#     df["pred_{}".format(target)] = y_oof
#     test_df[target] = y_test.mean(axis=1)
#     
#     score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)
#     overal_score += w*score
#     print(target, np.round(score, 4))
#     print()
#     
# print("Overal score:", np.round(overal_score, 4))

# sub_df = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
# sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")
# 
# sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
# assert sub_df.shape[0] == test_df.shape[0]*5
# sub_df.head(10)

# In[ ]:


sub_df = cudf.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
assert sub_df.shape[0] == test_df.shape[0]*5
sub_df.head(10)


# In[ ]:


sub_df.to_csv("submission_rapids_ensemble.csv", index=False)

