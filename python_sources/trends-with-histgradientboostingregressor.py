#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.model_selection import KFold


def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")
loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")


labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()
df = df[df["is_train"] == True].copy()

df.shape, test_df.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nNUM_FOLDS = 5\nkf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)\n\nfeatures = loading_features + fnc_features\n\noveral_score = 0\nfor target, c, w in [("age", 110, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]: \n    print(f\'Target: {target}\')\n    y_oof = np.zeros(df.shape[0])\n    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))\n    \n    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):\n        print(f\'Fold {f}\')\n        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]\n        train_df = train_df[train_df[target].notnull()]\n        \n\n        \n        model = HistGradientBoostingRegressor(validation_fraction=None, max_iter=250, l2_regularization=0.2, min_samples_leaf=25, max_depth=15)\n        model.fit(train_df[features], train_df[target])\n        temp_oof = model.predict(val_df[features])\n        temp_test = model.predict(test_df[features])\n\n        y_oof[val_ind] = temp_oof\n        y_test[:, f] = temp_test\n        \n    df["pred_{}".format(target)] = y_oof\n    test_df[target] = y_test.mean(axis=1)\n    \n    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)\n    overal_score += w*score\n    print(target, np.round(score, 10))\n    print()\n    \nprint("Overal score:", np.round(overal_score, 10))')


# In[ ]:


sub_df = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
assert sub_df.shape[0] == test_df.shape[0]*5
sub_df.head(10)


# In[ ]:


sub_df.to_csv("submission_hgbr.csv", index=False)

