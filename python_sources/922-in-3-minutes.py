#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import logit

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
features = [x for x in train_df.columns if x.startswith("var")]

hist_df = pd.DataFrame()
for var in features:
    var_stats = train_df[var].append(test_df[var]).value_counts()
    hist_df[var] = pd.Series(test_df[var]).map(var_stats)
    hist_df[var] = hist_df[var] > 1

ind = hist_df.sum(axis=1) != 200
var_stats = {var:train_df[var].append(test_df[ind][var]).value_counts() for var in features}

pred = 0
for var in features:
    model = lgb.LGBMClassifier(**{
        'learning_rate':0.05, 'max_bin': 165, 'max_depth': 5, 'min_child_samples': 150,
        'min_child_weight': 0.1, 'min_split_gain': 0.0018, 'n_estimators': 41,
        'num_leaves': 6, 'reg_alpha': 2.0, 'reg_lambda': 2.54, 'objective': 'binary', 'n_jobs': -1})
    model = model.fit(np.hstack([train_df[var].values.reshape(-1,1),
                                 train_df[var].map(var_stats[var]).values.reshape(-1,1)]),
                               train_df["target"].values)
    pred += logit(model.predict_proba(np.hstack([test_df[var].values.reshape(-1,1),
                                 test_df[var].map(var_stats[var]).values.reshape(-1,1)]))[:,1])
    
pd.DataFrame({"ID_code":test_df["ID_code"], "target":pred}).to_csv("submission.csv", index=False)

