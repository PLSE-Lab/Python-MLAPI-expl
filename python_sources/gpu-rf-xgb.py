#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import gc

df = pd.read_csv("../input/remove-trends-giba-explained/train_clean_giba.csv").sort_values("time").reset_index(drop=True)
test_df = pd.read_csv("../input/remove-trends-giba-explained/test_clean_giba.csv").sort_values("time").reset_index(drop=True)


# In[ ]:


import multiprocessing
multiprocessing.cpu_count()


# In[ ]:


df["group"] = np.arange(df.shape[0])//500_000
df["mg"] = np.arange(df.shape[0])//100_000
df["group"].value_counts()


# In[ ]:


df["category"] = 0
test_df["category"] = 0

# train segments with more then 9 open channels classes
df.loc[2_000_000:2_500_000-1, 'category'] = 1
df.loc[4_500_000:5_000_000-1, 'category'] = 1

# test segments with more then 9 open channels classes (potentially)
test_df.loc[500_000:600_000-1, "category"] = 1
test_df.loc[700_000:800_000-1, "category"] = 1


# In[ ]:


TARGET = "open_channels"

aug_df = df[df["group"] == 5].copy()
aug_df["category"] = 1
aug_df["group"] = 10

for col in ["signal", TARGET]:
    aug_df[col] += df[df["group"] == 8][col].values
    
df = df.append(aug_df, sort=False)


# In[ ]:


df.groupby("group")["signal"].agg({"mean", "std"})


# In[ ]:


NUM_SHIFT = 20

features = ["signal", "signal"]

for i in range(1, NUM_SHIFT + 1):
    f_pos = "shift_{}".format(i)
    f_neg = "shift_{}".format(-i)
    features.append(f_pos)
    features.append(f_neg)
    for data in [df, test_df]:
        data[f_pos] = data["signal"].shift(i).fillna(-3) # Groupby shift!!!
        data[f_neg] = data["signal"].shift(-i).fillna(-3) # Groupby shift!!!
        
data.head()


# In[ ]:


#         model = RandomForestClassifier(
#                 n_estimators=40,
#                 max_samples=0.5,
#                 max_depth=17,
#                 max_features=10,
#                 min_samples_leaf=10,
#                 random_state=42,
#                 n_jobs=-1,
#                 verbose=1
#             )


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import StratifiedKFold\nimport xgboost as xgb\n\nNUM_FOLDS = 5\nskf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)\n\noof_preds = np.zeros((len(df), 11))\ny_test = np.zeros((len(test_df), 11))\n\nfor fold, (train_ind, val_ind) in enumerate(skf.split(df, df["group"])):\n    train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]\n    print(fold, len(train_df), len(val_df))\n\n    for cat in range(2):\n        fit_df = train_df[train_df["category"] == cat]\n        y = fit_df[TARGET].values\n        y[0] = 0 # hack\n        #model.fit(fit_df[features], y)\n\n        model = xgb.XGBRFClassifier( \n            n_estimators = 40,\n            learning_rate=1,\n            subsample=0.50,\n            colsample_bynode=0.25,\n            reg_lambda=1e-05,\n            objective= \'multi:softmax\',\n            num_class= len(np.unique(y)),\n            max_depth = 17,\n            num_parallel_tree = 1,\n            tree_method = \'gpu_hist\',\n            n_jobs = 2,\n            verbosity = 0,\n            predictor = \'gpu_predictor\',\n           ).fit( fit_df[features].values, y )\n        \n        pred = model.predict_proba(val_df[val_df["category"] == cat][features].values)\n        oof_preds[val_ind[np.where(val_df["category"].values == cat)[0]], :pred.shape[1]] = pred\n        \n        y_test[np.where(test_df["category"].values == cat)[0], :pred.shape[1]] += model.predict_proba(test_df[test_df["category"] == cat][features].values)/NUM_FOLDS\n        del model; _=gc.collect()\n    ')


# In[ ]:


from sklearn.metrics import f1_score

f1_score(df["open_channels"], oof_preds.argmax(axis=1), average="macro")


# In[ ]:


test_df[TARGET] = y_test.argmax(axis=1)

test_df.iloc[:600_000][TARGET].value_counts()/600_000


# In[ ]:


np.save("y_oof.npy", oof_preds)
np.save("y_test.npy", y_test)

test_df.to_csv('submission.csv', index=False, float_format='%.4f', columns=["time", TARGET])


# In[ ]:




