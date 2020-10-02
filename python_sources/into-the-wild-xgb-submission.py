#!/usr/bin/env python
# coding: utf-8

# ### This work totally belongs to [Ahmet](https://www.kaggle.com/aerdem4), kudos should go to him :)
# #### Clean Dataset;
# I used Giba's clean dataset from a private notebook. You can use Giba's public notebook for the same clean dataset;
# https://www.kaggle.com/titericz/remove-trends-giba-explained
# 
# #### Scripts to Create OOFs;
# ##### https://www.kaggle.com/meminozturk/into-the-wild-rfc-classification
# ##### https://www.kaggle.com/meminozturk/into-the-wild-mlp-regression/
# ##### https://www.kaggle.com/meminozturk/into-the-wild-lgb-regression
# 
# #### Wavenet Model;
# ##### https://www.kaggle.com/meminozturk/into-the-wild-wavenet/

# In[ ]:


import numpy as np
import pandas as pd
import gc

df = pd.read_csv("/kaggle/input/remove-trends-giba/train_clean_giba.csv", usecols=["signal","open_channels"], dtype={'signal': np.float32, 'open_channels':np.int32})
test_df  = pd.read_csv("/kaggle/input/remove-trends-giba/test_clean_giba.csv", usecols=["signal"], dtype={'signal': np.float32})

df.shape, test_df.shape


# In[ ]:


df['group'] = np.arange(df.shape[0])//500_000
aug_df = df[df["group"] == 5].copy()
aug_df["group"] = 10

for col in ["signal", "open_channels"]:
    aug_df[col] += df[df["group"] == 8][col].values

df = df.append(aug_df, sort=False).reset_index(drop=True)
df.shape

del aug_df
gc.collect()


# In[ ]:


wavenet_oof = (np.load("/kaggle/input/into-the-wild-wavenet/wavenet.npz")["valid"] + np.load("/kaggle/input/into-the-wild-wavenet/wavenet.npz")["tta"])/2
wavenet_test = np.load("/kaggle/input/into-the-wild-wavenet/wavenet.npz")["test"]

for i in range(wavenet_oof.shape[1]):
    df["prob_{}".format(i)] = wavenet_oof[:, i]
    test_df["prob_{}".format(i)] = wavenet_test[:, i]
    
df["wave_pred"] = np.argmax(wavenet_oof, axis=1)
test_df["wave_pred"] = np.argmax(wavenet_test, axis=1)

df['noise'] = df['signal'].values - df['wave_pred'].values
test_df['noise'] = test_df['signal'].values - test_df['wave_pred'].values


# In[ ]:


df.sample(5, random_state=0)


# In[ ]:


def get_margin(x):
    return np.log(x)

M = get_margin(wavenet_oof)
M_test = get_margin(wavenet_test)


# In[ ]:


from sklearn.metrics import f1_score, log_loss

f1_score(df["open_channels"], df["wave_pred"], average="macro")


# In[ ]:


log_loss(df["open_channels"], wavenet_oof)


# In[ ]:


NUM_FOLDS = 5

df["mg"] = df.index//100_000
test_df["mg"] = test_df.index//100_000

df["fold"] = df["mg"] % NUM_FOLDS


# In[ ]:


for data in [df, test_df]:
    y_time_since = np.empty((data.shape[0], 11))
    y_time_till = np.empty((data.shape[0], 11))
    y_pred = data["wave_pred"].values

    for sec in range(data.shape[0]//100_000):
        begin, end = sec*100_000, (sec+1)*100_000
        # print(begin, end)

        last_seen = np.array([np.nan]*11)
        for index in range(begin, end):
            y_time_since[index] = index - last_seen
            last_seen[y_pred[index]] = index

        last_seen = np.array([np.nan]*11)
        for index in reversed(range(begin, end)):
            y_time_till[index] = last_seen - index
            last_seen[y_pred[index]] = index

    for i in range(11):
        f = "time_since_{}".format(i)
        data[f] = y_time_since[:, i]
        data[f] = np.clip(data[f].fillna(np.inf), 0, 1_000)

        f = "time_till_{}".format(i)
        data[f] = y_time_till[:, i]
        data[f] = np.clip(data[f].fillna(np.inf), 0, 1_000)
    
test_df.sample(2).T


# In[ ]:


features = ["signal", "noise",
            "prob_0", "prob_1", "prob_2", "prob_3", "prob_4", "prob_5", "prob_6", "prob_7", "prob_8", "prob_9", "prob_10"]

for i in range(11):
    f = "time_since_{}".format(i)
    features.append(f)
    f = "time_till_{}".format(i)
    features.append(f)


# In[ ]:


import xgboost as xgb

params = {"objective": "multi:softprob",
          "num_class": 11,
          "learning_rate" : 0.2,
          "max_leaves": 2**4,
          "grow_policy": "lossguide",
          'min_child_weight': 50,
          'lambda': 2,
          'eval_metric': 'mlogloss',
          "base_score": 0,
          "tree_method": 'gpu_hist', "gpu_id": 0
         }


target = "open_channels"

y_oof = np.zeros_like(wavenet_oof)
y_test = np.zeros_like(wavenet_test)

del  wavenet_oof, wavenet_test
gc.collect()

X_test = test_df[features].values

d_test = xgb.DMatrix(X_test)
d_test.set_base_margin(M_test.flatten())

for f in range(NUM_FOLDS):
    train_df, val_df = df[df["fold"] != f].copy(), df[df["fold"] == f].copy()
    train_ind = np.where(df["fold"].values != f)[0]
    val_ind = np.where(df["fold"].values == f)[0]
    
    X_train, X_val = train_df[features].values, val_df[features].values
    y_train, y_val = train_df[target].values, val_df[target].values
    
    d_train = xgb.DMatrix(X_train, y_train)
    d_train.set_base_margin(M[train_ind].flatten())

    d_val = xgb.DMatrix(X_val, y_val)
    d_val.set_base_margin(M[val_ind].flatten())
    
    del X_train, X_val, y_train, y_val
    gc.collect()
    
    model = xgb.train(params, d_train, evals=[(d_train, "train"), (d_val, "eval")], verbose_eval=10, num_boost_round=31)
    y_oof[val_ind] = model.predict(d_val)
    y_test += model.predict(d_test)/NUM_FOLDS
    print()
    
del d_train, d_val
gc.collect()


# In[ ]:


log_loss(df["open_channels"], y_oof)


# In[ ]:


df["xgb_pred"] = np.argmax(y_oof, axis=1)

f1_score(df["open_channels"], df["xgb_pred"], average="macro")


# In[ ]:


f1_score(df.iloc[:5_000_000]["open_channels"], df.iloc[:5_000_000]["xgb_pred"], average="macro")


# In[ ]:


df["ensemble_pred"] = df[["wave_pred", "xgb_pred"]].max(axis=1)

f1_score(df["open_channels"], df["ensemble_pred"], average="macro")


# In[ ]:


f1_score(df.iloc[:5_000_000]["open_channels"], df.iloc[:5_000_000]["ensemble_pred"], average="macro")


# In[ ]:


test_df["xgb_pred"] = np.argmax(y_test, axis=1)


# In[ ]:


sample_submission  = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})
sample_submission['open_channels'] = test_df[["wave_pred", "xgb_pred"]].max(axis=1)
sample_submission.to_csv(f'submission.csv', index=False, float_format='%.4f')
print(sample_submission.open_channels.mean())
display(sample_submission.head())


# In[ ]:




