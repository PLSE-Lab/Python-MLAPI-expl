#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import gc

df = pd.read_csv("../input/remove-trends-giba/train_clean_giba.csv").sort_values("time").reset_index(drop=True)
test_df = pd.read_csv("../input/remove-trends-giba/test_clean_giba.csv").sort_values("time").reset_index(drop=True)


# In[ ]:


df["group"] = np.arange(df.shape[0])//500_000
df["batch"] = np.arange(df.shape[0])//100_000
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
    
df = df.append(aug_df, sort=False).reset_index(drop=True)

del aug_df
gc.collect()


# In[ ]:


df.groupby("group")["signal"].agg({"mean", "std"})


# In[ ]:


df['batch'] = np.arange(df.shape[0])//100_000
test_df['batch'] = np.arange(test_df.shape[0])//100_000

shift_sizes = np.arange(1,21)
for temp in [df,test_df]:
    for shift_size in shift_sizes:    
        temp['signal_shift_pos_'+str(shift_size)] = temp.groupby('batch')['signal'].shift(shift_size).fillna(-3)
        # temp['signal_shift_pos_'+str(shift_size)] = temp.groupby("batch")['signal_shift_pos_'+str(shift_size)].transform(lambda x: x.bfill())
        temp['signal_shift_neg_'+str(shift_size)] = temp.groupby('batch')['signal'].shift(-1*shift_size).fillna(-3)
        # temp['signal_shift_neg_'+str(shift_size)] = temp.groupby("batch")['signal_shift_neg_'+str(shift_size)].transform(lambda x: x.ffill())


# In[ ]:


remove_fea=['time','batch','batch_index','batch_slices','batch_slices2','group',"open_channels","type","category"]
features=[i for i in df.columns if i not in remove_fea]
df[features].head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold

NUM_FOLDS = 5
oof_preds = np.zeros((len(df), 11))
y_test = np.zeros((len(test_df), 11))

target = "open_channels"
df['group'] = np.arange(df.shape[0])//4000
group = df['group']
kf = GroupKFold(n_splits=NUM_FOLDS)
splits = [x for x in kf.split(df, df["open_channels"], group)]
            
for train_ind, val_ind in splits:
    train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]
    print(len(train_df), len(val_df))

    for cat in range(2):
        model = RandomForestClassifier(
                n_estimators=150,
                max_samples=0.5,
                max_depth=17,
                max_features=10,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        
        fit_df = train_df[train_df["category"] == cat]
        y = fit_df[TARGET].values
        y[y.argmin()] = 0 # hack to have 11 class in each fold
        
        model.fit(fit_df[features], y)
        
        pred = model.predict_proba(val_df[val_df["category"] == cat][features])
        oof_preds[val_ind[np.where(val_df["category"].values == cat)[0]], :pred.shape[1]] = pred
        
        y_test[np.where(test_df["category"].values == cat)[0], :pred.shape[1]] += model.predict_proba(test_df[test_df["category"] == cat][features])/NUM_FOLDS


# In[ ]:


from sklearn.metrics import f1_score

f1_score(df["open_channels"], oof_preds.argmax(axis=1), average="macro")


# In[ ]:


oof_f1 = f1_score(df["open_channels"].iloc[:5000_000], oof_preds[:5000_000].argmax(axis=1), average="macro")
oof_f1


# In[ ]:


test_df[TARGET] = y_test.argmax(axis=1)
test_df.iloc[:600_000][TARGET].value_counts()/600_000


# In[ ]:


np.savez_compressed('rfc_clf.npz',valid=oof_preds, test=y_test)
test_df.to_csv(f'submission.csv', index=False, float_format='%.4f', columns=["time", TARGET])
print(test_df["open_channels"].mean())
test_df["open_channels"].value_counts()

