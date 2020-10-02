#!/usr/bin/env python
# coding: utf-8

# ## Add this Dataset to you notebook:  https://www.kaggle.com/cdeotte/rapids
# ## Install https://rapids.ai/start.html

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
import gc
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import cudf
from cuml.ensemble import RandomForestRegressor


# In[ ]:


df      = pd.read_csv("../input/remove-trends-giba-explained/train_clean_giba.csv").sort_values("time").reset_index(drop=True)
test_df = pd.read_csv("../input/remove-trends-giba-explained/test_clean_giba.csv").sort_values("time").reset_index(drop=True)

df.signal        = df.signal.astype('float32')
df.open_channels = df.open_channels.astype('float32')

test_df.signal   = test_df.signal.astype('float32')


# In[ ]:


df["group"] = np.arange(df.shape[0])//500_000
df["mg"]    = np.arange(df.shape[0])//100_000
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

df['category']      = df['category'].astype( np.float32 )
test_df['category'] = test_df['category'].astype( np.float32 )


# In[ ]:


TARGET = "open_channels"

aug_df = df[df["group"] == 5].copy()
aug_df["category"] = 1
aug_df["group"] = 10

for col in ["signal", TARGET]:
    aug_df[col] += df[df["group"] == 8][col].values
    
aug_df['category'] = aug_df['category'].astype( np.float32 )
    

df = df.append(aug_df, sort=False)


# In[ ]:


NUM_SHIFT = 20

features = ["signal","signal","category"]

for i in range(1, NUM_SHIFT + 1):
    f_pos = "shift_{}".format(i)
    f_neg = "shift_{}".format(-i)
    features.append(f_pos)
    features.append(f_neg)
    for data in [df, test_df]:
        data[f_pos] = data["signal"].shift(i).fillna(-3).astype( np.float32 ) # Groupby shift!!!
        data[f_neg] = data["signal"].shift(-i).fillna(-3).astype( np.float32 ) # Groupby shift!!!
        
data.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nNUM_FOLDS = 5\nskf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)\n\ntest_df = cudf.from_pandas( test_df )\n\noof_preds = np.zeros((len(df)))\ny_test = np.zeros((len(test_df)))\nfor fold, (train_ind, val_ind) in enumerate(skf.split(df, df["group"])):\n    train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]\n    print(\'Fold\', fold )\n\n    train_df = cudf.from_pandas( train_df )\n    val_df   = cudf.from_pandas( val_df )\n\n    model = RandomForestRegressor(\n            n_estimators=35,\n            rows_sample = 0.35,\n            max_depth=18,\n            max_features=11,        \n            split_algo=0,\n            bootstrap=False, #Don\'t use repeated rows, this is important to set to False to improve accuracy\n        ).fit( train_df[features], train_df[TARGET] )\n        \n    pred = model.predict( val_df[features] ).to_array()\n    oof_preds[val_ind] = np.round( pred )\n        \n    y_test += model.predict( test_df[features] ).to_array() / NUM_FOLDS\n    del model; _=gc.collect()\n    \ny_test = np.round( y_test )')


# In[ ]:


f1_score( df["open_channels"], oof_preds, average="macro")


# In[ ]:


test_df['time'] = [ "{0:.4f}".format(v) for v in test_df['time'].to_array() ]
test_df[TARGET] = y_test.astype(np.int32)
test_df.to_csv('submission.csv', index=False, columns=["time", TARGET])


# In[ ]:


print( test_df[["time", TARGET]].tail() )


# In[ ]:




