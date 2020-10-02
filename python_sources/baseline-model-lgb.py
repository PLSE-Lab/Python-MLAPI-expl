#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[ ]:


import os, time, sys, gc
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#wave analysis
import pywt 
from statsmodels.robust import mad
import scipy
from scipy import stats 
from scipy import signal
from scipy.signal import hann, hilbert, convolve, butter, deconvolve
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold

import lightgbm as lgb
from sklearn.metrics import f1_score


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


INPUTDIR = '/kaggle/input/liverpool-ion-switching/'
NROWS = None
SHIFT = 1
WINDOW = [10, 25, 50, 100, 500, 1000, 5000, 10000, 25000]
n_fold = 3
ERSR = 500
DISP = 1000
num_round = 5000
folds = KFold(n_splits=n_fold, shuffle=True, random_state=6666)


# ## Import Data

# In[ ]:


get_ipython().run_cell_magic('time', '', "df_train = pd.read_csv(f'{INPUTDIR}/train.csv', nrows=NROWS, dtype={'time':np.float32, 'signal':np.float32})\ndf_test = pd.read_csv(f'{INPUTDIR}/test.csv', nrows=NROWS, dtype={'time':np.float32, 'signal':np.float32})\nsub_df = pd.read_csv(f'{INPUTDIR}/sample_submission.csv', nrows=NROWS)")


# In[ ]:


print(df_train.columns)
print(df_test.columns)
print(df_train.shape)
print(df_test.shape)


# ## Define Functions

# ### Rolling Features

# In[ ]:


def rolling_feature(df, sft, window_sizes):
    for window in window_sizes:
        df["roll_sum_" + str(window)] = df['signal'].rolling(window=window).sum()
        df["roll_mean_" + str(window)] = df['signal'].rolling(window=window).mean()
                    
        df["roll_std_" + str(window)] = df['signal'].rolling(window=window).std()
        df["roll_max2min_" + str(window)] = df['signal'].rolling(window=window).max()/            df['signal'].rolling(window=window).min()
        df["roll_max_diff_min_" + str(window)] = df['signal'].rolling(window=window).max()-            df['signal'].rolling(window=window).min()
        
        df["roll_mean_diffwt2_" + str(window)] = df['signal'].rolling(window=window).mean()-            df['signal'].shift(sft*2).rolling(window=window).mean()
        
        df["roll_mean_diffwt3_" + str(window)] = df['signal'].rolling(window=window).mean()-            df['signal'].shift(sft*3).rolling(window=window).mean()
        
        df["roll_skew_" + str(window)] = df['signal'].rolling(window=window).skew()
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_train = rolling_feature(df_train, SHIFT, WINDOW)\ndf_test = rolling_feature(df_test, SHIFT, WINDOW)')


# In[ ]:


df_train.head(10)


# In[ ]:


drop_list = [ 'time', 'signal', 'open_channels']
features = [c for c in df_train.columns if c not in drop_list]
target = df_train['open_channels']
len(features)


# In[ ]:


param_lgb = {'num_leaves': 128,
          'min_data_in_leaf': 64,
          'objective': 'huber',
          'max_depth': -1,
          'learning_rate': 0.1,
          'boosting': 'gbdt',
          'bagging_freq': 5,
          'bagging_fraction': 0.8,
          'bagging_seed': 66,
          'metric': 'mae',
          'verbosity': 1,
          'nthread': -1,
          'random_state': 6666}


# In[ ]:


sts_time = time.time()
oof = np.zeros(len(df_train))
#train_pred = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

#run model
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train)):
    strLog = "Fold {}".format(fold_+1)
    print(strLog ,'started at', time.ctime())
    
    
    X_tr, X_val = df_train.iloc[trn_idx][features], df_train.iloc[val_idx][features]
    y_tr, y_val = target.iloc[trn_idx], target.iloc[val_idx]
    
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(param_lgb, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=DISP, early_stopping_rounds = ERSR)
    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    
    #feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = model.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    #predictions
    #train_pred += model.predict(df_train[features], num_iteration=model.best_iteration) / folds.n_splits
    predictions += model.predict(df_test[features], num_iteration=model.best_iteration) / folds.n_splits
    
    ed_time = time.time()
    calc_time = ed_time - sts_time
    print('Calc Time : %.2f [sec]' % calc_time)

end_time = time.time()
calc_time = end_time - sts_time
print('Calc Time : %.2f [sec]' % calc_time)


# In[ ]:


cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:200].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()


# ## Submission

# In[ ]:


sub_df['open_channels'] = np.round(predictions).astype(np.int)
CVscore = f1_score(target, np.round(oof).astype(np.int), labels=None, pos_label=1, average='macro',  zero_division='warn') 
print('CV Score is: {:.3f}'.format(CVscore))


# In[ ]:


sub_df.to_csv("submission.csv", index=False, float_format='%.4f')

