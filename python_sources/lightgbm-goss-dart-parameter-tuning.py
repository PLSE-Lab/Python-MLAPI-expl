#!/usr/bin/env python
# coding: utf-8

# ## Thanks to this kernel:
# 
# 1. https://www.kaggle.com/fabiendaniel/elo-world
# 2. https://www.kaggle.com/mfjwr1/simple-lightgbm-without-blending

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge
warnings.simplefilter(action='ignore', category=FutureWarning)

from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
import gc
from tqdm import tqdm_notebook
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0']
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# rmse
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
    
# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')
    
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', "# load csv\ndef load_data():\n    train_df = pd.read_csv('../input/elo-blending/train_feature.csv')\n    test_df = pd.read_csv('../input/elo-blending/test_feature.csv')\n    display(train_df.head())\n    display(test_df.head())\n    print(train_df.shape,test_df.shape)\n    train = pd.read_csv('../input/elo-merchant-category-recommendation/train.csv', index_col=['card_id'])\n    test = pd.read_csv('../input/elo-merchant-category-recommendation/test.csv', index_col=['card_id'])\n    train_df['card_id'] = train.index\n    test_df['card_id'] = test.index\n    train_df.index = train_df['card_id']\n    test_df.index = test_df['card_id']\n    display(train_df.head())\n    display(test.head())\n    print(train.shape,test.shape)\n    del train,test    \n    return (train_df,test_df)\n\nprint(gc.collect())")


# In[ ]:


# train_df,test_df = load_data()


# In[ ]:


boosting = ["goss","dart"]
boosting[0],boosting[1]


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nboosting = ["goss","dart"]\n# LightGBM GBDT with KFold or Stratified KFold\ndef kfold_lightgbm(train_df, test_df, num_folds, stratified = False, boosting = boosting[0]):\n    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))\n\n    # Cross validation model\n    if stratified:\n        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=326)\n    else:\n        folds = KFold(n_splits= num_folds, shuffle=True, random_state=2045)\n\n    # Create arrays and dataframes to store results\n    oof_preds = np.zeros(train_df.shape[0])\n    sub_preds = np.zeros(test_df.shape[0])\n    feature_importance_df = pd.DataFrame()\n    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]\n\n    # k-fold\n    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df[\'outliers\'])):\n        train_x, train_y = train_df[feats].iloc[train_idx], train_df[\'target\'].iloc[train_idx]\n        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df[\'target\'].iloc[valid_idx]\n\n        # set data structure\n        lgb_train = lgb.Dataset(train_x,label=train_y,free_raw_data=False)\n        lgb_test = lgb.Dataset(valid_x,label=valid_y,free_raw_data=False)\n\n        # params optimized by optuna\n        params ={\n                        \'task\': \'train\',\n                        \'boosting\': \'goss\',\n                        \'objective\': \'regression\',\n                        \'metric\': \'rmse\',\n                        \'learning_rate\': 0.005,\n                        \'subsample\': 0.9855232997390695,\n                        \'max_depth\': 8,\n                        \'top_rate\': 0.9064148448434349,\n                        \'num_leaves\': 87,\n                        \'min_child_weight\': 41.9612869171337,\n                        \'other_rate\': 0.0721768246018207,\n                        \'reg_alpha\': 9.677537745007898,\n                        \'colsample_bytree\': 0.5665320670155495,\n                        \'min_split_gain\': 9.820197773625843,\n                        \'reg_lambda\': 8.2532317400459,\n                        \'min_data_in_leaf\': 21,\n                        \'verbose\': -1,\n                        \'seed\':int(2**n_fold),\n                        \'bagging_seed\':int(2**n_fold),\n                        \'drop_seed\':int(2**n_fold)\n                        }\n\n        reg = lgb.train(\n                        params,\n                        lgb_train,\n                        valid_sets=[lgb_train, lgb_test],\n                        valid_names=[\'train\', \'test\'],\n                        num_boost_round=10000,\n                        early_stopping_rounds= 200,\n                        verbose_eval=100\n                        )\n\n        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)\n        sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits\n\n        fold_importance_df = pd.DataFrame()\n        fold_importance_df["feature"] = feats\n        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type=\'gain\', iteration=reg.best_iteration))\n        fold_importance_df["fold"] = n_fold + 1\n        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)\n        print(\'Fold %2d RMSE : %.6f\' % (n_fold + 1, rmse(valid_y, oof_preds[valid_idx])))\n        del reg, train_x, train_y, valid_x, valid_y\n        gc.collect()\n\n    # display importances\n    display_importances(feature_importance_df)\n    \n        # save submission file\n    submission = pd.read_csv("../input/elo-merchant-category-recommendation/sample_submission.csv")\n    submission[\'target\'] = sub_preds\n    submission.to_csv(boosting+".csv", index=False)\n    display(submission.head())\n    return (submission)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df,test_df = load_data()\nprint(gc.collect())\nsubmission = kfold_lightgbm(train_df, test_df, num_folds=7, stratified=False, boosting=boosting[0])')


# In[ ]:


submission1 = kfold_lightgbm(train_df, test_df, num_folds=7, stratified=False, boosting=boosting[1])


# In[ ]:


final = pd.read_csv("../input/elo-merchant-category-recommendation/sample_submission.csv")
final['target'] = submission['target'] * 0.5 + submission1['target'] * 0.5
final.to_csv("blend.csv",index = False)  

