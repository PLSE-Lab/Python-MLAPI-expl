#!/usr/bin/env python
# coding: utf-8

# This notebook is based on this [baseline notebook](https://www.kaggle.com/yasufuminakama/trends-lgb-baseline).  
# 
# The two major changes are as follows.
# * Calculate early stooping using the feature-weighted, normalized absolute errors.
# * There's only one model.
# 
# I tried a few things, including changing the ovjective of Lightgbm, but I couldn't get a better score than the original beseline notebook.  
# It might be better to create five models per feature.

# ## update
# * ver2  
# Changed from deleting lines with missing values to deleting only missing values.

# # Library

# In[ ]:


#===========================================================
# Library
#===========================================================
import os
import gc
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from contextlib import contextmanager
import time

import numpy as np
import pandas as pd
import scipy as sp
from scipy.sparse import csr_matrix
import random

import matplotlib.pyplot as plt
import seaborn as sns

from functools import partial

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn import preprocessing
import category_encoders as ce
from sklearn.metrics import mean_squared_error


import torch

import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


os.listdir('../input/trends-assessment-prediction/')


# # Utils

# In[ ]:


#===========================================================
# Utils
#===========================================================
def get_logger(filename='log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

logger = get_logger()


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f'[{name}] done in {time.time() - t0:.0f} s')


def seed_everything(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
def load_df(path, df_name, debug=False):
    if path.split('.')[-1]=='csv':
        df = pd.read_csv(path)
        if debug:
            df = pd.read_csv(path, nrows=1000)
    elif path.split('.')[-1]=='pkl':
        df = pd.read_pickle(path)
    if logger==None:
        print(f"{df_name} shape / {df.shape} ")
    else:
        logger.info(f"{df_name} shape / {df.shape} ")
    return df


# # Config

# In[ ]:


#===========================================================
# Config
#===========================================================
OUTPUT_DICT = ''

ID = 'Id'
TARGET_COLS = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
SEED = 42
seed_everything(seed=SEED)

N_FOLD = 5


# # Data Loading

# In[ ]:


train = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv', dtype={'Id':str})
reveal_ID = pd.read_csv('../input/trends-assessment-prediction/reveal_ID_site2.csv', dtype={'Id':str})
ICN_numbers = pd.read_csv('../input/trends-assessment-prediction/ICN_numbers.csv')
loading = pd.read_csv('../input/trends-assessment-prediction/loading.csv', dtype={'Id':str})
fnc = pd.read_csv('../input/trends-assessment-prediction/fnc.csv', dtype={'Id':str})
sample_submission = pd.read_csv('../input/trends-assessment-prediction/sample_submission.csv', dtype={'Id':str})


# In[ ]:


sample_submission['ID_num'] = sample_submission[ID].apply(lambda x: int(x.split('_')[0]))
test = pd.DataFrame({ID: sample_submission['ID_num'].unique().astype(str)})
del sample_submission['ID_num']; gc.collect()
test.head()


# # prepare folds

# In[ ]:


Fold = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
for n, (train_index, val_index) in enumerate(Fold.split(train)):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)
train.head()


# In[ ]:


train.isnull().sum()


# # FE

# In[ ]:


full = pd.concat([train, test], axis=0, sort=False)
full = full.set_index([ID, 'fold'])     .stack(dropna=False).reset_index().rename(columns={'level_2': 'feature',
                                                       0: 'target'})
full['full_Id'] = [f'{i}_{j}' for i, j in zip(full['Id'], full['feature'])]


# In[ ]:


# merge
full = full.merge(loading, on=ID, how='left')
full = full.merge(fnc, on=ID, how='left')
full.head()


# In[ ]:


# category
feature_dict = {'age': 0,
                'domain1_var1': 1,
                'domain1_var2': 2,
                'domain2_var1': 3,
                'domain2_var2': 4}
full['feature_cat'] = full['feature'].map(feature_dict)
full.head()


# In[ ]:


train = full.loc[full['target'].notnull()].reset_index(drop=True)
train['fold'] = train['fold'].astype(int)
test = full.loc[full['target'].isnull()].reset_index(drop=True)       


# In[ ]:


train.shape, test.shape


# # MODEL

# In[ ]:


# metric

class Metric():
    # https://www.kaggle.com/girmdshinsei/for-japanese-beginner-with-wrmsse-in-lgbm
    def __init__(self, feature):
        super().__init__()
        self.feature_name = ['age', 'domain1_var1', 'domain1_var2',
                             'domain2_var1', 'domain2_var2']
        self.feature_mat_csr = self._get_feature_matrix_csr(feature)

    def _get_feature_matrix_csr(self, feature):
        mat = pd.get_dummies(feature).loc[:, self.feature_name].values
        mat_csr = csr_matrix(mat)
        return mat_csr

    def _calc_fwnae_csr(self, preds, y_true):
        scores = np.abs(preds - y_true) * self.feature_mat_csr / (y_true * self.feature_mat_csr)
        score = np.average(scores, weights=np.array([0.3, 0.175, 0.175, 0.175, 0.175]))
        return score

    def _calc_fwnae(self, preds, y_true):
        scores = np.abs(preds - y_true) * self.feature_mat_csr / np.dot(y_true, self.feature_mat)
        score = np.average(scores, weights=np.array([0.3, 0.175, 0.175, 0.175, 0.175]))
        return score

    def fwnae_lgb(self, preds, data):
        """
        # this function is calculate feature-weighted, normalized absolute errors
        # https://www.kaggle.com/c/trends-assessment-prediction/overview/evaluation
        """
        y_true = data.get_label()  # actual obserbed values
        fwnae = self._calc_fwnae_csr(preds, y_true)

        return 'fwnae', fwnae, False

    def _calc_each_fwnae_csr(self, preds, y_true):
        scores = np.abs(preds - y_true) * self.feature_mat_csr / (y_true * self.feature_mat_csr)
        return scores

    def calc_each_fwnae(self, preds, y_train):
        scores = self._calc_each_fwnae_csr(preds, y_train)
        scores_dict = {i: j for i, j in zip(self.feature_name, scores)}
        score = np.average(scores, weights=np.array([0.3, 0.175, 0.175, 0.175, 0.175]))
        return scores_dict, score


# In[ ]:


#===========================================================
# model
#===========================================================
def run_single_lightgbm(param, train_df, test_df, folds, features, target, fold_num=0, categorical=[]):
    
    trn_idx = folds[folds.fold != fold_num].index
    val_idx = folds[folds.fold == fold_num].index
    logger.info(f'len(trn_idx) : {len(trn_idx)}')
    logger.info(f'len(val_idx) : {len(val_idx)}')
    metric = Metric(folds.iloc[val_idx]['feature'])
    
    if categorical == []:
        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features],
                               label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train_df.iloc[val_idx][features],
                               label=target.iloc[val_idx])
    else:
        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features],
                               label=target.iloc[trn_idx],
                               categorical_feature=categorical)
        val_data = lgb.Dataset(train_df.iloc[val_idx][features],
                               label=target.iloc[val_idx],
                               categorical_feature=categorical)

    oof = np.zeros(len(train_df))
    predictions = np.zeros(len(test_df))

    num_round = 2000

    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets=[val_data],
                    feval=metric.fwnae_lgb,
                    verbose_eval=100,
                    early_stopping_rounds=100)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_num

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration)
    
    # RMSE
    logger.info("fold{} RMSE score: {:<8.5f}".format(fold_num, np.sqrt(mean_squared_error(target[val_idx], oof[val_idx]))))
    
    return oof, predictions, fold_importance_df


def run_kfold_lightgbm(param, train, test, folds, features, target, n_fold=5, categorical=[]):
    
    logger.info(f"================================= {n_fold}fold lightgbm =================================")
    
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    for fold_ in range(n_fold):
        print("Fold {}".format(fold_))
        _oof, _predictions, fold_importance_df = run_single_lightgbm(param,
                                                                     train,
                                                                     test,
                                                                     folds,
                                                                     features,
                                                                     target,
                                                                     fold_num=fold_,
                                                                     categorical=categorical)
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        oof += _oof
        predictions += _predictions / n_fold

    # RMSE
    logger.info("CV RMSE score: {:<8.5f}".format(np.sqrt(mean_squared_error(target, oof))))

    logger.info(f"=========================================================================================")
    
    return feature_importance_df, predictions, oof

    
def show_feature_importance(feature_importance_df):
    cols = (feature_importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:50].index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    plt.figure(figsize=(8, 16))
    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('Features importance (averaged/folds)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DICT+f'feature_importance.png')


# In[ ]:


TARGET_COLS = ['target']
target = train[TARGET_COLS[0]]

# features
folds = train[['fold', 'feature']]
cat_features = []
num_features = [c for c in test.columns if (test.dtypes[c] != 'object') & (c not in cat_features)]
features = num_features + cat_features
DROP_COLS = ['fold', 'feature', 'full_Id']
drop_features = [ID] + TARGET_COLS + DROP_COLS
features = [c for c in features if c not in drop_features]

lgb_param = {'objective': 'regression',
         'boosting_type': 'gbdt',
         'learning_rate': 0.03,
         'seed': SEED,
         'max_depth': -1,
         'verbosity': -1
        }

feature_importance_df, predictions, oof = run_kfold_lightgbm(lgb_param, train, test, folds, features, target, 
                                                             n_fold=N_FOLD, categorical=cat_features)

show_feature_importance(feature_importance_df)


# In[ ]:


metric = Metric(folds['feature'])
scores = metric.calc_each_fwnae(oof, train['target'].values)

logger.info(f'Local Score: {scores[1]}')
logger.info(f'Local Score: {scores[0]}')


# # Submission

# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission.set_index(ID, inplace=True)
sub = test[['full_Id']]
sub['Predicted'] = predictions
sub.set_index('full_Id', inplace=True)
sample_submission['Predicted'] = sub['Predicted']
sample_submission.reset_index(inplace=True)
sample_submission.to_csv('submission.csv', index=False)

