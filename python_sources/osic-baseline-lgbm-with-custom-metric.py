#!/usr/bin/env python
# coding: utf-8

# ## About
# 
# In this competition, participants are requiered to predict `FVC` and its **_`Confidence`_**.  
# Here, I trained Lightgbm to predict them at the same time by utilizing custom metric.
# 
# Most of codes in this notebook are forked from @yasufuminakama 's [lgbm baseline](https://www.kaggle.com/yasufuminakama/osic-lgb-baseline). Thanks!

# ## Library

# In[ ]:


import os
import operator
import typing as tp
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from functools import partial


import numpy as np
import pandas as pd
import random
import math

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import mean_squared_error
import category_encoders as ce

from PIL import Image
import cv2
import pydicom

import torch

import lightgbm as lgb
from sklearn.linear_model import Ridge

import warnings
warnings.filterwarnings("ignore")


# ## Utils

# In[ ]:


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


def seed_everything(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ## Config

# In[ ]:


OUTPUT_DICT = './'

ID = 'Patient_Week'
# TARGET = 'FVC'
TARGET = 'virtual_FVC'
SEED = 42
VIRTUAL_BASE_FVC = 2000
seed_everything(seed=SEED)

N_FOLD = 4


# # Data Loading

# In[ ]:


train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
train[ID] = train['Patient'].astype(str) + '_' + train['Weeks'].astype(str)
print(train.shape)
train.head()


# In[ ]:


# construct train input

output = pd.DataFrame()
gb = train.groupby('Patient')
tk0 = tqdm(gb, total=len(gb))
for _, usr_df in tk0:
    usr_output = pd.DataFrame()
    for week, tmp in usr_df.groupby('Weeks'):
        rename_cols = {'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Percent': 'base_Percent', 'Age': 'base_Age'}
        tmp = tmp.drop(columns='Patient_Week').rename(columns=rename_cols)
        drop_cols = ['Age', 'Sex', 'SmokingStatus', 'Percent']
        _usr_output = usr_df.drop(columns=drop_cols).rename(columns={'Weeks': 'predict_Week'}).merge(tmp, on='Patient')
        _usr_output['Week_passed'] = _usr_output['predict_Week'] - _usr_output['base_Week']
        usr_output = pd.concat([usr_output, _usr_output])
    output = pd.concat([output, usr_output])
    
train = output[output['Week_passed']!=0].reset_index(drop=True)
print(train.shape)
train.head()


# In[ ]:


# make new taret: virutal FVC
train['virtual_FVC'] = train["FVC"] / train["base_FVC"] * VIRTUAL_BASE_FVC


# In[ ]:


# construct test input

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')        .rename(columns={'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Percent': 'base_Percent', 'Age': 'base_Age'})
submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
submission['Patient'] = submission['Patient_Week'].apply(lambda x: x.split('_')[0])
submission['predict_Week'] = submission['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)
test = submission.drop(columns=['FVC', 'Confidence']).merge(test, on='Patient')
test['Week_passed'] = test['predict_Week'] - test['base_Week']
print(test.shape)
test.head()


# In[ ]:


test["FVC"] = np.nan
test["virtual_FVC"] = np.nan


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
print(submission.shape)
submission.head()


# # Prepare folds

# In[ ]:


folds = train[[ID, 'Patient', TARGET]].copy()
#Fold = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
Fold = GroupKFold(n_splits=N_FOLD)
groups = folds['Patient'].values
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[TARGET], groups)):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)
folds.head()


# ## Custom Objective / Metric
# 
# The competition evaluation metric is:
# 
# $
# \displaystyle \sigma_{clipped} = \max \left ( \sigma, 70 \right ) \\
# \displaystyle \Delta = \min \left ( \|FVC_{ture} - FVC_{predicted}\|, 1000 \right ) \\
# \displaystyle f_{metric} = - \frac{\sqrt{2} \Delta}{\sigma_{clipped}} - \ln \left( \sqrt{2} \sigma_{clipped} \right) .
# $
# 
# This is too complex to directly optimize by custom metric.
# Here I use negative loglilelihood loss (_NLL_) of gaussian.  
# 
# Let $FVC_{ture}$ is $t$ and $FVC_{predicted}$ is $\mu$, the _NLL_ $l$ is formulated by:
# 
# $
# \displaystyle l\left( t, \mu, \sigma \right) =
# -\ln \left [ \frac{1}{\sqrt{2 \pi} \sigma} \exp \left \{ - \frac{\left(t - \mu \right)^2}{2 \sigma^2} \right \} \right ]
# = \frac{\left(t - \mu \right)^2}{2 \sigma^2} + \ln \left( \sqrt{2 \pi} \sigma \right).
# $
# 
# `grad` and `hess` are calculated as follows:
# 
# $
# \displaystyle  \frac{\partial l}{\partial \mu } = -\frac{t - \mu}{\sigma^2} \ , \ \frac{\partial^2 l}{\partial \mu^2 } = \frac{1}{\sigma^2}
# $
# 
# $
# \displaystyle \frac{\partial l}{\partial \sigma}
# =-\frac{\left(t - \mu \right)^2}{\sigma^3} + \frac{1}{\sigma} = \frac{1}{\sigma} \left\{ 1 - \left ( \frac{t - \mu}{\sigma} \right)^2 \right \}
# \\
# \displaystyle \frac{\partial^2 l}{\partial \sigma^2}
# = -\frac{1}{\sigma^2} \left\{ 1 - \left ( \frac{t - \mu}{\sigma} \right)^2 \right \}
# +\frac{1}{\sigma} \frac{2 \left(t - \mu \right)^2 }{\sigma^3}
# = -\frac{1}{\sigma^2} \left\{ 1 - 3 \left ( \frac{t - \mu}{\sigma} \right)^2 \right \}
# $

# For numerical stability, I replace $\sigma$ with $\displaystyle \tilde{\sigma} := \log\left(1 + \mathrm{e}^{\sigma} \right).$
# 
# $
# \displaystyle l'\left( t, \mu, \sigma \right)
# = \frac{\left(t - \mu \right)^2}{2 \tilde{\sigma}^2} + \ln \left( \sqrt{2 \pi} \tilde{\sigma} \right).
# $
# 
# $
# \displaystyle \frac{\partial l'}{\partial \mu } = -\frac{t - \mu}{\tilde{\sigma}^2} \ , \ \frac{\partial^2 l}{\partial \mu^2 } = \frac{1}{\tilde{\sigma}^2}
# $
# <br>
# 
# $
# \displaystyle \frac{\partial l'}{\partial \sigma}
# = \frac{1}{\tilde{\sigma}} \left\{ 1 - \left ( \frac{t - \mu}{\tilde{\sigma}} \right)^2 \right \} \frac{\partial \tilde{\sigma}}{\partial \sigma}
# \\
# \displaystyle \frac{\partial^2 l'}{\partial \sigma^2}
# = -\frac{1}{\tilde{\sigma}^2}  \left\{ 1 - 3 \left ( \frac{t - \mu}{\tilde{\sigma}} \right)^2 \right \}
# \left( \frac{\partial \tilde{\sigma}}{\partial \sigma} \right) ^2
# +\frac{1}{\tilde{\sigma}} \left\{ 1 - \left ( \frac{t - \mu}{\tilde{\sigma}} \right)^2 \right \} \frac{\partial^2 \tilde{\sigma}}{\partial \sigma^2}
# $
# 
# , where  
# 
# $
# \displaystyle
# \frac{\partial \tilde{\sigma}}{\partial \sigma} = \frac{1}{1 + \mathrm{e}^{-\sigma}} \\
# \displaystyle
# \frac{\partial^2 \tilde{\sigma}}{\partial^2 \sigma} = \frac{\mathrm{e}^{-\sigma}}{\left( 1 + \mathrm{e}^{-\sigma} \right)^2}
# = \frac{\partial \tilde{\sigma}}{\partial \sigma} \left( 1 - \frac{\partial \tilde{\sigma}}{\partial \sigma} \right)
# $

# In[ ]:


class OSICLossForLGBM:
    """
    Custom Loss for LightGBM.
    
    * Objective: return grad & hess of NLL of gaussian
    * Evaluation: return competition metric
    """
    
    def __init__(self, epsilon: float=1e-09) -> None:
        """Initialize."""
        self.name = "osic_loss"
        self.n_class = 2  # FVC & Confidence
        self.epsilon = epsilon
    
    def __call__(self, preds: np.ndarray, labels: np.ndarray, weight: tp.Optional[np.ndarray]=None) -> float:
        """Calc loss."""
        mu = preds[:, 0]
        sigma = preds[:, 1]
        sigma_t = np.log(1 + np.exp(sigma))
        loss_by_sample = ((labels - mu) / sigma_t) ** 2 / 2 + np.log(np.sqrt(2 * np.pi) * sigma_t)
        loss = np.average(loss_by_sample, weight)
        
        return loss
    
    def _calc_grad_and_hess(
        self, preds: np.ndarray, labels: np.ndarray, weight: tp.Optional[np.ndarray]=None
    ) -> tp.Tuple[np.ndarray]:
        """Calc Grad and Hess"""
        mu = preds[:, 0]
        sigma = preds[:, 1]
        
        sigma_t = np.log(1 + np.exp(sigma))
        grad_sigma_t = 1 / (1 + np.exp(- sigma))
        hess_sigma_t = grad_sigma_t * (1 - grad_sigma_t)
        
        grad = np.zeros_like(preds)
        hess = np.zeros_like(preds)
        grad[:, 0] = - (labels - mu) / sigma_t ** 2
        hess[:, 0] = 1 / sigma_t ** 2
        
        tmp = ((labels - mu) / sigma_t) ** 2
        grad[:, 1] = 1 / sigma_t * (1 - tmp) * grad_sigma_t
        hess[:, 1] = (
            - 1 / sigma_t ** 2 * (1 - 3 * tmp) * grad_sigma_t ** 2
            + 1 / sigma_t * (1 - tmp) * hess_sigma_t
        )
        if weight is not None:
            grad = grad * weight[:, None]
            hess = hess * weight[:, None]
        return grad, hess
    
    def return_loss(self, preds: np.ndarray, data: lgb.Dataset) -> tp.Tuple[str, float, bool]:
        """Return Loss for lightgbm"""
        labels = data.get_label()
        weight = data.get_weight()
        n_example = len(labels)
        
        # # reshape preds: (n_class * n_example,) => (n_class, n_example) =>  (n_example, n_class)
        preds = preds.reshape(self.n_class, n_example).T
        # # calc loss
        loss = self(preds, labels, weight)
        
        return self.name, loss, False
    
    def return_grad_and_hess(self, preds: np.ndarray, data: lgb.Dataset) -> tp.Tuple[np.ndarray]:
        """Return Grad and Hess for lightgbm"""
        labels = data.get_label()
        weight = data.get_weight()
        n_example = len(labels)
        
        # # reshape preds: (n_class * n_example,) => (n_class, n_example) =>  (n_example, n_class)
        preds = preds.reshape(self.n_class, n_example).T
        # # calc grad and hess.
        grad, hess =  self._calc_grad_and_hess(preds, labels, weight)

        # # reshape grad, hess: (n_example, n_class) => (n_class, n_example) => (n_class * n_example,) 
        grad = grad.T.reshape(n_example * self.n_class)
        hess = hess.T.reshape(n_example * self.n_class)
        
        return grad, hess
    
    
    def calc_comp_metric(self, preds: np.ndarray, labels: np.ndarray, weight: tp.Optional[np.ndarray]=None) -> float:
        """Calc competition metric."""
        sigma_clip = np.maximum(preds[:, 1], 70)
        Delta = np.minimum(np.abs(preds[:, 0] - labels), 1000)
        loss_by_sample = - np.sqrt(2) * Delta / sigma_clip - np.log(np.sqrt(2) * sigma_clip)
        loss = np.average(loss_by_sample, weight)
        
        return loss


# ## Training Utils

# In[ ]:


#===========================================================
# model
#===========================================================
def run_single_lightgbm(
    model_param, fit_param, train_df, test_df, folds, features, target,
    fold_num=0, categorical=[], my_loss=None,
):
    trn_idx = folds[folds.fold != fold_num].index
    val_idx = folds[folds.fold == fold_num].index
    logger.info(f'len(trn_idx) : {len(trn_idx)}')
    logger.info(f'len(val_idx) : {len(val_idx)}')
    
    if categorical == []:
        trn_data = lgb.Dataset(
            train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
        val_data = lgb.Dataset(
            train_df.iloc[val_idx][features], label=target.iloc[val_idx])
    else:
        trn_data = lgb.Dataset(
            train_df.iloc[trn_idx][features], label=target.iloc[trn_idx],
            categorical_feature=categorical)
        val_data = lgb.Dataset(
            train_df.iloc[val_idx][features], label=target.iloc[val_idx],
            categorical_feature=categorical)

    oof = np.zeros((len(train_df), 2))
    predictions = np.zeros((len(test_df), 2))
    
    clf = lgb.train(
        model_param, trn_data, **fit_param,
        valid_sets=[trn_data, val_data],
        fobj=my_loss.return_grad_and_hess,
        feval=my_loss.return_loss,
    )
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_num

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration)
    
    # RMSE
    logger.info("fold{} RMSE score: {:<8.5f}".format(
        fold_num, np.sqrt(mean_squared_error(target[val_idx], oof[val_idx, 0]))))
    # Competition Metric
    logger.info("fold{} Metric: {:<8.5f}".format(
        fold_num, my_loss(oof[val_idx], target[val_idx])))
    
    return oof, predictions, fold_importance_df


def run_kfold_lightgbm(
    model_param, fit_param, train, test, folds,
    features, target, n_fold=5, categorical=[], my_loss=None,
):
    
    logger.info(f"================================= {n_fold}fold lightgbm =================================")
    
    oof = np.zeros((len(train), 2))
    predictions = np.zeros((len(test), 2))
    feature_importance_df = pd.DataFrame()

    for fold_ in range(n_fold):
        print("Fold {}".format(fold_))
        _oof, _predictions, fold_importance_df =            run_single_lightgbm(
                model_param, fit_param, train, test, folds,
                features, target, fold_num=fold_, categorical=categorical, my_loss=my_loss
            )
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        oof += _oof
        predictions += _predictions / n_fold

    # RMSE
    logger.info("CV RMSE score: {:<8.5f}".format(np.sqrt(mean_squared_error(target, oof[:, 0]))))
    # Metric
    logger.info("CV Metric: {:<8.5f}".format(my_loss(oof, target)))
                

    logger.info(f"=========================================================================================")
    
    return feature_importance_df, predictions, oof

    
def show_feature_importance(feature_importance_df, name):
    cols = (feature_importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:50].index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    #plt.figure(figsize=(8, 16))
    plt.figure(figsize=(6, 4))
    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('Features importance (averaged/folds)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DICT+f'feature_importance_{name}.png')


# ## predict "virutal" FVC & Confidence(sigma)

# In[ ]:


target = train[TARGET]

# features
cat_features = ['Sex', 'SmokingStatus']
num_features = [c for c in test.columns if (test.dtypes[c] != 'object') & (c not in cat_features)]
features = num_features + cat_features
drop_features = [ID, TARGET, 'predict_Week', 'base_Week', "FVC", "base_FVC"]
features = [c for c in features if c not in drop_features]

if cat_features:
    ce_oe = ce.OrdinalEncoder(cols=cat_features, handle_unknown='impute')
    ce_oe.fit(train)
    train = ce_oe.transform(train)
    test = ce_oe.transform(test)
        
lgb_model_param = {
    'num_class': 2,
    # 'objective': 'regression',
    'metric': 'None',
    'boosting_type': 'gbdt',
    'learning_rate': 1e-01,
    'seed': SEED,
    'max_depth': 1,
    "lambda_l2": 5e-03,
    'verbosity': -1,
}
lgb_fit_param = {
    "num_boost_round": 10000,
    "verbose_eval":100,
    "early_stopping_rounds": 100,
}

feature_importance_df, predictions, oof = run_kfold_lightgbm(
    lgb_model_param, lgb_fit_param, train, test,
    folds, features, target,
    n_fold=N_FOLD, categorical=cat_features, my_loss=OSICLossForLGBM())
    
show_feature_importance(feature_importance_df, TARGET)


# In[ ]:


predictions[:5]


# In[ ]:


# convert virtual FVC to FVC
oof = oof / VIRTUAL_BASE_FVC * train[["base_FVC"]].values
predictions = predictions / VIRTUAL_BASE_FVC * test[["base_FVC"]].values


# In[ ]:


OSICLossForLGBM().calc_comp_metric(oof, train["FVC"].values)


# In[ ]:


predictions[:5]


# In[ ]:


train["FVC_pred"] = oof[:, 0]
train["Confidence"] = oof[:, 1]
test["FVC_pred"] = predictions[:, 0]
test["Confidence"] = predictions[:, 1]


# # Submission

# In[ ]:


submission.head()


# In[ ]:


sub = submission.drop(columns=['FVC', 'Confidence']).merge(test[['Patient_Week', 'FVC_pred', 'Confidence']], 
                                                           on='Patient_Week')
sub.columns = submission.columns
sub.to_csv('submission.csv', index=False)
sub.head()

