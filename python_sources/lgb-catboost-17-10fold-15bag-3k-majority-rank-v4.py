#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import os
# if True:
#     os.environ["OMP_NUM_THREADS"] = "1"
#     os.environ["MKL_NUM_THREADS"] = "1"
#     os.environ["OPENBLAS_NUM_THREADS"] = "1"
#     os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#     os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np

import numpy as np
import pandas as pd
import gc
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from scipy.stats import rankdata

import matplotlib.pyplot as plt

import seaborn as sns

from itertools import chain

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from collections import defaultdict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import Voronoi
import time
import torch
from torch import nn
from tqdm import tqdm_notebook as tqdm
import time
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler
from random import random
from copy import copy, deepcopy

from joblib import Parallel, delayed
import multiprocessing
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from functools import partial
import scipy as sp

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[ ]:


path = "/kaggle/input/data-science-bowl-2019/"

def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv(f'{path}/train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv(f'{path}/test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv(f'{path}/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv(f'{path}/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv(f'{path}/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission

def preprocess(train, test, train_labels):
    # encode title
    
    #train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    #test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    
    train['hour'] = train['timestamp'].dt.hour
    test['hour'] = test['timestamp'].dt.hour
    train['weekday'] = train['timestamp'].dt.weekday
    test['weekday'] = test['timestamp'].dt.weekday
    
    return train, test


# In[ ]:


train, test, train_labels, specs, sample_submission = read_data()


# In[ ]:


train, test = preprocess(train, test, train_labels)


# In[ ]:


unique_values = {}
for f in ["title", "event_code", "event_id", "type"]:
    unique_values[f] = pd.concat([train[f], test[f]]).unique()
    
unique_values["assessment"] = pd.concat([train[train["type"]=="Assessment"]["title"], test[test["type"]=="Assessment"]["title"]]).unique()


# In[ ]:


from collections import defaultdict
from copy import deepcopy
import json

def parse_json(s):
    s = json.loads(s)
    ret = {}
    for f in ['round', "level", "misses", "correct"]:
        if f in s:
            if f == "correct":
                s[f] = int(s[f])
            ret[f] = s[f]
        else:
            ret[f] = -1

    return ret

def get_json(df):
    x = df.event_data.apply(parse_json)

    x = pd.io.json.json_normalize(x).fillna(-1)
   # print(x)

    return x

def get_data(user_sample, test_set=False):
    

    
    all_assessments = []
    
    features = defaultdict(np.float32)
    
    for f in ["title", "event_code", "event_id", "type"]:
        for k in unique_values[f]:
            features[f"{f}_{k}"] = 0
            
    for f in ["title", "event_code", "event_id"]:
        for k in unique_values[f]:
            features[f"assessment_{f}_{k}"] = 0
            
    for f in ["title"]:
        for k in unique_values[f]:
            features[f"{f}_{k}_correct"] = -1
            
    user_sample = pd.concat([user_sample.reset_index(drop=True), get_json(user_sample).fillna(-1).reset_index(drop=True)], axis=1)
            
#     for f in range(4):
#         features[f"accuracy_group_{f}"] = 0
            
    assessment_sequence = defaultdict(list)
    
    curr_assessment_sequence = defaultdict(lambda: defaultdict(list))

    for i, session in user_sample.groupby('game_session', sort=False):
        
        for f in ["title"]:
            for k in unique_values[f]:
                features[f"curr_assessment_{f}_{k}"] = 0

        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        features["installation_id"] = session['installation_id'].iloc[0]     
        features["timestamp"] = session['timestamp'].iloc[0]   
        features["s_title"] = session_title
        
        #features["session_count"] += 1

        if (session_type == 'Assessment'):
            
            variables = {}
            
            features[f"curr_assessment_title_{session_title}"] = 1
            
            #features["assessment_count"] += 1
            
            #features['hour'] = session['hour'].iloc[0]
            #features['weekday'] = session['weekday'].iloc[0]
            
          #  features["event_count"] = session.iloc[0]["event_count"]
            
            for f in ["accuracy_group", "correct"]:
                features[f"assessment_sequence_{f}_mean"] = np.mean(assessment_sequence[f]) if len(assessment_sequence[f]) > 0 else -1
                
            for f in ["accuracy_group", "false_attempts", "accuracy"]:
                features[f"curr_assessment_sequence_{f}_mean"] = np.mean(curr_assessment_sequence[session_title][f]) if len(curr_assessment_sequence[session_title][f]) > 0 else -1
                features[f"curr_assessment_sequence_{f}_sum"] = np.sum(curr_assessment_sequence[session_title][f]) if len(curr_assessment_sequence[session_title][f]) > 0 else -1
                features[f"curr_assessment_sequence_{f}_lag1"] = curr_assessment_sequence[session_title][f][-1] if len(curr_assessment_sequence[session_title][f]) > 0 else -1

#             for a in unique_values["assessment"]:
#                 for f in ["accuracy_group"]:
#                     features[f"curr_assessment_sequence_{a}_{f}_sum"] = np.sum(curr_assessment_sequence[a][f]) if len(curr_assessment_sequence[a][f]) > 0 else -1
                    #features[f"curr_assessment_sequence_{a}_{f}_lag1"] = curr_assessment_sequence[a][f][-1] if len(curr_assessment_sequence[a][f]) > 0 else -1
                
            if session["title"].iloc[0] == 'Bird Measurer (Assessment)':
                all_attempts = session.query(f'event_code == 4110')
            else:
                all_attempts = session.query(f'event_code == 4100')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else -1
            
            
            
            
            variables["true_attempts"] = true_attempts
            variables["false_attempts"] = false_attempts
            variables["accuracy"] = accuracy
            
            if accuracy == -1:
                features['accuracy_group'] = -1
            elif accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
                
            if true_attempts+false_attempts > 0:
                all_assessments.append(deepcopy(features))
                
#                 #features[f"assessment_title_{session_title}"] += 1
#                 for f in ["event_id"]:
#                     for k in session[f].values:
#                         features[f"assessment_{f}_{k}"] += 1
                
                for f in ["accuracy_group", "accuracy"]:
                    assessment_sequence[f].append(features['accuracy_group'])
                
                    curr_assessment_sequence[session_title][f].append(features['accuracy_group'])
                    
                #features["accumulated_false_attempts"] += false_attempts
                
                #features[f"accuracy_group_{features['accuracy_group']}"] += 1
                
            elif test_set:
                all_assessments.append(deepcopy(features))
                
            for f in ["true_attempts", "false_attempts"]:
                curr_assessment_sequence[session_title][f].append(variables[f])
           # curr_assessment_sequence[session_title]["len"].append(len(session))
        
        else:
            correct = [x for x in session["correct"] if x != -1]
            if len(correct) > 0:
                assessment_sequence["correct"].append(np.mean(correct))
                
#                 for c in correct:
#                     if c == 0:
#                         features[f"{session_title}_incorrect"] += 1
#                     else:
#                         features[f"{session_title}_correct"] += 1
        
        #features[f"title_{session_title}"] += 1
        for f in ["event_code", "type", "event_id"]:
            for k in session[f].values:
                features[f"{f}_{k}"] += 1
                
        #features["accumulated_actions"] += len(session)

                        
    if test_set:
        return [all_assessments[-1]]
    return all_assessments


# In[ ]:


res = Parallel(n_jobs=4, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")(       delayed(get_data)(user_sample) for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total=train["installation_id"].nunique()))
train_df = list(chain.from_iterable(res))
train_df = pd.DataFrame(train_df)
train_df = train_df.fillna(-1)
train_df = train_df.loc[:, (train_df != 0).any(axis=0)]
train_df = train_df.loc[:, (train_df != -1).any(axis=0)]


# In[ ]:


train_df.head()


# In[ ]:


train_df.shape


# In[ ]:


del train
gc.collect()


# In[ ]:


tmp = test
res = Parallel(n_jobs=2, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")(       delayed(get_data)(user_sample, True) for i, (ins_id, user_sample) in tqdm(enumerate(tmp.groupby('installation_id', sort = False)), total=tmp["installation_id"].nunique()))
test_df = list(chain.from_iterable(res))
test_df = pd.DataFrame(test_df)
test_df = test_df.fillna(-1)
for col in train_df.columns:
    if col not in test_df.columns:
        test_df[col] = 0
test_df = test_df[train_df.columns]


# In[ ]:


tmp = test
res = Parallel(n_jobs=2, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")(       delayed(get_data)(user_sample, False) for i, (ins_id, user_sample) in tqdm(enumerate(tmp.groupby('installation_id', sort = False)), total=tmp["installation_id"].nunique()))
test_train_df = list(chain.from_iterable(res))
test_train_df = pd.DataFrame(test_train_df)
test_train_df = test_train_df.fillna(-1)
for col in train_df.columns:
    if col not in test_train_df.columns:
        test_train_df[col] = 0
test_train_df = test_train_df[train_df.columns]


# In[ ]:


test_train_df.shape


# In[ ]:


del test
gc.collect()


# In[ ]:


test_df.head()


# In[ ]:


train_df = pd.concat([train_df, test_train_df])


# In[ ]:


train_df.shape


# In[ ]:


del test_train_df
gc.collect()


# In[ ]:


train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# In[ ]:


train_df = train_df.sort_values("timestamp").reset_index(drop=True)


# In[ ]:



def pred_to_int(x):
    return np.round(np.clip(x, 0, 3))

def plot_hist(p):
    #print(confusion_matrix(train_df["accuracy_group"].values, pred_to_int(oof)))
    
    print(pd.Series(p).value_counts())

    pd.Series(p).hist(color="blue", alpha=0.5, normed=True)
    train_df['accuracy_group'].hist(alpha=0.5, color="red", normed=True)

def opt_dist(df, p_pred, verbose=True, dist=None):
    
    if dist is None:
        dist = df['accuracy_group'].value_counts().sort_values() / len(df)
    #print(dist)
        
    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(p_pred, acum * 100)
    if verbose:
        print(bound)

    def classify(x):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3

    return np.array(list(map(classify, p_pred)))


# In[ ]:


import numpy as np
from collections import Counter, defaultdict
from sklearn.utils import check_random_state

class RepeatedStratifiedGroupKFold():

    def __init__(self, n_splits=5, n_repeats=1, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        
    # Implementation based on this kaggle kernel:
    #    https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    def split(self, X, y=None, groups=None):
        k = self.n_splits
        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std(
                    [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
                )
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)
            
        rnd = check_random_state(self.random_state)
        for repeat in range(self.n_repeats):
            labels_num = np.max(y) + 1
            y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
            y_distr = Counter()
            for label, g in zip(y, groups):
                y_counts_per_group[g][label] += 1
                y_distr[label] += 1

            y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
            groups_per_fold = defaultdict(set)
        
            groups_and_y_counts = list(y_counts_per_group.items())
            rnd.shuffle(groups_and_y_counts)

            for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
                best_fold = None
                min_eval = None
                for i in range(k):
                    fold_eval = eval_y_counts_per_fold(y_counts, i)
                    if min_eval is None or fold_eval < min_eval:
                        min_eval = fold_eval
                        best_fold = i
                y_counts_per_fold[best_fold] += y_counts
                groups_per_fold[best_fold].add(g)

            all_groups = set(groups)
            for i in range(k):
                train_groups = all_groups - groups_per_fold[i]
                test_groups = groups_per_fold[i]

                train_indices = [i for i, g in enumerate(groups) if g in train_groups]
                test_indices = [i for i, g in enumerate(groups) if g in test_groups]

                yield train_indices, test_indices


# In[ ]:


train_df.index.max()


# In[ ]:


import lightgbm as lgb
from numba import jit

from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool, cv


@jit
def qwk(a1, a2):
    """
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168

    :param a1:
    :param a2:
    :param max_rat:
    :return:
    """
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e

if "p" in train_df:
    del train_df["p"]

class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])

        return -qwk(y, X_p)

    def fit(self, X, y, init=[1, 1.5, 2.5]):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = init
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead', options={"maxiter": 100_000, 'maxiter': 100_000})

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])


    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']
    
class OptimizedRounderTruncated(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y, indices):

        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        kappas = []
        for idx in indices:
#             print(y.shape)
#             print(idx)
            X_p = (pd.cut(X[idx], [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])).astype(np.int)
            #print(y[idx])
            #print(X_p.astype(np.int))
            kappas.append(qwk(y[idx], X_p))
        
        #print(np.mean(kappas))
        return -np.median(kappas)

    def fit(self, X, y, indices, init=[1, 1.5, 2.5]):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        
        dist = pd.Series(y).value_counts().sort_values() / len(y)
        acum = 0
        bound = []
        for i in range(3):
            acum += dist[i]
            bound.append(np.percentile(X, acum * 100))
        init = bound
            
        loss_partial = partial(self._kappa_loss, X=X, y=y, indices=indices)
        initial_coef = init
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead', options={"maxiter": 10_000})

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])


    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']
    

def eval_truncated(p, run_opt_dist=False, dist=None, test_preds=None, random_state=0):

    train_df["p"] = p
    scores = []
    hists = []
    
    trunc_indices = []
    for j in tqdm(range(1000)):

        np.random.seed(random_state+j)
        s = train_df[["installation_id"]].sample(frac=1.0, random_state=random_state+j).groupby('installation_id', sort=False).head(1)

        trunc_indices.append(s.index.values)
        
    print(np.mean(trunc_indices[:1000]))
    
    optR = OptimizedRounderTruncated()
    optR.fit(p.reshape(-1,), train_df["accuracy_group"].values, trunc_indices)

    coefficients = optR.coefficients()
    print(coefficients)
    
    for idx in tqdm(trunc_indices):

        #y = train_df[["accuracy_group", "installation_id", "p", "game_session"]].sample(frac=1.0, random_state=j).groupby('installation_id', sort=False).head(1)
        y = train_df.iloc[idx]
        #if run_opt_dist == True:
        #    y["p"] = opt_dist(y, y["p"], verbose=False, dist=dist)
        
        p = optR.predict(y["p"], coefficients)
        
        #score = cohen_kappa_score(y["accuracy_group"], pred_to_int(y["p"]), weights='quadratic') 
        score = qwk(y["accuracy_group"], p) 
        scores.append(score)
        
        #y = y.sort_values("accuracy_group")
        #hist = y["accuracy_group"].value_counts().sort_values() / len(y)
        #hists.append(hist)

    print()
    print(np.round(np.median(scores),5), np.round(np.mean(scores),5), np.round(np.std(scores),5), np.round(np.min(scores),5), np.round(np.max(scores),5))
    #print(np.mean(hists, axis=0))
    
    del train_df["p"]
    
    if test_preds is not None:
        return optR.predict(test_preds, coefficients).astype(np.int), np.round(np.median(scores),5)


# In[ ]:


train_df.columns = train_df.columns.map(str)
test_df.columns = test_df.columns.map(str)

nbags = 15
n_splits = 10
test_preds_all = []
trunc_score_all = []
for bag in range(nbags):
    kf = RepeatedStratifiedGroupKFold(n_splits=n_splits, random_state=42+bag)
    #kf = StratifiedKFold(n_splits=n_splits, random_state=42+bag, shuffle=True)

    folds = []
    for trn_idx, val_idx in kf.split(np.arange(len(train_df)), train_df["accuracy_group"].values, groups=train_df["installation_id"].values):

        df = train_df.iloc[trn_idx][["installation_id"]].reset_index()
        df["idx"] = np.arange(len(trn_idx))
        trn_idx_rnd = []
        for j in range(1):
            s = df.sample(frac=1.0, random_state=j).groupby('installation_id', sort=False).head(1)
            trn_idx_rnd.append(s.index.values)

        df = train_df.iloc[val_idx][["installation_id"]].reset_index()
        df["idx"] = np.arange(len(val_idx))
        val_idx_rnd = []
        for j in range(1):
            s = df[["installation_id"]].sample(frac=1.0, random_state=j).groupby('installation_id', sort=False).head(1)
            val_idx_rnd.append(s.index.values)
        folds.append([trn_idx, val_idx, trn_idx_rnd, val_idx_rnd])

    oof_lgb = np.zeros(len(train_df))
    oof_catboost = np.zeros(len(train_df))

    pred_test_lgb = []
    pred_test_catboost = []

    truncated_scores = []
    
    if "p" in train_df:
        del train_df["p"]

    remove_col = ["game_session",   "accuracy_group"]
    

    for fold, (trn_idx, val_idx, trn_idx_rnd, val_idx_rnd) in enumerate(folds):

        X_train = train_df[[c for c in train_df.columns if c not in remove_col]].iloc[trn_idx]
        y_train = train_df["accuracy_group"].iloc[trn_idx]

        X_val = train_df[[c for c in train_df.columns if c not in remove_col]].iloc[val_idx]
        y_val = train_df["accuracy_group"].iloc[val_idx].values

        #X_test_train = test_train_df[[c for c in train_df.columns if c not in remove_col]]
        #y_test_train = test_train_df["accuracy_group"]

        X_test = test_df[[c for c in train_df.columns if c not in remove_col]]
        
#         print("y_train", y_train.shape)
        
#         ts = X_val.groupby("installation_id")["timestamp"].last()
#         X_train["ts_diff"] = X_train["installation_id"].map(ts).values
#         X_train["ts_diff"] = (X_train["timestamp"].dt.tz_convert(None) - X_train["ts_diff"]).astype('timedelta64[s]')

#         X_train = X_train[X_train["ts_diff"]>0]
#         y_train = y_train[X_train.index]
#         print("y_train", y_train.shape)
        
#         del X_train["ts_diff"]
        

        #X_train = pd.concat([X_train, X_test_train])
        #y_train = pd.concat([y_train, y_test_train])

        y_train = y_train.values
        
        params = {
            'loss_function': 'RMSE',

            'eval_metric': "RMSE",
            'task_type' : 'CPU',
            'early_stopping_rounds' : 300,
            "boosting_type": "Plain",
            "learning_rate": 0.05,
            'thread_count': -1,
            'max_depth': 6,
            "iterations": 10_000,
            'random_strength': 5,
            'border_count': 64,
            #'max_ctr_complexity': 10,
            'bagging_temperature': 1,
            'bootstrap_type': "Bayesian",
            "grow_policy": "SymmetricTree",
           # "ignored_features": [768, 769, 770, 771, 772],
            "simple_ctr":'Buckets',
            "combinations_ctr":'Buckets',
           # 'per_feature_ctr':'CtrType=Buckets',
           # "max_leaves": 31,
           # "one_hot_max_size": 128,
          # "min_data_in_leaf": 100,
            'l2_leaf_reg': 2,
            'feature_border_type': 'GreedyLogSum',
           'has_time': True,
            'colsample_bylevel': 0.3,
            'use_best_model': True,
           # 'od_type': "Iter",
            #'logging_level':'Silent',
        }
        
        for f in []:
            if f in X_train:
                del X_train[f]
                del X_val[f]
                del X_test[f]
        
        categorical_columns = ["s_title", "installation_id"]
        
        #print("cat")
        model = CatBoostRegressor(**params)
        model.fit(X=X_train, y=y_train, 
               eval_set=(X_val, y_val), 
               early_stopping_rounds=params['early_stopping_rounds'], verbose=100,
               cat_features=categorical_columns)


        pred_val = model.predict(X_val)
        oof_catboost[val_idx] = rankdata(pred_val) / len(pred_val)

        p_t = model.predict(X_test)
        pred_test_catboost.append(rankdata(p_t) / len(p_t))

        params = {'n_estimators':10000,
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'subsample': 0.3,
                'subsample_freq': 1,
                'learning_rate': 0.01,
                'feature_fraction': 0.5,
             'num_leaves': 32,
                'lambda_l1': 1,  
                'lambda_l2': 1,
                'verbose': 100,
              'min_data_in_leaf': 100,
                'early_stopping_rounds': 300, 'eval_metric': 'cappa',
              'n_jobs': -1
                }#
        
        for f in ["s_title", "timestamp"]:
            if f in X_train:
                del X_train[f]
                del X_val[f]
                del X_test[f]
                
        categorical_columns = ["installation_id"]
        
        for f in categorical_columns:
            if f in X_train:
                X_train[f] = X_train[f].astype("category")
                X_val[f] = X_val[f].astype("category")
                X_test[f] = X_test[f].astype("category")

#         categorical_columns = ["installation_id"]
        
#         X_train[categorical_columns] = X_train[categorical_columns].astype("category")
#         X_val[categorical_columns] = X_val[categorical_columns].astype("category")


        
        #print("lgb")
        model = lgb.LGBMRegressor(**params)
        model.fit(X=X_train, y=y_train, 
                   eval_set=[[X_val, y_val]], 
                   verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'], eval_metric="rmse",
                   categorical_feature=categorical_columns)
        
        pred_val = model.predict(X_val)
        #oof_lgb[val_idx] = rankdata(pred_val) / len(pred_val)
        oof_lgb[val_idx] = rankdata(pred_val) / len(pred_val)

        p_t = model.predict(X_test)
        pred_test_lgb.append(rankdata(p_t) / len(p_t))

        print()
        
        #break
        
    pred_test_lgb = np.mean(pred_test_lgb, axis=0)
    pred_test_catboost = np.mean(pred_test_catboost, axis=0)
    
    oof = 0.75 * oof_lgb + 0.25* oof_catboost
    pred_test = 0.75 * pred_test_lgb + 0.25 * pred_test_catboost


    score = cohen_kappa_score(train_df["accuracy_group"], pred_to_int(oof), weights='quadratic') 
    print("Full CV Kappa: {:<8.4f}".format(score))    

    #score = cohen_kappa_score(train_df["accuracy_group"], pred_to_int(oof), weights='quadratic') 
    print("Truncated CV Mean Kappa: {:<8.4f}".format(np.mean(truncated_scores)))    

    oof_opt = opt_dist(train_df, oof, verbose=False)
    score = cohen_kappa_score(train_df["accuracy_group"], pred_to_int(oof_opt), weights='quadratic')
    print("Full CV Kappa DIST OPT: {:<8.4f}".format(score))   
    
    score = np.sqrt(mean_squared_error(train_df["accuracy_group"], oof))
    print("Full CV RMSE: {:<8.4f}".format(score))  

    sns.distplot(pred_test)
    sns.distplot(oof)
    plt.show()
    
    pred_test_opt, trunc_score = eval_truncated(oof, test_preds = pred_test.reshape(-1), random_state=bag*10_000)
    test_preds_all.append(pred_test_opt)
    trunc_score_all.append(trunc_score)
    
    print()
    print()
    print()


# In[ ]:


print(trunc_score_all)


# In[ ]:


print(np.mean(trunc_score_all))


# In[ ]:


test_preds_all = np.vstack(test_preds_all).T


# In[ ]:


pred_test_opt = sp.stats.mode(test_preds_all, axis=1)[0].reshape(-1)


# In[ ]:


plot_hist(pred_test_opt)


# In[ ]:


sample_submission["accuracy_group"] = pred_test_opt.astype(np.int)
sample_submission["accuracy_group"].value_counts()


# In[ ]:


sample_submission.to_csv('submission.csv', index=False)


# In[ ]:


(train_df.columns == test_df.columns).mean()


# In[ ]:





# In[ ]:





# In[ ]:




