#!/usr/bin/env python
# coding: utf-8

# There is a growing interest in which validation strategy to employ (like [this discussion](https://www.kaggle.com/c/liverpool-ion-switching/discussion/138850), as is always the case with a tough kaggle competition. When it comes to whether StratifiedKFold or GroupKFold is better, there is actually the third good option: StratifiedGroupKFold. Here I show you an implementation and how to use it.
# 
# As a dataset, I use a cleaned signal uploaded [here](https://www.kaggle.com/cdeotte/data-without-drift).

# # Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib_venn import venn2
import seaborn as sns
sns.set_context("talk")
# sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from tqdm import tqdm
from joblib import Parallel, delayed
from functools import partial
import scipy as sp
from sklearn import metrics
import os, sys, gc
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Load data

# In[ ]:


# train = pd.read_csv("/kaggle/input/liverpool-ion-switching/train.csv")
# test = pd.read_csv("/kaggle/input/liverpool-ion-switching/test.csv")
train = pd.read_csv("/kaggle/input/data-without-drift/train_clean.csv")
bs = 50_000
test = pd.read_csv("/kaggle/input/data-without-drift/test_clean.csv")
# submission = pd.read_csv("/kaggle/input/liverpool-ion-switching/sample_submission.csv")


# In[ ]:


print(train.shape)
train.head()


# In[ ]:


print(test.shape)
test.head()


# # Feature engineering

# In[ ]:


def feature_engineering(df):
    df = df.sort_values(by=['time']).reset_index(drop=True)
    df_index = ((df.time * bs // 5) - 1).values
    df['batch'] = df_index // bs
    df['batch'] = df['batch'].astype(np.int16)
    df['signal'] = df['signal'].astype(np.float16)
    df['batch_index'] = df_index  - (df.batch * bs)
    df['batch_slices'] = df['batch_index']  // (0.1 * bs)
    df['batch_slices'] = df['batch_slices'].astype(np.int16)
    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)
    
    for c in ['batch','batch_slices2']:
        d = {}
        d['mean'+c] = df.groupby([c])['signal'].mean().astype(np.float16)
        d['std'+c] = df.groupby([c])['signal'].std().astype(np.float16)
        d['mean_abs_chg'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))
        d['abs_max'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))
        d['abs_min'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))
        for v in d:
            df[v] = df[c].map(d[v].to_dict())
            df[v] = df[v].astype(np.float16)
    
    #add shifts
    df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])
    df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]
    for i in df[df['batch_index']==0].index:
        df['signal_shift_+1'][i] = np.nan
    for i in df[df['batch_index']==49999].index:
        df['signal_shift_-1'][i] = np.nan

    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]:
        df[c+'_msignal'] = df[c] - df['signal']
        df[c+'_msignal'] = df[c+'_msignal'].astype(np.float16)
        
    return df

train = feature_engineering(train)
test = feature_engineering(test)


# In[ ]:


print(test.shape)
test.head()


# # Utilities

# In[ ]:


class OptimizedRounderF1(object):
    """
    An optimizer for rounding thresholds
    to maximize f1 score
    """
    def __init__(self):
        self.coef_ = 0

    def _f1_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        return -metrics.f1_score(y, X_p, average='macro')

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._f1_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']


# # StratifiedGroupKFold
# The following cell is an example of its implementation.

# In[ ]:


import random
from collections import Counter, defaultdict
from sklearn import model_selection
class GroupKFold(object):
    """
    GroupKFold with random shuffle with a sklearn-like structure

    """

    def __init__(self, n_splits=4, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, group=None):
        return self.n_splits

    def split(self, X, y, group):
        kf = model_selection.KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        unique_ids = X[group].unique()
        for fold, (tr_group_idx, va_group_idx) in enumerate(kf.split(unique_ids)):
            # split group
            tr_group, va_group = unique_ids[tr_group_idx], unique_ids[va_group_idx]
            train_idx = np.where(X[group].isin(tr_group))[0]
            val_idx = np.where(X[group].isin(va_group))[0]
            yield train_idx, val_idx

class StratifiedGroupKFold(object):
    """
    StratifiedGroupKFold with random shuffle with a sklearn-like structure

    """

    def __init__(self, n_splits=4, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, group=None):
        return self.n_splits

    def split(self, X, y, group):
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        groups = X[group].values
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(self.n_splits)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)
        
        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(self.random_state).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(self.n_splits):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_idx = [i for i, g in enumerate(groups) if g in train_groups]
            test_idx = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_idx, test_idx


# # models
# This is a part of my model pipeline which is under development, so please ignore some redundancy. For the sake of time, I use a linear model.

# In[ ]:


def get_oof_ypred(model, x_val, x_test, modelname="linear", task="binary"):  
    """
    get oof and target predictions
    """
    sklearns = ["xgb", "catb", "linear", "knn"]

    if task == "binary": # classification
        # sklearn API
        if modelname in sklearns:
            oof_pred = model.predict_proba(x_val)
            y_pred = model.predict_proba(x_test)
            oof_pred = oof_pred[:, 1]
            y_pred = y_pred[:, 1]
        else:
            oof_pred = model.predict(x_val)
            y_pred = model.predict(x_test)

            # NN specific
            if modelname == "nn":
                oof_pred = oof_pred.ravel()
                y_pred = y_pred.ravel()        

    elif task == "multiclass":
        # sklearn API
        if modelname in sklearns:
            oof_pred = model.predict_proba(x_val)
            y_pred = model.predict_proba(x_test)
        else:
            oof_pred = model.predict(x_val)
            y_pred = model.predict(x_test)

        oof_pred = np.argmax(oof_pred, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    elif task == "regression": # regression
        oof_pred = model.predict(x_val)
        y_pred = model.predict(x_test)

        # NN specific
        if modelname == "nn":
            oof_pred = oof_pred.ravel()
            y_pred = y_pred.ravel()

    return oof_pred, y_pred


# In[ ]:


from sklearn import linear_model

def lin_model(cls, train_set, val_set):
    """
    Linear model hyperparameters and models
    """

    params = {
            'max_iter': 8000,
            'fit_intercept': True,
            'random_state': cls.seed
        }

    if cls.task == "regression":
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
        model = linear_model.Ridge(**{'alpha': 220, 'solver': 'lsqr', 'fit_intercept': params['fit_intercept'],
                                'max_iter': params['max_iter'], 'random_state': params['random_state']})
    elif (cls.task == "binary") | (cls.task == "multiclass"):
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        model = linear_model.LogisticRegression(**{"C": 1.0, "fit_intercept": params['fit_intercept'], 
                                "random_state": params['random_state'], "solver": "lbfgs", "max_iter": params['max_iter'], 
                                "multi_class": 'auto', "verbose":0, "warm_start":False})
                                
    model.fit(train_set['X'], train_set['y'])

    # feature importance (for multitask, absolute value is computed for each feature)
    if cls.task == "multiclass":
        fi = np.mean(np.abs(model.coef_), axis=0).ravel()
    else:
        fi = model.coef_.ravel()

    return model, fi


# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, f1_score

class RunModel(object):
    """
    Model Fitting and Prediction Class:

    train_df : train pandas dataframe
    test_df : test pandas dataframe
    target : target column name (str)
    features : list of feature names
    categoricals : list of categorical feature names
    model : lgb, xgb, catb, linear, or nn
    task : options are ... regression, multiclass, or binary
    n_splits : K in KFold (default is 3)
    cv_method : options are ... KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold, StratifiedGroupKFold
    group : group feature name when GroupKFold or StratifiedGroupKFold are used
    parameter_tuning : bool, only for LGB
    seed : seed (int)
    scaler : options are ... None, MinMax, Standard
    verbose : bool
    """

    def __init__(self, train_df, test_df, target, features, categoricals=[],
                model="lgb", task="regression", n_splits=4, cv_method="KFold", 
                group=None, parameter_tuning=False, seed=1220, scaler=None, verbose=True):
        self.train_df = train_df
        self.test_df = test_df
        self.target = target
        self.features = features
        self.categoricals = categoricals
        self.model = model
        self.task = task
        self.n_splits = n_splits
        self.cv_method = cv_method
        self.group = group
        self.parameter_tuning = parameter_tuning
        self.seed = seed
        self.scaler = scaler
        self.verbose = verbose
        self.cv = self.get_cv()
        self.y_pred, self.score, self.model, self.oof, self.y_val, self.fi_df = self.fit()

    def train_model(self, train_set, val_set):

        # compile model
        if self.model == "lgb": # LGB             
            model, fi = lgb_model(self, train_set, val_set)

        elif self.model == "xgb": # xgb
            model, fi = xgb_model(self, train_set, val_set)

        elif self.model == "catb": # catboost
            model, fi = catb_model(self, train_set, val_set)

        elif self.model == "linear": # linear model
            model, fi = lin_model(self, train_set, val_set)

        elif self.model == "knn": # knn model
            model, fi = knn_model(self, train_set, val_set)

        elif self.model == "nn": # neural network
            model, fi = nn_model(self, train_set, val_set)
        
        return model, fi # fitted model and feature importance

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        if self.model == "lgb":
            train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
            val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
        else:
            if (self.model == "nn") & (self.task == "multiclass"):
                n_class = len(np.unique(self.train_df[self.target].values))
                train_set = {'X': x_train, 'y': onehot_target(y_train, n_class)}
                val_set = {'X': x_val, 'y': onehot_target(y_val, n_class)}
            else:
                train_set = {'X': x_train, 'y': y_train}
                val_set = {'X': x_val, 'y': y_val}
        return train_set, val_set

    def calc_metric(self, y_true, y_pred): # this may need to be changed based on the metric of interest
        if self.task == "multiclass":
            return f1_score(y_true, y_pred, average="macro")
        elif self.task == "binary":
            return roc_auc_score(y_true, y_pred) # log_loss
        elif self.task == "regression":
            return np.sqrt(mean_squared_error(y_true, y_pred))

    def get_cv(self):
        # return cv.split
        if self.cv_method == "KFold":
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return cv.split(self.train_df)
        elif self.cv_method == "StratifiedKFold":
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return cv.split(self.train_df, self.train_df[self.target])
        elif self.cv_method == "TimeSeriesSplit":
            cv = TimeSeriesSplit(max_train_size=None, n_splits=self.n_splits)
            return cv.split(self.train_df)
        elif self.cv_method == "GroupKFold":
            cv = GroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return cv.split(self.train_df, self.train_df[self.target], self.group)
        elif self.cv_method == "StratifiedGroupKFold":
            cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return cv.split(self.train_df, self.train_df[self.target], self.group)

    def fit(self):
        # initialize
        oof_pred = np.zeros((self.train_df.shape[0], ))
        y_vals = np.zeros((self.train_df.shape[0], ))
        y_pred = np.zeros((self.test_df.shape[0], ))

        # group does not kick in when group k fold is used
        if self.group is not None:
            if self.group in self.features:
                self.features.remove(self.group)
            if self.group in self.categoricals:
                self.categoricals.remove(self.group)
        fi = np.zeros((self.n_splits, len(self.features)))

        # scaling, if necessary
        if self.scaler is not None:
            # fill NaN
            numerical_features = [f for f in self.features if f not in self.categoricals]
            self.train_df[numerical_features] = self.train_df[numerical_features].fillna(self.train_df[numerical_features].median())
            self.test_df[numerical_features] = self.test_df[numerical_features].fillna(self.test_df[numerical_features].median())
            self.train_df[self.categoricals] = self.train_df[self.categoricals].fillna(self.train_df[self.categoricals].mode().iloc[0])
            self.test_df[self.categoricals] = self.test_df[self.categoricals].fillna(self.test_df[self.categoricals].mode().iloc[0])

            # scaling
            if self.scaler == "MinMax":
                scaler = MinMaxScaler()
            elif self.scaler == "Standard":
                scaler = StandardScaler()
            df = pd.concat([self.train_df[numerical_features], self.test_df[numerical_features]], ignore_index=True)
            scaler.fit(df[numerical_features])
            x_test = self.test_df.copy()
            x_test[numerical_features] = scaler.transform(x_test[numerical_features])
            if self.model == "nn":
                x_test = [np.absolute(x_test[i]) for i in self.categoricals] + [x_test[numerical_features]]
            else:
                x_test = x_test[self.features]
        else:
            x_test = self.test_df[self.features]

        # fitting with out of fold
        for fold, (train_idx, val_idx) in enumerate(self.cv):
            # train test split
            x_train, x_val = self.train_df.loc[train_idx, self.features], self.train_df.loc[val_idx, self.features]
            y_train, y_val = self.train_df.loc[train_idx, self.target], self.train_df.loc[val_idx, self.target]

            # fitting & get feature importance
            if self.scaler is not None:
                x_train[numerical_features] = scaler.transform(x_train[numerical_features])
                x_val[numerical_features] = scaler.transform(x_val[numerical_features])
                if self.model == "nn":
                    x_train = [np.absolute(x_train[i]) for i in self.categoricals] + [x_train[numerical_features]]
                    x_val = [np.absolute(x_val[i]) for i in self.categoricals] + [x_val[numerical_features]]

            # model fitting
            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
            model, importance = self.train_model(train_set, val_set)
            fi[fold, :] = importance
            y_vals[val_idx] = y_val

            # predictions
            oofs, ypred = get_oof_ypred(model, x_val, x_test, self.model, self.task)
            oof_pred[val_idx] = oofs.reshape(oof_pred[val_idx].shape)
            y_pred += ypred.reshape(y_pred.shape) / self.n_splits

            # check cv score
            print('Partial score of fold {} is: {}'.format(fold, self.calc_metric(y_vals[val_idx], oof_pred[val_idx])))

        # feature importance data frame
        fi_df = pd.DataFrame()
        for n in np.arange(self.n_splits):
            tmp = pd.DataFrame()
            tmp["features"] = self.features
            tmp["importance"] = fi[n, :]
            tmp["fold"] = n
            fi_df = pd.concat([fi_df, tmp], ignore_index=True)
        gfi = fi_df[["features", "importance"]].groupby(["features"]).mean().reset_index()
        fi_df = fi_df.merge(gfi, on="features", how="left", suffixes=('', '_mean'))

        # outputs
        loss_score = self.calc_metric(y_vals, oof_pred)
        if self.verbose:
            print('Our oof loss score is: ', loss_score)
        return y_pred, loss_score, model, oof_pred, y_vals, fi_df

    def plot_feature_importance(self, rank_range=[1, 50]):
        # plot feature importance
        _, ax = plt.subplots(1, 1, figsize=(10, 20))
        sorted_df = self.fi_df.sort_values(by = "importance_mean", ascending=False).reset_index().iloc[self.n_splits * (rank_range[0]-1) : self.n_splits * rank_range[1]]
        sns.barplot(data=sorted_df, x ="importance", y ="features", orient='h')
        ax.set_xlabel("feature importance")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return sorted_df


# # Fitting

# In[ ]:


features = test.columns.values.tolist()
dropcols = ['batch', 'batch_slices', 'batch_slices2', 'open_channels', 'time']
features = [f for f in features if f not in dropcols]
target = 'open_channels'
categoricals = []
group = "batch"


# In[ ]:


model = RunModel(train, test, target, features, categoricals=categoricals,
            model="linear", task="regression", n_splits=4, cv_method="StratifiedGroupKFold", 
            group=group, parameter_tuning=False, seed=1220, scaler="Standard", verbose=True)
print("Training done")


# # Optimize thresholds for F1

# In[ ]:


N = 2
reg_scores = np.zeros(N)
optRfs = []
for i in np.arange(N):
    optRf = OptimizedRounderF1()
    optRf.fit(model.oof, model.y_val)
    coefficientsf = optRf.coefficients()
    opt_valsf = optRf.predict(model.oof, coefficientsf)
    reg_score = f1_score(model.y_val, opt_valsf, average='macro')
    print(f"CV score round {i} (reg) = {reg_score}")
    optRfs.append(optRf)
    reg_scores[i] = reg_score
bst_i = np.argmax(reg_scores)
optRf = optRfs[bst_i]
coefficientsf = optRf.coefficients()
print(coefficientsf)
opt_valsf = optRf.predict(model.oof, coefficientsf)
opt_predsf = optRf.predict(model.y_pred, coefficientsf)


# # CV score

# In[ ]:


print(f"CV score = {model.score}")


# # Submit

# In[ ]:


del train, test
gc.collect()
submit = pd.read_csv("/kaggle/input/liverpool-ion-switching/sample_submission.csv")
submit["open_channels"] = opt_predsf
submit["open_channels"] = submit["open_channels"].astype(int)
submit.to_csv('submission_knnreg.csv', index=False, float_format='%.4f')
print("submission file saved!")


# In[ ]:


submit["open_channels"].hist(alpha=0.5)

