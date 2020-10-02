#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pickle
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

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, f1_score
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import optuna

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
import warnings
warnings.filterwarnings('ignore')


# # Load data

# In[ ]:


train = pd.read_csv("/kaggle/input/killer-shrimp-invasion/train.csv")
test = pd.read_csv("/kaggle/input/killer-shrimp-invasion/test.csv")


# In[ ]:


print(train.shape)
train.head()


# In[ ]:


print(test.shape)
test.head()


# # Modeling Pipeline
# The following is a part of my modeling pipeline (https://github.com/katsu1110/DataScienceComp).

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


# # Model Fitting

# In[ ]:


features = test.columns.values.tolist()
categoricals = []
target = "Presence"
group = None
dropcols = ["pointid"]
features = [f for f in features if f not in dropcols]
categoricals = [f for f in categoricals if f not in dropcols]


# In[ ]:


model = RunModel(train, test, target, features, categoricals=categoricals,
            model="linear", task="binary", n_splits=4, cv_method="StratifiedKFold", 
            group=group, parameter_tuning=False, seed=1220, scaler="Standard", verbose=True)
print("Training done")


# # Feature importance
# I mean, coefficients.

# In[ ]:


fi = model.plot_feature_importance()


# # Submission

# In[ ]:


sub = pd.read_csv("/kaggle/input/killer-shrimp-invasion/temperature_submission.csv")
sub["Presence"] = model.y_pred
sub.to_csv("submission.csv", index=False)
sub.head()


# In[ ]:


sub["Presence"].hist()

