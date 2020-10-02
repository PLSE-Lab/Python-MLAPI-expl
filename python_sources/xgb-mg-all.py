#!/usr/bin/env python
# coding: utf-8

# In[35]:


# Importing core libraries
import os
import gc
import numpy as np
import pandas as pd
from time import time
import pprint
import joblib
import logging


# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")

# Boosting models
import catboost as cat
import xgboost as xgb
import lightgbm as lgb

# Hyperparameters distributions
from scipy.stats import randint
from scipy.stats import uniform

# Preprocesing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Utilities
from sklearn.pipeline import Pipeline
from tqdm import tqdm_notebook
from itertools import chain
from sklearn.model_selection import train_test_split

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns


# Model selection
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score


# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV
from skopt import gp_minimize # Bayesian optimization using Gaussian Processes
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args # decorator to convert a list of parameters to named arguments
from skopt.callbacks import DeadlineStopper # Stop the optimization before running out of a fixed budget of time.
from skopt.callbacks import VerboseCallback # Callback to control the verbosity
from skopt.callbacks import DeltaXStopper # Stop the optimization If the last two positions at which the objective has been evaluated are less than delta


# In[36]:


IS_LOCAL = False
if(IS_LOCAL):
    PATH="../input/LANL/"
else:
    PATH="../input/"
os.listdir(PATH)


# In[37]:


scaled_train_X = pd.read_csv('../input/all-features/train_X.csv')
scaled_test_X = pd.read_csv('../input/all-features/test_X.csv')
train_y = pd.read_csv('../input/all-features/train_y.csv')


# In[38]:


scaled_train_X.shape, scaled_test_X.shape, train_y.shape


# In[39]:


submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv', index_col='seg_id')
submission.shape


# In[40]:


# Converting average precision score into a scorer suitable for model selection
mse_scoring = make_scorer(mean_squared_error, greater_is_better=False)
# Setting a 5-fold stratified cross-validation (note: shuffle=True)
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)S
k_fold = KFold(n_splits=7, shuffle=True, random_state=13)


# In[41]:


# The objective function to be minimized
def make_objective(model, X, y, space, cv, scoring):
    # This decorator converts your objective function with named arguments into one that
    # accepts a list as argument, while doing the conversion automatically.
    @use_named_args(space) 
    def objective(**params):
        model.set_params(**params)
        return -np.mean(cross_val_score(model, 
                                        X, y, 
                                        cv=cv, 
                                        n_jobs=-1,
                                        scoring=scoring))

    return objective


# In[42]:


#xgb_bayes_params = best_params

xgb_bayes_params = {'colsample_bytree': 0.3706219857878677,
 'learning_rate': 0.16624226726409647,
 'max_bin': 93400,
 'max_depth': 134,
 'min_child_samples': 22,
 'min_child_weight': 4,
 'n_estimators': 8028,
 'num_leaves': 27,
 'reg_alpha': 1.081049236893711e-05,
 'reg_lambda': 1.043686239159047,
 'scale_pos_weight': 0.19222548462579486,
 'subsample': 0.6941640075502717,
 'subsample_for_bin': 375140,
 'subsample_freq': 7}
params=xgb_bayes_params


# In[43]:


counter = 0

xgb_model = xgb.XGBRegressor(
    metric='mae',
    n_jobs=1, 
    verbose=0
)

dimensions = [Real(0.01, 1.0, 'log-uniform', name='learning_rate'),
              Integer(2, 500, name='num_leaves'),
              Integer(0, 500, name='max_depth'),
              Integer(0, 200, name='min_child_samples'),
              Integer(100, 100000, name='max_bin'),
              Real(0.01, 1.0, 'uniform', name='subsample'),
              Integer(0, 10, name='subsample_freq'),
              Real(0.01, 1.0, 'uniform', name='colsample_bytree'),
              Integer(0, 10, name='min_child_weight'),
              Integer(100000, 500000, name='subsample_for_bin'),
              Real(1e-9, 1000, 'log-uniform', name='reg_lambda'),
              Real(1e-9, 1.0, 'log-uniform', name='reg_alpha'),
              Real(1e-6, 500, 'log-uniform', name='scale_pos_weight'),
              Integer(10, 10000, name='n_estimators')]

objective = make_objective(xgb_model,
                           scaled_train_X, train_y,
                           space=dimensions,
                           cv=k_fold,
                           scoring=mse_scoring)


# In[65]:


from typing import TypeVar, List, Dict, Tuple
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')

def run_xgb(
    X_tr: PandasDataFrame,
    X_val: PandasDataFrame,
    y_tr: PandasDataFrame,
    y_val: PandasDataFrame,
    test_data: PandasDataFrame,
    params: Dict
):
    """CV train lgb Booster.
    
    Args:
        params: Params for Booster.
        X_train: Training dataset.
        X_test: Testing dataset.
        
    Returns:
        model: Trained model.
        oof_train_lgb:  Training CV predictions.
        oof_test_lgb:  Testing CV predictions.
    """
    
    early_stop = 200
    num_rounds = 10000
    verbose_eval=1000

    d_train = xgb.DMatrix(X_tr.values, label = y_tr)
    d_valid = xgb.DMatrix(X_val.values, label = y_val )
    d_test = xgb.DMatrix(test_data.values) #+ mg
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    
    #1st run to get importance
    model = xgb.train(params, dtrain=d_train,num_boost_round=num_rounds,evals=watchlist, 
                      verbose_eval=verbose_eval,early_stopping_rounds=early_stop)    
    #importance
    imps=model.get_score(importance_type='gain')
    f_nums = model.feature_names
    f_names = list(scaled_train_X)
    f_dict = dict(zip(f_nums,f_names))
    keys=[]
    for i in imps:
        keys.append(f_dict.get(i))
    X_tr_new = X_tr.loc[:,keys]
    X_val_new=X_val.loc[:,keys]
    test_data_new=test_data.loc[:,keys]
    
    d_train = xgb.DMatrix(X_tr_new.values, label = y_tr)
    d_valid = xgb.DMatrix(X_val_new.values, label = y_val )
    d_test = xgb.DMatrix(test_data_new.values) #+ mg
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    
    model = xgb.train(params, dtrain=d_train,num_boost_round=num_rounds,evals=watchlist, 
                      verbose_eval=verbose_eval,early_stopping_rounds=early_stop) 

    val_pred = model.predict(d_valid)
    prediction = model.predict(d_test)
    
    return val_pred, prediction

def run_cv_model(
    train_data,
    train_target,
    test_data,
    model_fn, 
    params,
    eval_fn=None,
    label='model',
    feature_imp=False,
    n_folds = 5
):
    oof_val = np.zeros(len(train_data))
    predictions = np.zeros(len(test_data))
    oof_predict = np.zeros((n_folds, test_data.shape[0]))
    scores = []

   # feature_importance_df = pd.DataFrame()
    
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_columns = train_data.columns.values
    

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_data, train_target.values)):
        strLog = "fold {} for {}".format(fold_, label)
        print(strLog)
        X_tr, X_val = train_data.iloc[trn_idx], train_data.iloc[val_idx]
        y_tr, y_val = train_target.iloc[trn_idx], train_target.iloc[val_idx]
        
        val_pred, prediction= model_fn( #, feature_importances 
            X_tr, X_val,
            y_tr, y_val,
            test_data,
            params
        )
        score = mean_squared_error(y_val, val_pred)
        scores.append(score)

        
        oof_val[val_idx] = val_pred
        
        #feature importance
        if feature_imp == True:
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = train_columns
            fold_importance_df["importance"] = feature_importances[:len(train_columns)]
            fold_importance_df["fold"] = fold_ + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        #predictions
        oof_predict[fold_] = prediction
        predictions += prediction/folds.n_splits
        
        print('CV score: {0:.4f}, std: {1:.4f}.\n'.format(np.mean(score), np.std(score)))  
        print('CV mean score: {0:.4f}, std: {1:.4f}.\n'.format(np.mean(scores), np.std(scores)))  


            
    return oof_val, oof_predict, predictions #, feature_importance_df


# In[66]:


fix_xgb_params = {
    "boosting": "gblinear",
    "metric": 'mae',
}

xgb_params = {**fix_xgb_params, **xgb_bayes_params}


oof_val, oof_predict, predictions  = run_cv_model( #extra param: '', feature_importance_df'''
    train_data=scaled_train_X,
    train_target=train_y,
    test_data=scaled_test_X,
    model_fn=run_xgb,
    params=xgb_params,
    eval_fn=None,
    label="xgb",
    feature_imp=False,
    n_folds=10
)


# In[ ]:


submission.time_to_failure = predictions
submission.to_csv('submission.csv',index=True)

