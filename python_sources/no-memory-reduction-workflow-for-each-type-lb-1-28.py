#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# If you feel 16GB memory is too small for >200 features bond-wise, you are not alone. In this kernel, we updated Andrew's workflow by generating features for each type to save memory. No `reduce_mem_usage` by Andrew is used (except when importing QM9), so every computation is retained its `np.float64` default accuracy. 
# 
# Within the loop of the training for each type, first the features are generated for a minimal passed train/test dataframe, then [Giba's features](https://www.kaggle.com/scaomath/lgb-giba-features-qm9-custom-objective-in-python) are loaded using the format of a kernel I made earlier. Finally the [Yukawa potentials](https://www.kaggle.com/scaomath/parallelization-of-coulomb-yukawa-interaction) are added as well using `structures` dataframe by type to save a tons of memory.
# 
# Just wrapping your feature generation into a function you are good to go.
# 
# 
# ### References:
# - [Brute force feature engineering](https://www.kaggle.com/artgor/brute-force-feature-engineering)
# - [Keras Neural Net for CHAMPS](https://www.kaggle.com/todnewman/keras-neural-net-for-champs)
# - [Giba R + data.table + Simple Features](https://www.kaggle.com/titericz/giba-r-data-table-simple-features-1-17-lb)

# # Libraries and functions

# In[ ]:


import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import lightgbm as lgb
import time
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn import metrics
from sklearn import linear_model
import gc
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


## Andrew's utils
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
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()
    

def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
    
    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    
    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]
    
    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'sklearn_scoring_function': metrics.mean_absolute_error},
                    'group_mae': {'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'scoring_function': group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                        'catboost_metric_name': 'MSE',
                        'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    
    result_dict = {}
    
    # out-of-fold predictions on train data
    oof = np.zeros(len(X))
    
    # averaged predictions on train data
    prediction = np.zeros(len(X_test))
    
    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred    
        
        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= folds.n_splits
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= folds.n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            
            result_dict['feature_importance'] = feature_importance
        
    return result_dict


# ## Data loading and overview

# In[ ]:


file_folder = '../input/champs-scalar-coupling' 
train = pd.read_csv(f'{file_folder}/train.csv')
test = pd.read_csv(f'{file_folder}/test.csv')
sub = pd.read_csv(f'{file_folder}/sample_submission.csv')
structures = pd.read_csv(f'{file_folder}/structures.csv')


# In[ ]:


path = '../input/parallelization-of-coulomb-yukawa-interaction'
structures_yukawa = pd.read_csv(f'{path}/structures_yukawa.csv')
structures = pd.concat([structures, structures_yukawa], axis=1)
del structures_yukawa


# In[ ]:


y = train['scalar_coupling_constant']
train = train.drop(columns = ['scalar_coupling_constant'])


# # Feature generation funcs
# 
# The features here are:
# 
# - First the `type` is encoded by a label encoder.
# - The merging template and selected features from [Andrew's brute force feature engineering](https://www.kaggle.com/artgor/brute-force-feature-engineering)
# - Cosine features originally from [Effective feature](https://www.kaggle.com/kmat2019/effective-feature) and expanded in [Keras Neural Net for CHAMPS](https://www.kaggle.com/todnewman/keras-neural-net-for-champs), I simplified the generation procedure by removing unnecessary `pandas` operations since vanilla `numpy` arrays operation is faster.
# - QM9 dataset from [Quantum Machine 9 - QM9](https://www.kaggle.com/zaharch/quantum-machine-9-qm9).
# - Parallelization computed [Yukawa potentials](https://www.kaggle.com/scaomath/parallelization-of-coulomb-yukawa-interaction).
# - Giba's features from [Giba R + data.table + Simple Features](https://www.kaggle.com/titericz/giba-r-data-table-simple-features-1-17-lb), which I now export the features to a dataset: [Giba molecular features](https://www.kaggle.com/scaomath/giba-molecular-features).

# In[ ]:


def map_atom_info(df_1, df_2, atom_idx):
    df = pd.merge(df_1, df_2, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    return df

    
def find_dist(df):
    df_p_0 = df[['x_0', 'y_0', 'z_0']].values
    df_p_1 = df[['x_1', 'y_1', 'z_1']].values
    
    df['dist'] = np.linalg.norm(df_p_0 - df_p_1, axis=1)
    df['dist_inv2'] = 1/df['dist']**2
    df['dist_x'] = (df['x_0'] - df['x_1']) ** 2
    df['dist_y'] = (df['y_0'] - df['y_1']) ** 2
    df['dist_z'] = (df['z_0'] - df['z_1']) ** 2
    return df

def find_closest_atom(df):    
    df_temp = df.loc[:,["molecule_name",
                      "atom_index_0","atom_index_1",
                      "dist","x_0","y_0","z_0","x_1","y_1","z_1"]].copy()
    df_temp_ = df_temp.copy()
    df_temp_ = df_temp_.rename(columns={'atom_index_0': 'atom_index_1',
                                       'atom_index_1': 'atom_index_0',
                                       'x_0': 'x_1',
                                       'y_0': 'y_1',
                                       'z_0': 'z_1',
                                       'x_1': 'x_0',
                                       'y_1': 'y_0',
                                       'z_1': 'z_0'})
    df_temp_all = pd.concat((df_temp,df_temp_),axis=0)

    df_temp_all["min_distance"]=df_temp_all.groupby(['molecule_name', 
                                                     'atom_index_0'])['dist'].transform('min')
    df_temp_all["max_distance"]=df_temp_all.groupby(['molecule_name', 
                                                     'atom_index_0'])['dist'].transform('max')
    
    df_temp = df_temp_all[df_temp_all["min_distance"]==df_temp_all["dist"]].copy()
    df_temp = df_temp.drop(['x_0','y_0','z_0','min_distance'], axis=1)
    df_temp = df_temp.rename(columns={'atom_index_0': 'atom_index',
                                         'atom_index_1': 'atom_index_closest',
                                         'dist': 'distance_closest',
                                         'x_1': 'x_closest',
                                         'y_1': 'y_closest',
                                         'z_1': 'z_closest'})
    df_temp = df_temp.drop_duplicates(subset=['molecule_name', 'atom_index'])
    
    for atom_idx in [0,1]:
        df = map_atom_info(df,df_temp, atom_idx)
        df = df.rename(columns={'atom_index_closest': f'atom_index_closest_{atom_idx}',
                                        'distance_closest': f'distance_closest_{atom_idx}',
                                        'x_closest': f'x_closest_{atom_idx}',
                                        'y_closest': f'y_closest_{atom_idx}',
                                        'z_closest': f'z_closest_{atom_idx}'})
        
    df_temp= df_temp_all[df_temp_all["max_distance"]==df_temp_all["dist"]].copy()
    df_temp = df_temp.drop(['x_0','y_0','z_0','max_distance'], axis=1)
    df_temp= df_temp.rename(columns={'atom_index_0': 'atom_index',
                                         'atom_index_1': 'atom_index_farthest',
                                         'dist': 'distance_farthest',
                                         'x_1': 'x_farthest',
                                         'y_1': 'y_farthest',
                                         'z_1': 'z_farthest'})
    df_temp = df_temp.drop_duplicates(subset=['molecule_name', 'atom_index'])
        
    for atom_idx in [0,1]:
        df = map_atom_info(df,df_temp, atom_idx)
        df = df.rename(columns={'atom_index_farthest': f'atom_index_farthest_{atom_idx}',
                                        'distance_farthest': f'distance_farthest_{atom_idx}',
                                        'x_farthest': f'x_farthest_{atom_idx}',
                                        'y_farthest': f'y_farthest_{atom_idx}',
                                        'z_farthest': f'z_farthest_{atom_idx}'})
    return df


def add_cos_features(df):
    
    df["distance_center0"] = np.sqrt((df['x_0']-df['c_x'])**2                                    + (df['y_0']-df['c_y'])**2                                    + (df['z_0']-df['c_z'])**2)
    df["distance_center1"] = np.sqrt((df['x_1']-df['c_x'])**2                                    + (df['y_1']-df['c_y'])**2                                    + (df['z_1']-df['c_z'])**2)
    
    df['distance_c0'] = np.sqrt((df['x_0']-df['x_closest_0'])**2 +                                 (df['y_0']-df['y_closest_0'])**2 +                                 (df['z_0']-df['z_closest_0'])**2)
    df['distance_c1'] = np.sqrt((df['x_1']-df['x_closest_1'])**2 +                                 (df['y_1']-df['y_closest_1'])**2 +                                 (df['z_1']-df['z_closest_1'])**2)
    
    df["distance_f0"] = np.sqrt((df['x_0']-df['x_farthest_0'])**2 +                                 (df['y_0']-df['y_farthest_0'])**2 +                                 (df['z_0']-df['z_farthest_0'])**2)
    df["distance_f1"] = np.sqrt((df['x_1']-df['x_farthest_1'])**2 +                                 (df['y_1']-df['y_farthest_1'])**2 +                                 (df['z_1']-df['z_farthest_1'])**2)
    
    vec_center0_x = (df['x_0']-df['c_x'])/(df["distance_center0"]+1e-10)
    vec_center0_y = (df['y_0']-df['c_y'])/(df["distance_center0"]+1e-10)
    vec_center0_z = (df['z_0']-df['c_z'])/(df["distance_center0"]+1e-10)
    
    vec_center1_x = (df['x_1']-df['c_x'])/(df["distance_center1"]+1e-10)
    vec_center1_y = (df['y_1']-df['c_y'])/(df["distance_center1"]+1e-10)
    vec_center1_z = (df['z_1']-df['c_z'])/(df["distance_center1"]+1e-10)
    
    vec_c0_x = (df['x_0']-df['x_closest_0'])/(df["distance_c0"]+1e-10)
    vec_c0_y = (df['y_0']-df['y_closest_0'])/(df["distance_c0"]+1e-10)
    vec_c0_z = (df['z_0']-df['z_closest_0'])/(df["distance_c0"]+1e-10)
    
    vec_c1_x = (df['x_1']-df['x_closest_1'])/(df["distance_c1"]+1e-10)
    vec_c1_y = (df['y_1']-df['y_closest_1'])/(df["distance_c1"]+1e-10)
    vec_c1_z = (df['z_1']-df['z_closest_1'])/(df["distance_c1"]+1e-10)
    
    vec_f0_x = (df['x_0']-df['x_farthest_0'])/(df["distance_f0"]+1e-10)
    vec_f0_y = (df['y_0']-df['y_farthest_0'])/(df["distance_f0"]+1e-10)
    vec_f0_z = (df['z_0']-df['z_farthest_0'])/(df["distance_f0"]+1e-10)
    
    vec_f1_x = (df['x_1']-df['x_farthest_1'])/(df["distance_f1"]+1e-10)
    vec_f1_y = (df['y_1']-df['y_farthest_1'])/(df["distance_f1"]+1e-10)
    vec_f1_z = (df['z_1']-df['z_farthest_1'])/(df["distance_f1"]+1e-10)
    
    vec_x = (df['x_1']-df['x_0'])/df['dist']
    vec_y = (df['y_1']-df['y_0'])/df['dist']
    vec_z = (df['z_1']-df['z_0'])/df['dist']
    
    df["cos_c0_c1"] = vec_c0_x*vec_c1_x + vec_c0_y*vec_c1_y + vec_c0_z*vec_c1_z
    df["cos_f0_f1"] = vec_f0_x*vec_f1_x + vec_f0_y*vec_f1_y + vec_f0_z*vec_f1_z
    
    df["cos_c0_f0"] = vec_c0_x*vec_f0_x + vec_c0_y*vec_f0_y + vec_c0_z*vec_f0_z
    df["cos_c1_f1"] = vec_c1_x*vec_f1_x + vec_c1_y*vec_f1_y + vec_c1_z*vec_f1_z
    
    df["cos_center0_center1"] = vec_center0_x*vec_center1_x                               + vec_center0_y*vec_center1_y                               + vec_center0_z*vec_center1_z
    
    df["cos_c0"] = vec_c0_x*vec_x + vec_c0_y*vec_y + vec_c0_z*vec_z
    df["cos_c1"] = vec_c1_x*vec_x + vec_c1_y*vec_y + vec_c1_z*vec_z
    
    df["cos_f0"] = vec_f0_x*vec_x + vec_f0_y*vec_y + vec_f0_z*vec_z
    df["cos_f1"] = vec_f1_x*vec_x + vec_f1_y*vec_y + vec_f1_z*vec_z
    
    df["cos_center0"] = vec_center0_x*vec_x + vec_center0_y*vec_y + vec_center0_z*vec_z
    df["cos_center1"] = vec_center1_x*vec_x + vec_center1_y*vec_y + vec_center1_z*vec_z

    return df

def add_dist_features(df):
    # Andrew's features selected
    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')
    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']
    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')
    df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']
    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')

    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')
    df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']
    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')
    df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']
    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')
    
    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')
    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']
    
    return df


def dummies(df, list_cols):
    for col in list_cols:
        df_dummies = pd.get_dummies(df[col], drop_first=True, 
                                    prefix=(str(col)))
        df = pd.concat([df, df_dummies], axis=1)
    return df


def add_qm9_features(df):
    data_qm9 = pd.read_pickle('../input/quantum-machine-9-qm9/data.covs.pickle')
    to_drop = ['type', 
               'linear', 
               'atom_index_0', 
               'atom_index_1', 
               'scalar_coupling_constant', 
               'U', 'G', 'H', 
               'mulliken_mean', 'r2', 'U0']
    data_qm9 = data_qm9.drop(columns = to_drop, axis=1)
    data_qm9 = reduce_mem_usage(data_qm9,verbose=False)
    df = pd.merge(df, data_qm9, how='left', on=['molecule_name','id'])
    del data_qm9
    
    df = dummies(df, ['type', 'atom_1'])
    return df

def get_features(df, struct):
    for atom_idx in [0,1]:
        df = map_atom_info(df, struct, atom_idx)
        df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
        struct['c_x'] = struct.groupby('molecule_name')['x'].transform('mean')
        struct['c_y'] = struct.groupby('molecule_name')['y'].transform('mean')
        struct['c_z'] = struct.groupby('molecule_name')['z'].transform('mean')

    df = find_dist(df)
    df = find_closest_atom(df)
    df = add_cos_features(df)
    df = add_dist_features(df)
    df = add_qm9_features(df)
    
    return df


# In[ ]:


good_columns = ['type',
 'dist_C_0_x',
 'dist_C_1_x',
 'dist_C_2_x',
 'dist_C_3_x',
 'dist_C_4_x',
 'dist_F_0_x',
 'dist_F_1_x',
 'dist_F_2_x',
 'dist_F_3_x',
 'dist_F_4_x',
 'dist_H_0_x',
 'dist_H_1_x',
 'dist_H_2_x',
 'dist_H_3_x',
 'dist_H_4_x',
 'dist_N_0_x',
 'dist_N_1_x',
 'dist_N_2_x',
 'dist_N_3_x',
 'dist_N_4_x',
 'dist_O_0_x',
 'dist_O_1_x',
 'dist_O_2_x',
 'dist_O_3_x',
 'dist_O_4_x',
 'dist_C_0_y',
 'dist_C_1_y',
 'dist_C_2_y',
 'dist_C_3_y',
 'dist_C_4_y',
 'dist_F_0_y',
 'dist_F_1_y',
 'dist_F_2_y',
 'dist_F_3_y',
 'dist_F_4_y',
 'dist_H_0_y',
 'dist_H_1_y',
 'dist_H_2_y',
 'dist_H_3_y',
 'dist_H_4_y',
 'dist_N_0_y',
 'dist_N_1_y',
 'dist_N_2_y',
 'dist_N_3_y',
 'dist_N_4_y',
 'dist_O_0_y',
 'dist_O_1_y',
 'dist_O_2_y',
 'dist_O_3_y',
 'dist_O_4_y',
 'dist_inv2',
 'distance_closest_0',
 'distance_closest_1',
 'distance_farthest_0',
 'distance_farthest_1',
 'cos_c0_c1', 'cos_f0_f1','cos_c0_f0', 'cos_c1_f1',
 'cos_center0_center1', 'cos_c0', 'cos_c1', 'cos_f0', 'cos_f1',
 'cos_center0', 'cos_center1',
 'molecule_atom_index_0_dist_mean',
 'molecule_atom_index_0_dist_mean_diff',
 'molecule_atom_index_0_dist_min',
 'molecule_atom_index_0_dist_min_diff',
 'molecule_atom_index_0_dist_std',
 'molecule_atom_index_1_dist_mean',
 'molecule_atom_index_1_dist_mean_diff',
 'molecule_atom_index_1_dist_min',
 'molecule_atom_index_1_dist_min_diff',
 'molecule_atom_index_1_dist_std',
 'molecule_type_dist_mean',
 'molecule_type_dist_mean_diff',
 'rc_A',
 'rc_B',
 'rc_C',
 'mu',
 'alpha',
 'homo',
 'lumo',
 'gap',
 'zpve',
 'Cv',
 'freqs_min',
 'freqs_max',
 'freqs_mean',
 'mulliken_min',
 'mulliken_max',
 'mulliken_atom_0',
 'mulliken_atom_1']

giba_columns = ['inv_dist0',
 'inv_dist1',
 'inv_distP',
 'inv_dist0R',
 'inv_dist1R',
 'inv_distPR',
 'inv_dist0E',
 'inv_dist1E',
 'inv_distPE',
 'linkM0',
 'linkM1',
 'min_molecule_atom_0_dist_xyz',
 'mean_molecule_atom_0_dist_xyz',
 'max_molecule_atom_0_dist_xyz',
 'sd_molecule_atom_0_dist_xyz',
 'min_molecule_atom_1_dist_xyz',
 'mean_molecule_atom_1_dist_xyz',
 'max_molecule_atom_1_dist_xyz',
 'sd_molecule_atom_1_dist_xyz',
 'coulomb_C.x',
 'coulomb_F.x',
 'coulomb_H.x',
 'coulomb_N.x',
 'coulomb_O.x',
 'yukawa_C.x',
 'yukawa_F.x',
 'yukawa_H.x',
 'yukawa_N.x',
 'yukawa_O.x',
 'vander_C.x',
 'vander_F.x',
 'vander_H.x',
 'vander_N.x',
 'vander_O.x',
 'coulomb_C.y',
 'coulomb_F.y',
 'coulomb_H.y',
 'coulomb_N.y',
 'coulomb_O.y',
 'yukawa_C.y',
 'yukawa_F.y',
 'yukawa_H.y',
 'yukawa_N.y',
 'yukawa_O.y',
 'vander_C.y',
 'vander_F.y',
 'vander_H.y',
 'vander_N.y',
 'vander_O.y',
 'distC0',
 'distH0',
 'distN0',
 'distC1',
 'distH1',
 'distN1',
 'adH1',
 'adH2',
 'adH3',
 'adH4',
 'adC1',
 'adC2',
 'adC3',
 'adC4',
 'adN1',
 'adN2',
 'adN3',
 'adN4',
 'NC',
 'NH',
 'NN',
 'NF',
 'NO']


# In[ ]:


lbl = LabelEncoder()
lbl.fit(list(train['type'].values) + list(test['type'].values))
train['type'] = lbl.transform(list(train['type'].values))
test['type'] = lbl.transform(list(test['type'].values))


# # Training by type with time seed
# We use different numbers of iterations for different type, after running the label encoder
# ```
# train['type'].unique() = [0, 3, 1, 4, 2, 6, 5, 7]
# ```
# Hence the current the number of iteration `N` config (in order) is:
# 
# > Type 0 = `1JHC`. <br>
# > Type 1 = `1JHN`. <br>
# > Type 2 = `2JHC`. <br>
# > Type 3 = `2JHH`. <br>
# > Type 4 = `2JHN`. <br>
# > Type 5 = `3JHC`. <br>
# > Type 6 = `3JHH`. <br>
# > Type 7 = `3JHN`. <br>

# In[ ]:


n_fold = 3
seed = round(time.time())
folds = KFold(n_splits=n_fold, shuffle=True, random_state=seed)

params = {'num_leaves': 400,
          'objective': 'huber',
          'max_depth': 9,
          'learning_rate': 0.12,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.8,
          "metric": 'mae',
          "verbosity": -1,
          'lambda_l1': 0.8,
          'lambda_l2': 0.2,
          'feature_fraction': 0.6,
         }


# In[ ]:


X_short = pd.DataFrame({'ind': list(train.index), 
                        'type': train['type'].values,
                        'oof': [0] * len(train), 
                        'target': y.values})
X_short_test = pd.DataFrame({'ind': list(test.index), 
                             'type': test['type'].values, 
                             'prediction': [0] * len(test)})


# In[ ]:


get_ipython().run_cell_magic('time', '', "CV_score = 0\n###Iters###    [1JHC, 1JHN, 2JHC, 2JHH, 2JHN, 3JHC, 3JHH, 3JHN]\nn_estimators = [5000, 2500, 3000, 2500, 2500, 3000, 2500, 2500]\n\n\nfor t in train['type'].unique():\n    type_ = lbl.inverse_transform([t])[0]\n    print(f'\\nTraining of type {t}: {type_}.')\n    index_type = (train['type'] == t)\n    index_type_test = (test['type'] == t)\n    \n    X_t = train.loc[index_type].copy()\n    X_test_t = test.loc[index_type_test].copy()\n    y_t = y[index_type]\n    \n    print(f'Generating features...')\n    start_time = time.time()\n    \n    ## Generating features from the public kernels, just by type\n    ## no memory reduction is needed\n    X_t = get_features(X_t, structures.copy())\n    X_t = X_t[good_columns].fillna(0.0)\n    \n    X_test_t = get_features(X_test_t, structures.copy())\n    X_test_t = X_test_t[good_columns].fillna(0.0)\n    \n    ## load Giba's features just for type t by getting rows to be excluded when initiating read_csv\n    rows_to_exclude = np.where(index_type==False)[0]+1 # retain the header row\n    rows_to_exclude_test = np.where(index_type_test==False)[0]+1\n    train_giba_t = pd.read_csv('../input/giba-molecular-features/train_giba.csv/train_giba.csv',\n                        header=0, skiprows=rows_to_exclude, usecols=giba_columns)\n    test_giba_t = pd.read_csv('../input/giba-molecular-features/test_giba.csv/test_giba.csv',\n                       header=0, skiprows=rows_to_exclude_test, usecols=giba_columns)\n    \n    X_t = pd.concat((X_t, train_giba_t), axis=1)\n\n    X_test_t = pd.concat((X_test_t,test_giba_t), axis=1) \n    \n    del train_giba_t, test_giba_t\n    gc.collect()\n    \n    print(f'Done in {(time.time() - start_time):.2f} seconds for {X_t.shape[1]} features.')\n    ## feature generation done\n    \n    \n    result_dict_lgb = train_model_regression(X=X_t, X_test=X_test_t, \n                                              y=y_t, params=params, \n                                              folds=folds, \n                                              model_type='lgb', \n                                              eval_metric='mae', \n                                              plot_feature_importance=False,\n                                              verbose=2000, early_stopping_rounds=200, \n                                              n_estimators=n_estimators[t])\n    del X_t, X_test_t\n    gc.collect()\n    \n    X_short.loc[X_short['type'] == t, 'oof'] = result_dict_lgb['oof']\n    X_short_test.loc[X_short_test['type'] == t, 'prediction'] = result_dict_lgb['prediction']\n    \n    ## manually computing the cv score\n    CV_score += np.log(np.array(result_dict_lgb['scores']).mean())/8 # total 8 types")


# In[ ]:


sub['scalar_coupling_constant'] = X_short_test['prediction']
today = str(datetime.date.today())
sub.to_csv(f'LGB_{today}_{CV_score:.4f}.csv', index=False)

