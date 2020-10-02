#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###############################################
# Import Machine Learning Assets
###############################################
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

###############################################
# Import Miscellaneous Assets
###############################################
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from functools import partial
from pprint import pprint as pp
from tqdm import tqdm, tqdm_notebook

pd.set_option('display.expand_frame_repr', False)

###############################################
# Declare Global Variables
###############################################
CROSS_VALIDATION_PARAMS = dict(n_splits=5, shuffle=True, random_state=32)
XGBOOST_REGRESSOR_PARAMS = dict(
    learning_rate=0.2, n_estimators=200, subsample=0.8, colsample_bytree=0.8, 
    max_depth=10, n_jobs=-1
)

BAYESIAN_OPTIMIZATION_MAXIMIZE_PARAMS = dict(
    init_points=1,  # init_points=20,
    n_iter=2,  # n_iter=60,
    acq='poi', xi=0.0
)
BAYESIAN_OPTIMIZATION_BOUNDARIES = dict(
    max_depth=(5, 12.99),
    gamma=(0.01, 5),
    min_child_weight=(0, 6),
    scale_pos_weight=(1.2, 5),
    reg_alpha=(4.0, 10.0),
    reg_lambda=(1.0, 10.0),
    max_delta_step=(0, 5),
    subsample=(0.5, 1.0),
    colsample_bytree=(0.3, 1.0),
    learning_rate=(0.0, 1.0)
)
BAYESIAN_OPTIMIZATION_INITIAL_SEARCH_POINTS = dict(
    max_depth=[5, 10],
    gamma=[0.1511, 3.8463],
    min_child_weight=[2.4073, 4.9954],
    scale_pos_weight=[2.2281, 4.0345],
    reg_alpha=[8.0702, 9.0573],
    reg_lambda=[2.0126, 3.5934],
    max_delta_step=[1, 2],
    subsample=[0.8, 0.8234],
    colsample_bytree=[0.8, 0.7903],
    learning_rate=[0.2, 0.1]
)

reserve_features = [
    'rs1_x', 'rs1_y', 'rs2_x', 'rs2_y', 'rv1_x', 'rv1_y', 'rv2_x', 'rv2_y',
    'total_reserve_dt_diff_mean', 'total_reserve_mean', 'total_reserve_sum'
]

BASE_ESTIMATOR = partial(XGBRegressor)
# train_df, test_df = None, None
# oof_predictions, test_predictions = None, None
# train_input = None
# best_round = None
# target = None


# # Bayesian Optimization Parameter Overview
# 
# - BAYESIAN_OPTIMIZATION_MAXIMIZE_PARAMS:
#  - init_points: Number of random samples to conduct before maximizing a function
#  - n_iter: Number of iterations to fit the Gaussian Process object to maximize our function
# - BAYESIAN_OPTIMIZATION_BOUNDARIES:
#  - This needs to be a dictionary containing all of the hyperparameters you want to search, along with their minimum and maximum boundaries
# - BAYESIAN_OPTIMIZATION_INITIAL_SEARCH_POINTS:
#  - This is a dictionary containing some points to search first to give the Gaussian Process a sort of baseline for the hyperparameters' values. I've gotten the best results by setting my initial search points to cover a very wide range of values, rather than focusing in specific areas, but your experience may differ. 
#  - All of the list values in this dictionary must be of the same length.
# 
# # Data Preparation/Feature Engineering 
# 
# The next cell will contain all the feature engineering code that you've all seen hundreds of times by now.
# <br>
# Thank you very much to everyone who contributed to this beast. Your work is truly appreciated.
# <br>
# After doing our feature engineering, we'll get back to what this kernel is really about: Bayesian Hyperparameter Optimization. 

# In[ ]:


data = {
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
}

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])
for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_dow'] = data[df]['visit_datetime'].dt.dayofweek
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(
        lambda _: (_['visit_datetime'] - _['reserve_datetime']).days, axis=1
    )
    
    ###############################################
    # aharless's Same-Week Reservation Exclusion
    ###############################################
    data[df] = data[df][data[df]['reserve_datetime_diff'] > data[df]['visit_dow']]
    tmp1 = data[df].groupby(
        ['air_store_id','visit_datetime'], as_index=False
    )[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={
        'visit_datetime':'visit_date', 
        'reserve_datetime_diff': 'rs1', 
        'reserve_visitors':'rv1'
    })
    tmp2 = data[df].groupby(
        ['air_store_id','visit_datetime'], as_index=False
    )[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={
        'visit_datetime':'visit_date', 
        'reserve_datetime_diff': 'rs2', 
        'reserve_visitors':'rv2'
    })
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda _: str(_).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda _: '_'.join(_.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat(
    [pd.DataFrame({
        'air_store_id': unique_stores, 
        'dow': [_] * len(unique_stores)
    }) for _ in range(7)], 
    axis=0, ignore_index=True
).reset_index(drop=True)

###############################################
# Jerome Vallet's Optimization
###############################################
tmp = data['tra'].groupby(['air_store_id', 'dow']).agg(
    {'visitors': [np.min, np.mean, np.median, np.max, np.size]}
).reset_index()
tmp.columns = [
    'air_store_id', 'dow', 'min_visitors', 'mean_visitors', 'median_visitors', 'max_visitors', 
    'count_observations'
]
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])

###############################################
# Georgii Vyshnia's Features
###############################################
stores['air_genre_name'] = stores['air_genre_name'].map(lambda _: str(str(_).replace('/',' ')))
stores['air_area_name'] = stores['air_area_name'].map(lambda _: str(str(_).replace('-',' ')))
lbl = LabelEncoder()
for i in range(10):
    stores['air_genre_name' + str(i)] = lbl.fit_transform(stores['air_genre_name'].map(
        lambda _: str(str(_).split(' ')[i]) if len(str(_).split(' ')) > i else ''
    ))
    stores['air_area_name' + str(i)] = lbl.fit_transform(stores['air_area_name'].map(
        lambda _: str(str(_).split(' ')[i]) if len(str(_).split(' ')) > i else ''
    ))
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 

train = pd.merge(train, stores, how='inner', on=['air_store_id','dow']) 
test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id', 'visit_date']) 
    test = pd.merge(test, data[df], how='left', on=['air_store_id', 'visit_date'])

train['id'] = train.apply(
    lambda _: '_'.join([str(_['air_store_id']), str(_['visit_date'])]), axis=1
)

train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

###############################################
# JMBULL's Features 
###############################################
train['date_int'] = train['visit_date'].apply(lambda _: _.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda _: _.strftime('%Y%m%d')).astype(int)
train['var_max_lat'] = train['latitude'].max() - train['latitude']
train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = test['latitude'].max() - test['latitude']
test['var_max_long'] = test['longitude'].max() - test['longitude']

###############################################
# Georgii Vyshnia's Features
###############################################
train['lon_plus_lat'] = train['longitude'] + train['latitude'] 
test['lon_plus_lat'] = test['longitude'] + test['latitude']

lbl = LabelEncoder()
train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
test['air_store_id2'] = lbl.transform(test['air_store_id'])

col = [_ for _ in train if _ not in ['id', 'air_store_id', 'visit_date','visitors']]
train = train.fillna(-1)
test = test.fillna(-1)

train_df = train[col]
test_df = test[col]
target = pd.DataFrame()
target['visitors'] = np.log1p(train['visitors'].values)


# # The Function to be Maximized

# In[ ]:


def search_node(**kwargs):
    # global train_df, test_df, train_input, oof, test_predictions, best_round, target
    global train_df, target

    ###############################################
    # Unify Parameters
    ###############################################
    received_params = dict(dict(
        n_estimators=200,
    ), **{_k: _v if _k not in ('max_depth') else int(_v) for _k, _v in kwargs.items()})
    
    current_params = dict(XGBOOST_REGRESSOR_PARAMS, **received_params)

    ###############################################
    # Initialize Folds and Result Placeholders
    ###############################################
    folds = KFold(**CROSS_VALIDATION_PARAMS)
    evaluation = np.zeros((current_params['n_estimators'], CROSS_VALIDATION_PARAMS['n_splits']))
    oof_predictions = np.empty(len(train_df))
    np.random.seed(32)

    progress_bar = tqdm_notebook(
        enumerate(folds.split(target, target)), 
        total=CROSS_VALIDATION_PARAMS['n_splits'], 
        leave=False
    )
    
    ###############################################
    # Begin Cross-Validation
    ###############################################
    for fold, (train_index, validation_index) in progress_bar:
        train_input, validation_input = train_df.iloc[train_index], train_df.iloc[validation_index]
        train_target, validation_target = target.iloc[train_index], target.iloc[validation_index]

        ###############################################
        # Initialize and Fit Model With Current Parameters
        ###############################################
        model = BASE_ESTIMATOR(**current_params)
        eval_set = [(train_input, train_target), (validation_input, validation_target)]
        model.fit(train_input, train_target, eval_set=eval_set, verbose=False)

        ###############################################
        # Find Best Round for Validation Set
        ###############################################
        evaluation[:, fold] = model.evals_result_["validation_1"]['rmse']
        best_round = np.argsort(evaluation[:, fold])[0]

        progress_bar.set_description('Fold #{}:   {:.5f}'.format(
            fold, evaluation[:, fold][best_round]
        ), refresh=True)

    ###############################################
    # Compute Mean and Standard Deviation of RMSLE
    ###############################################
    mean_eval, std_eval = np.mean(evaluation, axis=1), np.std(evaluation, axis=1)
    best_round = np.argsort(mean_eval)[0]
    search_value = mean_eval[best_round]

    ###############################################
    # Update Best Score and Return Negative Value
    # In order to minimize error, instead of maximizing accuracy
    ###############################################
    print(' Stopped After {} Epochs... Validation RMSLE: {:.6f} +- {:.6f}'.format(
        best_round, search_value, std_eval[best_round]
    ))

    return -search_value


# The ```search_node``` function is going to be called repeatedly when we start the Bayesian optimization process.
# <br>
# We use ```**kwargs``` as the function's input because the ```BayesianOptimization``` class methods we're going to see below are just going to pass in a dictionary full of hyperparameters that need to be tested.
# <br>
# We end up with ```current_params``` after updating our baseline parameters with the ones we receive and after casting the necessary hyperparameters to integers.
# 
# **THIS IS IMPORTANT! **
# <br>
# The ```BayesianOptimization``` class works by picking some value for each hyperparameter somewhere within its boundaries.
# <br>
# For example, we can get floating point values for ```max_depth```. This is a problem because ```XGBRegressor``` expects integers for certain parameters, so it's gonna be angry and confused if you say you want your decision tree's maximum depth to be 5.3.
# <br>
# For more information, please [read the documentation.](http://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor "Please, read me") It'll tell you the same thing in less colorful language.
# 
# What follows is a fairly standard cross-validation loop, in which we record the RMSE for each fold.
# <br>
# At the end, this function is expected to return the value of that set of tested hyperparameters. 
# <br>
# By default, the Bayesian Optimization functions will attempt to find hyperparameters that maximize this value, but we're dealing with error, which we want to minimize, so we just return the negation of the error.
# 
# # Starting Bayesian Optimization

# In[ ]:


bayes_opt = BayesianOptimization(search_node, BAYESIAN_OPTIMIZATION_BOUNDARIES)
bayes_opt.explore(BAYESIAN_OPTIMIZATION_INITIAL_SEARCH_POINTS)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    bayes_opt.maximize(**BAYESIAN_OPTIMIZATION_MAXIMIZE_PARAMS)


# # Is That It?
# 
# Pretty much, yeah. 
# <br>
# In the first line, we initialize the ```BayesianOptimization``` class by giving it the function to maximize and its search boundaries.
# Then we tell it to ```explore``` the initial search points we selected above. This is just a way to get familiar with the hyperparameters and their "value", so the function has some knowledge before we actually tell it to find the hyperparameters that yield the maximum value.
# 
# In the last line, we call ```maximize```, which actually does two things:
# 1. It randomly samples the target function ```init_points``` times. This is like passing random points to ```explore```, but easier
# 2. It fits the Gaussian Process object to maximize our function for *exactly*  ```n_iter``` many iterations, so choose wisely

# In[ ]:


print('Maximum Value: {}'.format(bayes_opt.res['max']['max_val']))
print('Best Parameters:')
pp(bayes_opt.res['max']['max_params'])
bayes_opt.points_to_csv('bayes_opt_search_points.csv')

best_params = dict(XGBOOST_REGRESSOR_PARAMS, **dict(
    n_estimators=200,
    learning_rate=bayes_opt.res['max']['max_params']['learning_rate'],
    max_depth=int(bayes_opt.res['max']['max_params']['max_depth']),
    gamma=bayes_opt.res['max']['max_params']['gamma'],
    min_child_weight=bayes_opt.res['max']['max_params']['min_child_weight'],
    max_delta_step=int(bayes_opt.res['max']['max_params']['max_delta_step']),
    subsample=bayes_opt.res['max']['max_params']['subsample'],
    colsample_bytree=bayes_opt.res['max']['max_params']['colsample_bytree'],
    scale_pos_weight=bayes_opt.res['max']['max_params']['scale_pos_weight'],
    reg_alpha=bayes_opt.res['max']['max_params']['reg_alpha'],
    reg_lambda=bayes_opt.res['max']['max_params']['reg_lambda']
))


# Above, we're printing the maximum value (minimum error) achieved by Bayesian optimization. Of course, allowing it to run for more iterations will improve this.
# <br>
# Then, we print the hyperparameters that were used to reach the maximum value and save a record of all the points searched.
# <br>
# Finally, we construct a new parameters dictionary using the hyperparameters that yielded the greatest value, so we can...
# 
# # Train a Model With the Best Hyperparameters

# In[ ]:


def RMSLE(target, prediction):
    return metrics.mean_squared_error(target, prediction) ** 0.5

###############################################
# Initialize Folds and Result Placeholders
###############################################
folds = KFold(**CROSS_VALIDATION_PARAMS)
imp_df = np.zeros((len(train_df.columns), CROSS_VALIDATION_PARAMS['n_splits']))
best_evaluation = np.zeros((best_params['n_estimators'], CROSS_VALIDATION_PARAMS['n_splits']))
oof_predictions, test_predictions = np.empty(train_df.shape[0]), np.zeros(test_df.shape[0])
np.random.seed(32)

for fold, (train_index, validation_index) in enumerate(folds.split(target, target)):
    train_input, validation_input = train_df.iloc[train_index], train_df.iloc[validation_index]
    train_target, validation_target = target.iloc[train_index], target.iloc[validation_index]
    
    ###############################################
    # Initialize and Fit Model With Best Parameters
    ###############################################
    model = BASE_ESTIMATOR(**best_params)
    eval_set=[(train_input, train_target), (validation_input, validation_target)]
    model.fit(train_input, train_target, eval_set=eval_set, verbose=False)

    ###############################################
    # Record Feature Importances and Best OOF Round
    ###############################################
    imp_df[:, fold] = model.feature_importances_
    best_evaluation[:, fold] = model.evals_result_["validation_1"]['rmse']
    # best_round = np.argsort(xgb_evaluation[:, fold])[::-1][0]  # FLAG: ORIGINAL
    best_round = np.argsort(best_evaluation[:, fold])[0]  # FLAG: TEST

    ###############################################
    # Make OOF and Test Predictions With Best Round
    ###############################################
    oof_predictions[validation_index] = model.predict(validation_input, ntree_limit=best_round)
    test_predictions += model.predict(test_df, ntree_limit=best_round)

    ###############################################
    # Report Results for Fold
    ###############################################
    oof_rmsle = RMSLE(validation_target, oof_predictions[validation_index])
    print('Fold {}: {:.6f}     Best Score: {:.6f} @ {:4}'.format(
        fold, oof_rmsle, best_evaluation[best_round, fold], best_round
    ))

print('#' * 80 + '\n')
print('OOF RMSLE   {}'.format(RMSLE(target, oof_predictions)))

###############################################
# Compute Mean and Standard Deviation RMSLE
###############################################
test_predictions /= CROSS_VALIDATION_PARAMS['n_splits']
mean_eval, std_eval = np.mean(best_evaluation, axis=1), np.std(best_evaluation, axis=1)
best_round = np.argsort(mean_eval)[0]
print('Best Mean Score: {:.6f} +- {:.6f} @{:4}'.format(
    mean_eval[best_round], std_eval[best_round], best_round
))

importances = sorted(
    [(train_df.columns[i], imp) for i, imp in enumerate(imp_df.mean(axis=1))], 
    key=lambda x: x[1]
)

final_df = pd.DataFrame(
    data=list(zip(test['id'], np.expm1(test_predictions))), columns=['id', 'visitors']
).to_csv('submission_xgb-bayes-opt.csv', index=False, float_format="%.9f")

print('Feature Importances')
pp(importances)


# In[ ]:




