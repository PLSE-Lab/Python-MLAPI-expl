#!/usr/bin/env python
# coding: utf-8

# # Objective
# 
# * Try different asymmetric functions and check leaderboard
# 
# * Going to try 3 different versions, no weight, 1.10 weight, 1.15 weight.
# 
# # Comments
# 
# * Good results are being obtained by using rmsse as the cost function and weights in the lgb.Dataset creation.
# * Also train 28 models, where each model is used to predict 1 day. This way we can use recent information (demand lags).
# 
# In the future im going to try this aproaches
# 
# * Another important point is the validation strategy. I dont believe that validation of the last 28 days is a good strategy. Maybee, time series split validation (evaluate last 3 months) or evaluate the same period of the previous year is a good strat (just some ideas, need to check all of them).
# 
# # Results
# 
# * Experiment 1: 
# 
# cv -> 0.58130
# 
# lb -> 0.66572
# 
# * Experiment 2: 
# 
# cv -> 0.61086
# 
# lb -> 0.59939
# 
# Look at that using asymmetric cost function increase our lb but decrease our cv, wired.....something is wrong???
# 
# * Experiment 3:
# 
# cv -> Check the end of the notebook
# 
# lb -> Check the score of the notebook
# 
# I hope this results help you!.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
from typing import Union
from tqdm.auto import tqdm as tqdm
from sklearn import preprocessing
import gc
import lightgbm as lgb
import random

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# consistent results
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# function to read our preprocessed data with some basic features for the experiment
def read_data():
    # read main data
    data = pd.read_pickle('/kaggle/input/m5-small-fe/data_small_fe.pkl')
    # read calendar data
    calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
    # read price data
    sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
    # read submission
    submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
    # get data to evaluate our model
    sales_train_validation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
    train_fold_df = sales_train_validation.iloc[:,:-28]
    valid_fold_df = sales_train_validation.iloc[:,-28:]
    del sales_train_validation
    
    return data, calendar, sell_prices, submission, train_fold_df, valid_fold_df


# In[ ]:


# define lgbm simple model using custom loss and eval metric for early stopping
def run_lgb(data, features, custom_asymmetric_train, custom_asymmetric_valid):
    
    # going to evaluate with the last 28 days
    x_train = data[data['date'] <= '2016-03-27']
    y_train = x_train['demand']
    x_val = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
    y_val = x_val['demand']
    test = data[(data['date'] > '2016-04-24')]
    del data
    gc.collect()

    # define random hyperparammeters
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'n_jobs': -1,
        'seed': 42,
        'learning_rate': 0.1,
        'bagging_fraction': 0.75,
        'bagging_freq': 10, 
        'colsample_bytree': 0.75}

    train_set = lgb.Dataset(x_train[features], y_train)
    val_set = lgb.Dataset(x_val[features], y_val)
    
    del x_train, y_train

    model = lgb.train(params, train_set, num_boost_round = 2500, early_stopping_rounds = 50, 
                      valid_sets = [train_set, val_set], verbose_eval = 100, fobj = custom_asymmetric_train, 
                      feval = custom_asymmetric_valid)
    
    val_pred = model.predict(x_val[features])
    y_pred = model.predict(test[features])
    x_val['demand'] = val_pred
    test['demand'] = y_pred
    x_val = x_val[['id', 'date', 'demand']]
    test = test[['id', 'date', 'demand']]

    return x_val, test


# In[ ]:


# class to evaluate our model
class WRMSSEEvaluator(object):
    
    group_ids = ( 'all_id', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id',
        ['state_id', 'cat_id'],  ['state_id', 'dept_id'], ['store_id', 'cat_id'],
        ['store_id', 'dept_id'], ['item_id', 'state_id'], ['item_id', 'store_id'])

    def __init__(self, 
                 train_df: pd.DataFrame, 
                 valid_df: pd.DataFrame, 
                 calendar: pd.DataFrame, 
                 prices: pd.DataFrame):
        '''
        intialize and calculate weights
        '''
        self.calendar = calendar.copy()
        self.prices = prices.copy()
        self.train_df = train_df.copy()
        self.valid_df = valid_df.copy()
        self.train_target_columns = [i for i in self.train_df.columns if i.startswith('d_')]
        self.weight_columns = self.train_df.iloc[:, -28:].columns.tolist()

        self.train_df['all_id'] = "all"

        self.id_columns = [i for i in self.train_df.columns if not i.startswith('d_')]
        self.valid_target_columns = [i for i in self.valid_df.columns if i.startswith('d_')]

        if not all([c in self.valid_df.columns for c in self.id_columns]):
            self.valid_df = pd.concat([self.train_df[self.id_columns], self.valid_df],
                                      axis=1, 
                                      sort=False)
        self.train_series = self.trans_30490_to_42840(self.train_df, 
                                                      self.train_target_columns, 
                                                      self.group_ids)
        self.valid_series = self.trans_30490_to_42840(self.valid_df, 
                                                      self.valid_target_columns, 
                                                      self.group_ids)
        self.weights = self.get_weight_df()
        self.scale = self.get_scale()
        self.train_series = None
        self.train_df = None
        self.prices = None
        self.calendar = None

    def get_scale(self):
        '''
        scaling factor for each series ignoring starting zeros
        '''
        scales = []
        for i in tqdm(range(len(self.train_series))):
            series = self.train_series.iloc[i].values
            series = series[np.argmax(series!=0):]
            scale = ((series[1:] - series[:-1]) ** 2).mean()
            scales.append(scale)
        return np.array(scales)
    
    def get_name(self, i):
        '''
        convert a str or list of strings to unique string 
        used for naming each of 42840 series
        '''
        if type(i) == str or type(i) == int:
            return str(i)
        else:
            return "--".join(i)
    
    def get_weight_df(self) -> pd.DataFrame:
        """
        returns weights for each of 42840 series in a dataFrame
        """
        day_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()
        weight_df = self.train_df[["item_id", "store_id"] + self.weight_columns].set_index(
            ["item_id", "store_id"]
        )
        weight_df = (
            weight_df.stack().reset_index().rename(columns={"level_2": "d", 0: "value"})
        )
        weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)
        weight_df = weight_df.merge(
            self.prices, how="left", on=["item_id", "store_id", "wm_yr_wk"]
        )
        weight_df["value"] = weight_df["value"] * weight_df["sell_price"]
        weight_df = weight_df.set_index(["item_id", "store_id", "d"]).unstack(level=2)[
            "value"
        ]
        weight_df = weight_df.loc[
            zip(self.train_df.item_id, self.train_df.store_id), :
        ].reset_index(drop=True)
        weight_df = pd.concat(
            [self.train_df[self.id_columns], weight_df], axis=1, sort=False
        )
        weights_map = {}
        for i, group_id in enumerate(tqdm(self.group_ids, leave=False)):
            lv_weight = weight_df.groupby(group_id)[self.weight_columns].sum().sum(axis=1)
            lv_weight = lv_weight / lv_weight.sum()
            for i in range(len(lv_weight)):
                weights_map[self.get_name(lv_weight.index[i])] = np.array(
                    [lv_weight.iloc[i]]
                )
        weights = pd.DataFrame(weights_map).T / len(self.group_ids)

        return weights

    def trans_30490_to_42840(self, df, cols, group_ids, dis=False):
        '''
        transform 30490 sries to all 42840 series
        '''
        series_map = {}
        for i, group_id in enumerate(tqdm(self.group_ids, leave=False, disable=dis)):
            tr = df.groupby(group_id)[cols].sum()
            for i in range(len(tr)):
                series_map[self.get_name(tr.index[i])] = tr.iloc[i].values
        return pd.DataFrame(series_map).T
    
    def get_rmsse(self, valid_preds) -> pd.Series:
        '''
        returns rmsse scores for all 42840 series
        '''
        score = ((self.valid_series - valid_preds) ** 2).mean(axis=1)
        rmsse = (score / self.scale).map(np.sqrt)
        return rmsse

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds],
                                axis=1, 
                                sort=False)
        valid_preds = self.trans_30490_to_42840(valid_preds, 
                                                self.valid_target_columns, 
                                                self.group_ids, 
                                                True)
        self.rmsse = self.get_rmsse(valid_preds)
        self.contributors = pd.concat([self.weights, self.rmsse], 
                                      axis=1, 
                                      sort=False).prod(axis=1)
        return np.sum(self.contributors)


# In[ ]:


# function to evaluate our model results (get WRMSSE)
def evaluate(x_val, train_fold_df, valid_fold_df, calendar, sell_prices):
    x_val = pd.pivot(x_val, index = 'id', columns = 'date', values = 'demand').reset_index()
    x_val.columns = ['id'] + ['d_' + str(i) for i in range(1886, 1914)]
    x_val = train_fold_df[['id']].merge(x_val, on = 'id')
    x_val.drop('id', axis = 1, inplace = True)
    evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, sell_prices)
    score = evaluator.score(x_val)
    return score

def predict(test, submission):
    predictions = test[['id', 'date', 'demand']]
    predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()
    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    evaluation_rows = [row for row in submission['id'] if 'evaluation' in row] 
    evaluation = submission[submission['id'].isin(evaluation_rows)]

    validation = submission[['id']].merge(predictions, on = 'id')
    final = pd.concat([validation, evaluation])
    final.to_csv('submission.csv', index = False)


# In[ ]:


# define cost and eval functions
def custom_asymmetric_train(y_pred, y_true):
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual < 0, -2 * residual, -2 * residual * 1.15)
    hess = np.where(residual < 0, 2, 2 * 1.15)
    return grad, hess

def custom_asymmetric_valid(y_pred, y_true):
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual < 0, (residual ** 2) , (residual ** 2) * 1.15) 
    return "custom_asymmetric_eval", np.mean(loss), False


# In[ ]:


# seed everything
seed_everything(42)
# reading data
data, calendar, sell_prices, submission, train_fold_df, valid_fold_df = read_data()
# get feature columns, also ignoring some features because we dont have enough ram and they overffit
features = [col for col in data.columns if col not in ['id', 'demand', 'part', 'date', 'wm_yr_wk', 'mean_demand_month', 'std_demand_month', 'max_demand_month', 'mean_demand_week', 'std_demand_week', 
                                                       'max_demand_week']]

print(f'We are training with {len(features)} fetures')
data.tail()


# In[ ]:


# run model with asymmetric loss function
x_val, test = run_lgb(data, features, custom_asymmetric_train, custom_asymmetric_valid)
score = evaluate(x_val, train_fold_df, valid_fold_df, calendar, sell_prices)
print(f'Our wrmsse is {score}')
# predict test
predict(test, submission)

