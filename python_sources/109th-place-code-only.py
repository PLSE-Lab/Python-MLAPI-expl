#!/usr/bin/env python
# coding: utf-8

# This is a slightly altered version of the solution

# In[ ]:


import pandas as pd
import numpy as np
import random


# In[ ]:


SEED = 1
np.random.seed(SEED)
random.seed(SEED)


# In[ ]:


calendar = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')


def set_calendar_data_types(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    print('Setting calendat data types...')
    
    df['events_names'] = df[['event_name_1', 'event_name_2']].fillna('').sum(axis=1).replace('', 'None')
    df['events_types'] = df[['event_type_1', 'event_type_2']].fillna('').sum(axis=1).replace('', 'None')
    df['next_events_names'] = df[['event_name_1', 'event_name_2']].fillna('').sum(axis=1).replace('', np.nan).fillna(method='bfill')
    df['next_events_types'] = df[['event_type_1', 'event_type_2']].fillna('').sum(axis=1).replace('', np.nan).fillna(method='bfill')
    df['event_name_1'] = pd.Categorical(df['event_name_1'].fillna('None'))
    df['event_type_1'] = pd.Categorical(df['event_type_1'].fillna('None'))
    df['event_name_2'] = pd.Categorical(df['event_name_2'].fillna('None'))
    df['event_type_2'] = pd.Categorical(df['event_type_2'].fillna('None'))
    df['d'] = pd.Categorical(df['d'])
    df['wm_yr_wk'] = pd.Categorical(df['wm_yr_wk'],
                                    categories=df.wm_yr_wk.unique(), 
                                    ordered=True)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['wday'] = df['wday'].astype(np.int8)
    df['snap_CA'] = df['snap_CA'].astype(bool)
    df['snap_TX'] = df['snap_TX'].astype(bool)
    df['snap_WI'] = df['snap_WI'].astype(bool)
    df.drop(['year', 'weekday', 'month', 'wday'], axis=1, inplace=True)
    
    print('Done!')
    return df

def get_calendar_time_features(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    print('Get simple datetime features...')
    
    df['day_of_month'] = df['date'].dt.day.astype(np.int8)
    df['day_of_week'] = df['date'].dt.dayofweek.astype(np.int8)
    
    print('Done!')
    return df

def get_calendar_event_features(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    print('Getting event features...')
    
    events_all_dates = df[(df['event_name_1'] != 'None') | (df['event_name_2'] != 'None')]
    prev = 0
    
    for event in events_all_dates.iterrows():
        df.loc[prev:event[0], 'days_to_event'] = (df.loc[event[0], 'date'] - df.loc[prev:event[0], 'date']).dt.days
        prev = event[0]
    
    df['days_to_event'] = df['days_to_event'].replace(0, 25).astype(np.int8)
    
    events_names_dict = df['events_names'].value_counts().rank(method='first').to_dict()
    events_types_dict = df['events_types'].value_counts().rank(method='first').to_dict()
    df['events_names'] = df['events_names'].map(events_names_dict).astype(np.int8)
    df['events_types'] = df['events_types'].map(events_types_dict).astype(np.int8)
    df['next_events_names'] = df['next_events_names'].map(events_names_dict).astype(np.int8)
    df['next_events_types'] = df['next_events_types'].map(events_types_dict).astype(np.int8)
    
    df.drop(['event_name_1', 'event_name_2', 'event_type_1', 'event_type_2'], axis=1, inplace=True)
    
    print('Done!')
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ncalendar = (calendar\n            .pipe(set_calendar_data_types)\n            .pipe(get_calendar_time_features)\n            .pipe(get_calendar_event_features))')


# In[ ]:


sell_prices = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')

def set_sell_prices_cat_data_types(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    print('Setting data types...')
    
    df['store_id'] = pd.Categorical(df['store_id'])
    df['item_id'] = pd.Categorical(df['item_id'])
    df['wm_yr_wk'] = pd.Categorical(df['wm_yr_wk'],
                                    categories=df.wm_yr_wk.unique(), 
                                    ordered=True)
    
    print('Done!')
    return df

def get_sell_prices_features(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    print('Getting sell prices features...')
    
    df['dept_id'] = pd.Categorical(df['item_id'].str.slice(stop=-4))
    df['cat_id'] = pd.Categorical(df['item_id'].str.slice(stop=-6))
    df['state_id'] = pd.Categorical(df['store_id'].str.slice(stop=-2))
    df['sell_price'] = df['sell_price'].replace((np.inf, -np.inf, np.nan), 0).astype(np.float16)
    
    print('Done!')
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nsell_prices = (sell_prices\n               .pipe(set_sell_prices_cat_data_types)\n               .pipe(get_sell_prices_features))')


# In[ ]:


import category_encoders
from scipy import stats


# In[ ]:


VAL_CUTOFF = '2015-05-01'
TRAIN_CUTOFF = '2012-01-01'


# In[ ]:


import gc


# In[ ]:


sales = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_evaluation.csv')
cpi = pd.read_csv('../input/cpi-m5/cpi.csv')


def merge_df(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    print('Merging dataframes together...')
    
    df = df.merge(calendar, on=['wm_yr_wk'], how='left')
    
    df.drop(['wm_yr_wk'], axis=1, inplace=True)
    
    df = df.merge(pd.melt(sales, 
                          id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                          var_name='d', 
                          value_name='demand')[['item_id', 'store_id', 'd', 'demand']], 
                  on=['item_id', 'store_id', 'd'],
                  how='left')
    
    cpi['CPI_2m_ago'] = cpi.groupby('state_id')['cpi'].transform(lambda x: x.shift(2)).astype(np.float16)
    cpi.drop(['cpi', 'gasoline_price', 'employees', 'population', 'us_gasoline_price'], axis=1, inplace=True)
    df['month'] = df['date'].dt.month.astype(np.int8)
    df['year'] = df['date'].dt.year.astype(np.int16)
    df = df.merge(cpi, on=['state_id', 'month', 'year'], how='left')
    df.drop(['year', 'month'], axis=1, inplace=True)
    
    gc.collect()
    print('Done!')
    return df


def set_df_data_types(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    print('Setting data types...')
    
    df['demand'] = df['demand'].astype(np.float16)
    df['store_id'] = pd.Categorical(df['store_id'])
    df['item_id'] = pd.Categorical(df['item_id'])
    df['id'] = pd.Categorical(df['item_id'].str.cat(df['store_id'], sep='_'))
    df['d'] = df['d'].str.slice(start=2).astype(np.int16)
    df['CPI_2m_ago'] = df['CPI_2m_ago'].astype(np.float16)
    
    for state in df.state_id.unique():
        
        mask = df['state_id'] == state
        df.loc[mask, 'is_snap_avaliable'] = df.loc[mask, f'snap_{state}']
    
    df['is_snap_avaliable'] = df['is_snap_avaliable'].astype(bool)
    
    df.drop(['snap_TX', 'snap_CA', 'snap_WI'], axis=1, inplace=True)
    
    gc.collect()
    print('Done!')
    return df


def get_demand_feature(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    print('Getting demand lag values...')
    
    SHIFT = 29
    
    df[f'cumulative_mean_demand_{SHIFT}d_ago'] = (df.groupby(['store_id', 'item_id'])['demand']
                                                  .transform(lambda x: x.shift(SHIFT).expanding().mean())
                                                  .replace((np.inf, -np.inf, np.nan), 0)
                                                  .astype(np.float16))
    
    df[f'cumulative_md_low_demand_{SHIFT}d_ago'] = (df.groupby(['store_id', 'item_id'])['demand']
                                                    .transform(lambda x: x.shift(SHIFT).expanding().median().apply(np.floor))
                                                    .replace((np.inf, -np.inf, np.nan), 0)
                                                    .astype(np.int16))
    
    for i in [1, 4, 8]:
        
        df[f'ewm_mean_{i}w_demand_{SHIFT}d_ago'] = (df.groupby(['store_id', 'item_id'])['demand']
                                                    .transform(lambda x: x.shift(SHIFT).ewm(span=7*i).mean())
                                                    .replace((np.inf, -np.inf, np.nan), 0)
                                                    .astype(np.float16))
        
        df[f'rolling_mean_{i}w_demand_{SHIFT}d_ago'] = (df.groupby(['store_id', 'item_id'])['demand']
                                                        .transform(lambda x: x.shift(SHIFT).rolling(7*i).mean())
                                                        .replace((np.inf, -np.inf, np.nan), 0)
                                                        .astype(np.float16))
        
    gc.collect()
    print('Done!')
    return df


def encode_categorical_features(df: pd.core.frame.DataFrame, val_cutoff: str = VAL_CUTOFF) -> pd.core.frame.DataFrame:
    print('Encoding categorical features...')
    
    for feat in ['store_id', 'item_id', 'dept_id']:
        
        encoder = category_encoders.CatBoostEncoder(handle_unknown='value', handle_missing='value', a=1e-9)
        encoder.fit(df.loc[df.date < val_cutoff, feat], df.loc[df.date < val_cutoff, 'demand'])
        df[feat] = encoder.transform(df[feat]).astype(np.float16)
        del encoder
        
        if (feat == 'store_id') or (feat == 'dept_id'):
            
            temp_dict = {k: v for k, v in zip(df[feat].unique(), stats.rankdata(df[feat].unique()))}
            df[feat] = df[feat].map(temp_dict).astype(np.int8)
            del temp_dict
    
    df['expected_item_revenue'] = (df['item_id'] * df['sell_price']).replace((np.inf, -np.inf, np.nan), 0).astype(np.float16)
    
    df['is_in_CA'] = df['state_id'] == 'CA'
    df['is_in_TX'] = df['state_id'] == 'TX'
    df['is_in_WI'] = df['state_id'] == 'WI'
    
    df.drop(['state_id'], axis=1, inplace=True)
    
    df['is_foods'] = df['cat_id'] == 'FOODS'
    df['is_household'] = df['cat_id'] == 'HOUSEHOLD'
    df['is_hobbies'] = df['cat_id'] == 'HOBBIES'
    
    df.drop(['cat_id'], axis=1, inplace=True)
    
    gc.collect()
    print('Done!')
    return df


def cut_df(df: pd.core.frame.DataFrame, train_cutoff: str = TRAIN_CUTOFF) -> pd.core.frame.DataFrame:
    print('Cutting off dataframe...')
    
    df = df.loc[df.date >= train_cutoff]
    
    gc.collect()
    print('Done')
    return df.reset_index(drop=True)


def get_dev_val_test(df: pd.core.frame.DataFrame) -> list:
    print('Splitting dataset into Train-Validation-Test...')
    
    X_train = df[(df.date < VAL_CUTOFF)].drop(['demand', 'date', 'id', 'd'], axis=1)
    X_valid = df[(df['d'] < 1942) & (df.date >= VAL_CUTOFF)].drop(['demand', 'date', 'id', 'd'], axis=1)
    X_test = df[df['d'] >= 1914].drop(['demand', 'd'], axis=1)
    y_train = df[(df.date < VAL_CUTOFF)]['demand'].astype(np.int16)
    y_valid = df[(df['d'] < 1942) & (df.date >= VAL_CUTOFF)]['demand'].astype(np.int16)
    
    gc.collect()
    print('Done!')
    return [X_train, X_valid, X_test, y_train, y_valid]


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nX_train, X_valid, X_test, y_train, y_valid = (sell_prices\n                                              .pipe(merge_df)\n                                              .pipe(set_df_data_types)\n                                              .pipe(get_demand_feature)\n                                              .pipe(encode_categorical_features)\n                                              .pipe(cut_df)\n                                              .pipe(get_dev_val_test))')


# In[ ]:


del sell_prices, calendar, sales, cpi
gc.collect()


# In[ ]:


from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer


# In[ ]:


cols = X_train.columns[(X_train.astype(np.float16).corrwith(y_train).abs() > 0.01)]

non_bool_cols = X_train[cols].select_dtypes(exclude='bool').columns


# In[ ]:


col_tr = ColumnTransformer([('scale', RobustScaler(quantile_range=(5, 95)), non_bool_cols)], remainder='passthrough')


# In[ ]:


lr = PassiveAggressiveRegressor(C=1e-5, 
                                fit_intercept=True, 
                                max_iter=1000, 
                                tol=1e-3, 
                                shuffle=True, 
                                verbose=1,
                                loss='squared_epsilon_insensitive', 
                                epsilon=1e-5,
                                random_state=SEED, 
                                warm_start=True, 
                                average=True)

lr.fit(col_tr.fit_transform(X_train[cols]), y_train);
lr.fit(col_tr.transform(X_valid[cols]), y_valid);


# In[ ]:


import catboost as cb


# In[ ]:


dev_pool = cb.Pool(X_train, 
                   y_train.astype(int))
val_pool = cb.Pool(X_valid, 
                   y_valid.astype(int))

tr_val_ratio = X_valid.shape[0] / X_train.shape[0]


# In[ ]:


del X_train, X_valid, y_train, y_valid
gc.collect()


# In[ ]:


gbt_regressor = cb.CatBoostRegressor(iterations=5000,
                                     learning_rate=0.1,
                                     depth=7,
                                     l2_leaf_reg=3.5,
                                     loss_function='RMSE',
                                     boosting_type='Plain',
                                     eval_metric='RMSE',
                                     feature_border_type='UniformAndQuantiles',
                                     thread_count=4,
                                     random_seed=SEED,
                                     has_time=True,
                                     random_strength=1,
                                     bootstrap_type='MVS',
                                     subsample=0.8,
                                     max_bin=254,
                                     score_function='L2',
                                     model_shrink_rate=1e-5,
                                     boost_from_average=False,
                                     sampling_frequency='PerTreeLevel',
                                     fold_permutation_block=1,
                                     leaf_estimation_method='Newton',
                                     leaf_estimation_iterations=1,
                                     fold_len_multiplier=2,
                                     model_shrink_mode='Constant',
                                     task_type='CPU',
                                     langevin=True,
                                     diffusion_temperature=1e2,
                                     used_ram_limit='15gb',
                                     model_size_reg=0,
                                     allow_writing_files=False)


# In[ ]:


gbt_regressor.fit(dev_pool, eval_set=val_pool, early_stopping_rounds=300, verbose=100);


# In[ ]:


from matplotlib import pyplot as plt
plt.figure(figsize=(24,10));
plt.barh(gbt_regressor.feature_names_, gbt_regressor.feature_importances_);


# In[ ]:


del dev_pool
gc.collect()


# In[ ]:


cbr = cb.CatBoostRegressor(iterations=int(gbt_regressor.get_best_iteration() * tr_val_ratio),
                           learning_rate=0.1,
                           depth=7,
                           l2_leaf_reg=3.5,
                           loss_function='RMSE',
                           boosting_type='Plain',
                           eval_metric='RMSE',
                           feature_border_type='UniformAndQuantiles',
                           thread_count=4,
                           random_seed=SEED,
                           has_time=True,
                           random_strength=1,
                           bootstrap_type='MVS',
                           subsample=0.8,
                           max_bin=254,
                           score_function='L2',
                           model_shrink_rate=0,
                           boost_from_average=False,
                           sampling_frequency='PerTreeLevel',
                           fold_permutation_block=1,
                           leaf_estimation_method='Newton',
                           leaf_estimation_iterations=1,
                           fold_len_multiplier=2,
                           model_shrink_mode='Constant',
                           task_type='CPU',
                           langevin=True,
                           diffusion_temperature=1e2,
                           used_ram_limit='15gb',
                           model_size_reg=0,
                           allow_writing_files=False)


# In[ ]:


cbr.fit(val_pool, verbose=10, init_model=gbt_regressor);


# In[ ]:


del val_pool, gbt_regressor
gc.collect()


# In[ ]:


preds_gbt = np.maximum(cbr.predict(X_test.drop(['date', 'id'], axis=1)), 0)
preds_lr = np.maximum(lr.predict(col_tr.transform(X_test[cols])), 0)
X_test['demand'] = 0.9 * preds_gbt + 0.1 * preds_lr


# In[ ]:


del cbr, lr
gc.collect()


# In[ ]:


submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')


# In[ ]:


predictions = X_test.pivot_table(index='id', columns='date', values='demand').reset_index()


# In[ ]:


del X_test
gc.collect()


# In[ ]:


def get_sub(predictions, submission):
    
    new_cols = [f'F{i}' for i in range(1, 29)]
    validation = predictions.copy()
    validation = validation[validation.columns[1:29]]
    validation = validation.rename({k: v for k, v in zip(validation.columns, new_cols)}, axis=1)
    validation['id'] = predictions['id'].apply(lambda x: x + '_validation')
    
    evaluation = predictions.copy()
    evaluation = evaluation[evaluation.columns[29:]]
    evaluation = evaluation.rename({k: v for k, v in zip(evaluation.columns, new_cols)}, axis=1)
    evaluation['id'] = predictions['id'].apply(lambda x: x + '_evaluation')
    
    sub_1 = pd.merge(submission.loc[:30489, 'id'], validation, how='left')
    sub_2 = pd.merge(submission.loc[30490:, 'id'], evaluation, how='left')
    
    return pd.concat([sub_1, sub_2], axis=0)


# In[ ]:


submission = get_sub(predictions, submission)


# In[ ]:


submission.to_csv('submission.csv', index=False)

