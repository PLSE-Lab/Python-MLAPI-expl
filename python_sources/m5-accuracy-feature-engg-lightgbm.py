#!/usr/bin/env python
# coding: utf-8

# # M5-Accuracy

# ## Inspiration
# 
# * https://www.kaggle.com/ajinomoto132/reduce-mem
# * https://www.kaggle.com/zmnako/lgbm-update-0-85632
# * https://www.kaggle.com/ratan123/m5-forecasting-lightgbm-with-timeseries-splits
# * https://www.kaggle.com/ragnar123/very-fst-model

# ## Import Statements

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 999)

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

INPUT = '/kaggle/input/m5-forecasting-accuracy/'


# ## Helper functions

# In[ ]:


import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

# from: https://www.kaggle.com/ajinomoto132/reduce-mem
def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    """

    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


# In[ ]:


class M5Split2:
    
    def __init__(self, n_splits, group_id, date_col, train_size, gap_size, val_size, step):
        self.n_splits = n_splits + 1
        self.group_id = group_id
        self.date_col = date_col
        self.train_size = train_size
        self.gap_size = gap_size
        self.val_size = val_size
        self.step = step
        
        
    def split(self, df):
        df = df.sort_values(by=[self.date_col])
        indexes = []
        group_indexes = np.array(df.groupby(self.group_id, observed=True).apply(lambda x: x.index))
        for split in range(self.n_splits, 0, -1):
            val_idx = []
            gap_idx = []
            train_idx = []
                
            for idx_arr in group_indexes:
                
                if self.train_size + self.gap_size + self.val_size + self.step*split > len(idx_arr):
                    print(f'Max Split reached')
                    break
                    
                val_idx += list(idx_arr[-(self.val_size + self.step*(split-1)):len(idx_arr) - 1 - self.step*(split-1)])
                gap_idx += list(idx_arr[-(self.gap_size + self.val_size + self.step*(split-1)):-(self.val_size + self.step*(split-1))])
                train_idx += list(idx_arr[-(self.train_size + self.gap_size + self.val_size + self.step*(self.n_splits)):-(self.val_size + self.gap_size + self.step*(split-1))])
                
            yield train_idx, gap_idx, val_idx


# In[ ]:


# https://www.kaggle.com/ragnar123/very-fst-model
def simple_fe_extra(data):
    
    # rolling demand features
    data['lag_t56'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(56))
    data['lag_t57'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(57))
    data['lag_t58'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(58))
    data['rolling_mean_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(56).rolling(7).mean())
    data['rolling_std_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(56).rolling(7).std())
    data['rolling_mean_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(56).rolling(30).mean())
    data['rolling_mean_t90'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(56).rolling(90).mean())
    data['rolling_mean_t180'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(56).rolling(180).mean())
    data['rolling_std_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(56).rolling(30).std())
    data['rolling_skew_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(56).rolling(30).skew())
    data['rolling_kurt_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(56).rolling(30).kurt())
    
    
    # price features
    data['lag_price_t1'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
    data['price_change_t1'] = (data['lag_price_t1'] - data['sell_price']) / (data['lag_price_t1'])
    data['rolling_price_max_t365'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
    data['price_change_t365'] = (data['rolling_price_max_t365'] - data['sell_price']) / (data['rolling_price_max_t365'])
    data['rolling_price_std_t7'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
    data['rolling_price_std_t30'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
    data.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace = True, axis = 1)
    
    # time features
    data['date'] = pd.to_datetime(data['date'])
    data['week'] = data['date'].dt.week
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek
    
    
    return data


# ## Preprocessing

# In[ ]:


# ------------------------- CALENDAR ------------------------ #
df_calendar = pd.read_csv(filepath_or_buffer=f'{INPUT}calendar.csv')
df_calendar = reduce_mem_usage(df_calendar)
print(df_calendar.shape)
df_calendar['date'] = pd.to_datetime(df_calendar['date'])
df_calendar['weekday'] = df_calendar['weekday'].astype(str)
df_calendar['d'] = df_calendar['d'].astype(str)
df_calendar['event_name_1'] = df_calendar['event_name_1'].astype(str)
df_calendar['event_type_1'] = df_calendar['event_type_1'].astype(str)
df_calendar['event_name_2'] = df_calendar['event_name_2'].astype(str)
df_calendar['event_type_2'] = df_calendar['event_type_2'].astype(str)

df_calendar['event_tomorrow_1'] = df_calendar['event_name_1'].shift(-1)
df_calendar['event_tomorrow_2'] = df_calendar['event_name_2'].shift(-1)
df_calendar['event_type_tomorrow_1'] = df_calendar['event_type_1'].shift(-1)
df_calendar['event_type_tomorrow_2'] = df_calendar['event_type_2'].shift(-1)

df_calendar = df_calendar.fillna(value='nan')


# In[ ]:


# event_name_1 and event_name_2 should be fitted together since both are essentailly the same
le1 = LabelEncoder()
le1.fit(pd.concat(objs=[df_calendar['event_name_1'], df_calendar['event_name_2']], axis=0))
df_calendar['event_name_1'] = le1.transform(df_calendar['event_name_1'])
df_calendar['event_name_2'] = le1.transform(df_calendar['event_name_2'])

# event_type_1 and event_type_2 should be fitted together since both are essentailly the same
le2 = LabelEncoder()
le2.fit(pd.concat(objs=[df_calendar['event_type_1'], df_calendar['event_type_2']], axis=0))
df_calendar['event_type_1'] = le2.transform(df_calendar['event_type_1'])
df_calendar['event_type_2'] = le2.transform(df_calendar['event_type_2'])

le3 = LabelEncoder()
le3.fit(pd.concat(objs=[df_calendar['event_tomorrow_1'], df_calendar['event_tomorrow_2']], axis=0))
df_calendar['event_tomorrow_1'] = le3.transform(df_calendar['event_tomorrow_1'])
df_calendar['event_tomorrow_2'] = le3.transform(df_calendar['event_tomorrow_2'])

le4 = LabelEncoder()
le4.fit(pd.concat(objs=[df_calendar['event_type_tomorrow_1'], df_calendar['event_type_tomorrow_2']], axis=0))
df_calendar['event_type_tomorrow_1'] = le4.transform(df_calendar['event_type_tomorrow_1'])
df_calendar['event_type_tomorrow_2'] = le4.transform(df_calendar['event_type_tomorrow_2'])


df_calendar = reduce_mem_usage(df_calendar)


# In[ ]:


df_calendar.tail()


# In[ ]:


# ------------------------- SELLING PRICES ------------------------ #
df_selling_prices = pd.read_csv(f'{INPUT}sell_prices.csv')
print(df_selling_prices.shape)
df_selling_prices['store_id'] = df_selling_prices['store_id'].astype(str)
df_selling_prices['item_id'] = df_selling_prices['item_id'].astype(str)
df_selling_prices['sell_price_cents'] = df_selling_prices['sell_price'] - df_selling_prices['sell_price'].astype(int) 
df_selling_prices['sell_price_perceived'] = df_selling_prices['sell_price'].astype(int)
df_selling_prices = reduce_mem_usage(df_selling_prices)


# In[ ]:


# Generate which columns to be loaded

def generate_cols(factor=0.5):
    cols =['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    start = int(1914*(1-factor))
    days = ['d_' + str(i) for i in range(start, 1914, 1)]
    cols += days
    return cols

columns = generate_cols(factor=0.2)


# In[ ]:


# ------------------------- SALES TRAIN VALIDATION ---------------------- #
df_sales_train_val = pd.read_csv(f'{INPUT}sales_train_validation.csv', usecols=columns)
df_sales_train_val['id'] = df_sales_train_val['id'].astype(str)
df_sales_train_val['item_id'] = df_sales_train_val['item_id'].astype(str)
df_sales_train_val['store_id'] = df_sales_train_val['store_id'].astype(str)
df_sales_train_val['dept_id'] = df_sales_train_val['dept_id'].astype(str)
df_sales_train_val['cat_id'] = df_sales_train_val['cat_id'].astype(str)
df_sales_train_val['state_id'] = df_sales_train_val['state_id'].astype(str)

print(f'Shape before melting: {df_sales_train_val.shape}')
print(f"Unique products before melting: {df_sales_train_val['id'].nunique()}")

df_sales_train_val = pd.melt(df_sales_train_val, id_vars=['id', 'item_id', 'store_id', 'dept_id', 'cat_id', 'state_id'], var_name='d',
                              value_name='demand')

print(f'\nShape after melting: {df_sales_train_val.shape}')
print(f"Unique products after melting: {df_sales_train_val['id'].nunique()}")

# 0 means validation set
df_sales_train_val['set'] = 0
df_sales_train_val['strategy'] = 'train'
df_sales_train_val['id'] = df_sales_train_val['item_id'].str.cat(df_sales_train_val['store_id'], sep='_')


print(f'\nShape after making id from item_id and store_id: {df_sales_train_val.shape}')
print(f"Unique products after id from item_id and store_id: {df_sales_train_val['id'].nunique()}")


# In[ ]:


df_sales_train_val = df_sales_train_val.drop_duplicates()
print(f'Shape after making id from item_id and store_id: {df_sales_train_val.shape}')
print(f"Unique products after id from item_id and store_id: {df_sales_train_val['id'].nunique()}")


# In[ ]:


df_sales_train_val = reduce_mem_usage(df_sales_train_val)


# In[ ]:


# --------------------- ADDING CALENDAR AND SELLING PRICES to SALES TRAIN VAL ---------------------- #
df_sales_train_val = df_sales_train_val.merge(df_calendar, on = ['d'], how='left')
df_sales_train_val = df_sales_train_val.merge(df_selling_prices, on = ["store_id", "item_id", "wm_yr_wk"], how='left')


# In[ ]:


print(f'Shape after merging: {df_sales_train_val.shape}')


# In[ ]:


df_sales_train_val = df_sales_train_val.reset_index(drop=True)


# In[ ]:


le = []
for num, col in enumerate(['id', 'item_id', 'store_id', 'dept_id', 'cat_id', 'state_id']):
    le_id = LabelEncoder()
    le.append(le_id)
    le[num].fit(df_sales_train_val[col])
    df_sales_train_val[f'{col}_encoded'] = le[num].transform(df_sales_train_val[col])


# In[ ]:


# -------------------------- SAMPLE SUBMISSION -------------------------- #
df_sample_submission = pd.read_csv(f'{INPUT}sample_submission.csv')
df_sample_submission = reduce_mem_usage(df_sample_submission)
df_sample_submission['id'] = df_sample_submission['id'].astype(str)


# In[ ]:


df_sample_submission['state_id'] = df_sample_submission['id'].str[-15:-13]
df_sample_submission['store_id'] = df_sample_submission['id'].str[-15:-11]
df_sample_submission['item_id'] = df_sample_submission['id'].str[0:-16]
df_sample_submission['dept_id'] = df_sample_submission['id'].str[0:-20]
df_sample_submission['cat_id'] = df_sample_submission['id'].str[0:-22]
df_sample_submission.tail()


# In[ ]:


# -------------------------- SALES TEST -------------------------- #
df_sales_test = pd.melt(df_sample_submission, id_vars=['id', 'item_id', 'store_id', 'dept_id', 'cat_id', 'state_id'], var_name='d', value_name='demand')

df_sales_test['d2'] = df_sales_test['d'].str[1:]
df_sales_test['d2'] = df_sales_test['d2'].astype(int)
df_sales_test.loc[df_sales_test['id'].str[-10:]=='evaluation', 'd2'] = df_sales_test.loc[df_sales_test['id'].str[-10:]=='evaluation', 'd2'] + 28
df_sales_test['d2'] = df_sales_test['d2'] + 1913
df_sales_test['d2'] = df_sales_test['d2'].astype(str)
df_sales_test['d2'] = 'd_' + df_sales_test['d2']

# 1 indicates test set
df_sales_test['set'] = 1
df_sales_test.loc[df_sales_test['id'].str[-10:]=='validation', 'strategy'] = 'validation'
df_sales_test.loc[df_sales_test['id'].str[-10:]=='evaluation', 'strategy'] = 'evaluation'
df_sales_test['original_id'] = df_sales_test['id']
df_sales_test['id'] = df_sales_test['id'].str[0:-11]

df_sales_test = df_sales_test.drop(columns=['d'])
df_sales_test = df_sales_test.rename(columns={'d2': 'd'})


# In[ ]:


# # --------------------- ADDING CALENDAR AND SELLING PRICES to SALES TRAIN VAL ---------------------- #
df_sales_test = df_sales_test.merge(df_calendar, on = ['d'], how='left')
df_sales_test = df_sales_test.merge(df_selling_prices, on = ["store_id", "item_id", "wm_yr_wk"], how='left')


# In[ ]:


# ---------------------- Making {id}_encoded features ------------------------ #
for num, col in enumerate(['id', 'item_id', 'store_id', 'dept_id', 'cat_id', 'state_id']):
    df_sales_test[f'{col}_encoded'] = le[num].transform(df_sales_test[col])


# In[ ]:


# -------------------------- TRAIN VAL TEST -------------------------- #
df_sales_test = df_sales_test.drop(columns=['original_id'])
df_train_val_test = pd.concat(objs=[df_sales_train_val, df_sales_test], axis=0, ignore_index=True)
df_train_val_test = reduce_mem_usage(df_train_val_test)


# In[ ]:


df_train_val_test['d_number'] = df_train_val_test['d'].str[2:].astype(int)


# In[ ]:


df_train_val_test = df_train_val_test.sort_values(by=['id', 'd_number'])


# In[ ]:


df_train_val_test = df_train_val_test.reset_index(drop=True)


# In[ ]:


df_train_val_test = simple_fe_extra(df_train_val_test)


# In[ ]:


for col in df_train_val_test.columns:
    if is_numeric_dtype(df_train_val_test[col]):
        df_train_val_test[col] = df_train_val_test[col].round(decimals=3)


# In[ ]:


df_train_val_test['is_weekend'] = 0
df_train_val_test.loc[df_train_val_test['weekday'].isin(['Saturday', 'Sunday']), 'is_weekend'] = 1 


# In[ ]:


# Making a feature 'prob_high_demand' which is a float which is max on day, day before and day after the event_name_1 
# and which decreases as we move away from event date on both in past and future. Think of it as a bell curve over date
# peaking at event date

indexes = df_train_val_test.groupby(by=['event_name_1']).groups
del indexes[30]                # 30 representes 'nan' event
high_demand_indexes = []
mid_demand_indexes = []
mid_low_demand_indexes = []
for key, val in indexes.items():
    for value in indexes[key]:
        high_demand_indexes.append(value)
        high_demand_indexes.append(value + 1)
        mid_demand_indexes.append(value + 2)
        mid_low_demand_indexes.append(value + 3)
        high_demand_indexes.append(value - 1)
        mid_demand_indexes.append(value - 2)
        mid_low_demand_indexes.append(value - 3)
        
high_demand_indexes = list(set(high_demand_indexes))
mid_demand_indexes = list(set(mid_demand_indexes))
mid_low_demand_indexes = list(set(mid_low_demand_indexes))

df_train_val_test['prob_high_demand'] = 0
high_demand_indexes = list(set(df_train_val_test.index).intersection(set(high_demand_indexes)))
mid_demand_indexes = list(set(df_train_val_test.index).intersection(set(mid_demand_indexes)))
mid_low_demand_indexes = list(set(df_train_val_test.index).intersection(set(mid_low_demand_indexes)))

df_train_val_test.loc[mid_low_demand_indexes, 'prob_high_demand'] = 0.3
df_train_val_test.loc[mid_demand_indexes, 'prob_high_demand'] = 0.6
df_train_val_test.loc[high_demand_indexes, 'prob_high_demand'] = 0.9


# In[ ]:


df_train_val_test['date'] = pd.to_datetime(df_train_val_test['date'])


# In[ ]:


df_train_val_test['is_weekend'] = df_train_val_test['is_weekend'].astype('category')


# ## Modelling

# ## Cross validation

# In[ ]:


import lightgbm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error as mse
import time


# In[ ]:


cols = list(df_train_val_test.columns)


# In[ ]:


cols


# In[ ]:


# X_train = df_train_val_test.loc[df_train_val_test['set']==0, cols]
# y_train = X_train['demand']
# X_train = X_train.drop(columns=['d_number', 'set', 'demand', 'strategy', 'week', 'day', 'dayofweek',
#                                 'wday','month']) # drop 'set' bcoz it can cause data leak 


# In[ ]:


# scores = []
# m5split2 = m5s.M5Split2(n_splits=0, group_id='id', date_col='date', train_size=365, gap_size=0, val_size=28, step=28)
# fold = 0
# estimators = []
# for trn_idx, _, val_idx in m5split2.split(X_train):
#     fold_s_time = time.time()
#     s_time = time.time()
#     X_trn = X_train.loc[trn_idx, :]
#     y_trn = y_train[trn_idx]
#     X_val = X_train.loc[val_idx, :]
#     y_val = y_train[val_idx]
    
#     e_time = time.time()
#     print(f'Time taken for splitting of fold{fold}: {e_time-s_time} seconds')

#     s_time = time.time()
#     lgbmr = lightgbm.LGBMRegressor(objective = "regression",
#                                    boosting_type = 'gbdt',
#                                    n_estimators = 1000,
#                                    random_state = 20,
#                                    learning_rate = 0.1,
#                                    bagging_fraction = 0.75,
#                                    bagging_freq = 10, 
#                                    colsample_bytree = 0.75)
#     lgbmr.fit(X_trn.drop(columns=['date', 'd']), y_trn)
#     estimators.append(lgbmr)
#     y_predict = np.around(lgbmr.predict(X_val.drop(columns=['date', 'd'])))
#     e_time = time.time()
#     print(f'Time taken for learning of fold{fold}: {e_time-s_time} seconds')
    
#     s_time = time.time()
#     rmse_score = np.sqrt(mse(y_val, y_predict))
#     print(f'CV RMSE score of fold{fold} (train_len: {len(trn_idx)}, val_len: {len(val_idx)}): {rmse_score}')
#     scores.append(rmse_score)
#     fold_e_time = time.time()
#     print(f'Total time taken for fold{fold}: {fold_e_time-fold_s_time} seconds\n')
#     fold += 1

# print(f'Mean score: {sum(scores)/len(scores)}, Stdev: {np.std(scores)}')


# In[ ]:


# # Since the splitter above will not train model on last 28 days I manually trained on data which includes the last 28 days as well

# X_trn = X_train.loc[X_train['date']>='2015-04-24 00:00:00']
# y_trn = y_train[X_trn.index]
# lgbmr = lightgbm.LGBMRegressor(objective = "regression",
#                                    boosting_type = 'gbdt',
#                                    n_estimators = 1000,
#                                    random_state = 20,
#                                    learning_rate = 0.1,
#                                    bagging_fraction = 0.75,
#                                    bagging_freq = 10, 
#                                    colsample_bytree = 0.75)
# lgbmr.fit(X_trn.drop(columns=['date', 'd']), y_trn)
# estimators.append(lgbmr)


# ## Ensembling and Prediction

# In[ ]:


# X_test = df_train_val_test.loc[df_train_val_test['set']==1, cols]
# X_test = X_test.drop(columns=['d_number', 'set', 'demand', 'strategy', 'week', 'day', 'dayofweek', 
#                               'wday', 'month'])
# predictions = pd.DataFrame(columns=[estimator.__class__.__name__ + str(estimators.index(estimator)) for estimator in estimators])

# for i, estimator in enumerate(estimators):
#     y_predict = estimator.predict(X_test.drop(columns=['date', 'd']))
#     predictions[predictions.columns[i]] = pd.Series(y_predict)

# cols = list(predictions.columns)
# predictions['final'] = np.mean(predictions.loc[:, cols], axis=1)


# In[ ]:


# predictions.max()


# In[ ]:


# predictions.min()


# In[ ]:


# prediction_col = 'LGBMRegressor1'
# predictions.loc[predictions[prediction_col] < 0.0, prediction_col] = 0.0 

# predictions = predictions.set_index(X_test.index)

# X_test['pred'] = predictions['LGBMRegressor1']


# df_train_val_test.loc[X_test.index, 'demand'] = X_test['pred']
# df_submit = df_train_val_test.loc[X_test.index, :]
# df_submit['original_id'] = df_submit['id'].str.cat(df_submit['strategy'], sep='_')

# df_submit.to_csv(f'{INPUT}submit.csv', index=False)

# df_submit.head()


# In[ ]:


# df_submit = pd.read_csv(f'{INPUT}submit.csv')
# df_submit = reduce_mem_usage(df_submit)


# In[ ]:


# df_submit['d2'] = df_submit['d'].str[2:]
# df_submit['d2'] = df_submit['d2'].astype(int)
# df_submit.loc[df_submit['strategy']=='validation', 'd2'] = df_submit.loc[df_submit['strategy']=='validation', 'd2'] - 1913
# df_submit.loc[df_submit['strategy']=='evaluation', 'd2'] = df_submit.loc[df_submit['strategy']=='evaluation', 'd2'] - 1941
# df_submit['d2'] = df_submit['d2'].astype(str)
# df_submit['d2'] = 'F' + df_submit['d2']
# df_submit.loc[df_submit['strategy']=='evaluation', 'demand'] = 0.0
# df_submission = df_submit.loc[:, ['original_id', 'demand', 'd2']].pivot(index='original_id', columns='d2', values='demand')


# In[ ]:


# cols_order = ['F' + str(i) for i in range(1,29)]
# df_submission = df_submission[cols_order]

# df_sample_submission = pd.read_csv(f'{INPUT}sample_submission.csv')
# df_sample_submission = df_sample_submission.set_index(keys='id')
# df_submission = df_submission.reindex(df_sample_submission.index)
# df_submission = df_submission.reset_index()
# df_submission = df_submission.rename(columns={'original_id': 'id'})
# df_submission.to_csv(f'{INPUT}submission.csv', index=False)
# df_submission


# In[ ]:




