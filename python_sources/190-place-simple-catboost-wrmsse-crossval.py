#!/usr/bin/env python
# coding: utf-8

# # <font color='red'>WARNING!</font> 
# ### <font color='red'>This notebook was tested on the 24 Gb RAM machine with swap. It won't properly work on 16 Gb RAM machine (including Kaggle notebooks).</font>    
# ### <font color='red'>Possibly without swap memory you will need more than 24 Gb of RAM to train this model.</font>

# # Top 4% Solution [190 Place] - Simple Lag Features + CatBoost  
# ### (Bonus - WRMSSE Cross Validation!)

# **Surprisingly simple and straighforward solution**  
# 
# **Key aspects**:  
# 
# 1) Uses **only 500 last dates from the dataset** (including 28 forecasting dates)  
# 
# 2) **Features of the dataset**:  
# 
# - **Lag features on** "sells": lag 28, 35, and 42 (without any special aggregations)  
#     - 'f_value_lag*' columns in the dataset
# 
# 
# - **Rolling mean window features** for every "sells" lag feature: rolling mean with windows 7, 14, 21  
#     - 'f_rolling_mean_\*_by_value_lag\*' columns in the dataset
# 
# 
# - **Label encoded categorical features** - plain "event", "snap*" and product selling characteristics (department id, category id, store id etc.) 
#     - 'f_item_id'
#     - 'f_dept_id'
#     - 'f_cat_id'
#     - 'f_store_id'
#     - 'f_state_id'
#     - 'f_event_name_1'
#     - 'f_event_type_1'
#     - 'f_event_name_2'
#     - 'f_event_type_2'
#     - 'f_snap_CA'
#     - 'f_snap_TX'
#     - 'f_snap_WI'
#     
# 
# - **Sell price** without any modification
# 
# 
# - **Calendar** features:
#     - 'f_year'
#     - 'f_month'
#     - 'f_week'
#     - 'f_day'
#     - 'f_dayofweek'
# 
# 
# 3) **Model**  
# - `Catboost` with `eval_function="RMSE"` and default hyperparameters (only differences - using "Tweedie" as `loss_function`)
# - **Non-recursive prediction method**
# 
# 
# Also in the end of the notebook you will find **WRMSSE cross-validation** code that we have used to validate our model.

# **Special thanks to people whos discussions and notebook were used to make this solution:**  
# - @timetraveller98 for the discussion topic: [Why tweedie works?](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/150614)
# - @sibmike for his [Fast & Clear WRMSSE 18ms](https://www.kaggle.com/sibmike/fast-clear-wrmsse-18ms)  
# 
# **Thank you for your code and insights!**

# In[ ]:


import os
import sys
from datetime import datetime, timedelta
import gc
import joblib
from typing import Iterable, Union, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing, metrics
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, train_test_split, BaseCrossValidator
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from scipy.sparse import csr_matrix
from catboost import Pool, CatBoostRegressor


# # Functions

# #### For constructing features 

# In[ ]:


def add_lag_feature(data: pd.DataFrame, feature_name: str, shift: int, lag: int
                    ) -> Tuple[pd.DataFrame, str]:
    f_name = f'f_{feature_name}_lag{shift + lag}'
    data[f_name] = data.groupby(['f_id'])[feature_name].transform(lambda x: x.shift(shift + lag))
    return data, f_name


def add_rolling_window_mean_feature(
        data: pd.DataFrame, shift: int, lag_feature_name: str, window: int
    ) -> Tuple[pd.DataFrame, str]:

    lag_feature_wo_f = lag_feature_name[2:] if lag_feature_name.startswith('f_') else lag_feature_name
    f_name = f'f_rolling_mean_{window}_by_{lag_feature_wo_f}'

    data[f_name] = data.groupby(['f_id'])[lag_feature_name].transform(
        lambda x: x.shift(shift).rolling(window).mean())
    return data, f_name


def add_date_features(data: pd.DataFrame, date_feature_column_name: str) -> pd.DataFrame:
    data['f_year'] = data[date_feature_column_name].dt.year - data[date_feature_column_name].dt.year.min()
    data['f_month'] = data[date_feature_column_name].dt.month
    data['f_week'] = data[date_feature_column_name].dt.week
    data['f_day'] = data[date_feature_column_name].dt.day
    data['f_dayofweek'] = data[date_feature_column_name].dt.dayofweek
    return data

def load_data(input_dir: str) -> tuple:
    cal = pd.read_csv(f'{input_dir}/calendar.csv')
    stv = pd.read_csv(f'{input_dir}/sales_train_evaluation.csv')
    ss = pd.read_csv(f'{input_dir}/sample_submission.csv')
    sellp = pd.read_csv(f'{input_dir}/sell_prices.csv')

    cal = reduce_mem_usage(cal)
    stv = reduce_mem_usage(stv)
    ss = reduce_mem_usage(ss)
    sellp = reduce_mem_usage(sellp)

    return cal, stv, ss, sellp


# #### General "reduce_mem_usage"
# _(to reduce memory usage of the initial data)_ 

# In[ ]:


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
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
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# # Loading initial data

# In[ ]:


get_ipython().run_cell_magic('time', '', "INPUT_DIR = '/kaggle/input/m5-forecasting-accuracy'\ncal, stv, ss, sellp = load_data(INPUT_DIR)")


# # Joining and denormalizing data 
# _(for every "date" to be represented as a row instead of a column)_

# In[ ]:


d_cols = [c for c in stv.columns if 'd_' in c] # sales data columns
small_cal = cal[['date', 'd', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 
                 'snap_CA', 'snap_TX', 'snap_WI', 'wm_yr_wk']]


# **We've left only 500 history dates from the initial data** (instead of all 1941)

# In[ ]:


# How many history days to leave in dataframe
HISTORY_DAYS_TO_LEAVE = 500


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Leave only train data\ndf = stv[stv[\'id\'].str.endswith(\'_evaluation\')]\n\n# Fill with days to predict\nlast_day = int(df.columns[-1].replace(\'d_\', \'\'))\n\n# Drop days that are earlier than out CUT_DATE date\ncols_to_remove = [f\'d_{i}\' for i in range(1, last_day - HISTORY_DAYS_TO_LEAVE+1)]\ndf = df.drop(cols_to_remove, axis=1)\n\nfor day in range(last_day + 1, last_day + 28 + 1):\n    df[f\'d_{day}\'] = np.nan\n\ndf = df.melt(id_vars=[\'id\', \'item_id\', \'dept_id\', \'cat_id\', \'store_id\', \'state_id\'],\n             var_name="d",\n             value_name="value")\n# Join with calendar\ndf = df.join(small_cal.set_index(\'d\'), how=\'left\', on=\'d\')\n# Join with prices, inner for deleting the days where there was no price for item\ndf = df.merge(sellp, on = [\'store_id\', \'item_id\', \'wm_yr_wk\'], how = \'inner\')')


# Cleaning initial data to save memory

# In[ ]:


del cal, stv, sellp, small_cal
gc.collect()


# # Features construction

# - Give all features prefix 'f_' for easier distinction between initial data and constructed features

# In[ ]:


SHIFT_DAYS = 28


# In[ ]:


LAGS = [0, 7, 14]
WINDOWS = [7, 14, 21]


# ##### Categorical data preprocessing

# In[ ]:


categorical_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                    'snap_CA', 'snap_TX', 'snap_WI',
                    'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

df[categorical_cols] = df[categorical_cols].astype('category')

# Filling NaNs in categorical features
for i in categorical_cols:
    df[i] = df[i].cat.add_categories('unknown')
    df[i] = df[i].fillna('unknown')

# these features will not be preprocessed
df = df.rename(columns={i: f'f_{i}' for i in categorical_cols})


# ##### Add lag features for "sales" data

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nlag_features = []\nfor lag in tqdm(LAGS):\n    df, lag_f = add_lag_feature(df, 'value', SHIFT_DAYS, lag)\n    \n    lag_features.append(lag_f)")


# ##### Add window features for "sales" data

# In[ ]:


for lag_f in tqdm(lag_features):
    for window in WINDOWS:
        df, _ = add_rolling_window_mean_feature(df, 0, lag_f, window)


# ##### Add "date" features

# In[ ]:


df['date'] = pd.to_datetime(df['date'])


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndf = add_date_features(df, 'date')")


# ##### Leave "price" as a feature

# In[ ]:


df = df.rename(columns={'sell_price': 'f_sell_price'})


# # Cleaning the dataset after feature engineering

# ##### Leave only df with features and target

# In[ ]:


feature_cols = [i for i in df.columns if i.startswith('f_')]

df = df[['date', 'value'] + feature_cols]


# In[ ]:


feature_cols


# ##### Call garbage collector 
# ... to free some memory

# In[ ]:


gc.collect()


# ##### Store the resulted dataframe to the filesystem
# This way we can get the resulted dataframe from disk without doing feature engineering again

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# joblib.dump(df, 'df_for_training.joblib', protocol=4)")


# ### Training & submission

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# df = joblib.load('df_for_training.joblib')")


# In[ ]:


feature_cols = [i for i in df.columns if i.startswith('f_')]
feature_cols.remove('f_id')

categorical_cols = list(df.select_dtypes(include=['int8', 'category']).columns)
categorical_cols.remove('f_id')

num_cols = list(set(feature_cols) - set(categorical_cols))

for feature in categorical_cols:
    encoder = preprocessing.LabelEncoder()
    df[feature] = encoder.fit_transform(df[feature])


# In[ ]:


df['value'] = df['value'].astype(np.float32)


# In[ ]:


def train_model(model, train_data, features):
    train_data = data[data['date'] <= '2016-05-22']
    train_data = train_data.dropna()`
    
    model.fit(train_data[features], train_data['value'])

def predict_model(model, data, features):
    test = data[data['date'] > '2016-05-22']
    
    y_pred = model.predict(data[features])
    
    return y_pred

def form_submission(data, submission, filename):
    predictions = data[(data['date'] > '2016-05-22')][['id', 'date', 'value']]
    
    validation_rows = [row for row in submission['id'] if 'validation' in row] 
    validation = submission[submission['id'].isin(validation_rows)]
    
    predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'value').reset_index()
    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
    evaluation = submission[['id']].merge(predictions, on = 'id')
    
    final = pd.concat([validation, evaluation])
    final.to_csv(filename, index = False)


# **Fitting Catboost with most of the default parameters, but:**
# - using `eval_metric='RMSE'` as the most approxiate one to WRMSSE
# - using `loss_function='Tweedie:variance_power=1.5'` because target variable (sales) are appropriate for "Tweedie" distribution  
# _(most of the values of "sales" is around zero, none of them are less than zero and distribution has a large tail of greater than 0 values_  
# _more about Tweedie and why it is appropriate here is in this great discussion topic: [Why tweedie works?](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/150614) )

# In[ ]:


model = CatBoostRegressor(
    eval_metric='RMSE',
    cat_features=categorical_cols,
    verbose=1,
    loss_function='Tweedie:variance_power=1.5',
    # you can tweak "used_ram_limit" param to reduce memory usage when training
    # but the same 190 place result is not guaranteed
    # used_ram_limit='10gb'
)


# In[ ]:


train_model(model, df, feature_cols)


# In[ ]:


# predict and insert oredictions to df
pred = predict_model(model, df, feature_cols)
df['value'] = df['value'].astype(pred.dtype)

df.loc[df['date'] > '2016-05-22', 'value'] = pred


# In[ ]:


# Read in the data
INPUT_DIR = '../m5-forecasting-accuracy'
ss = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')


# In[ ]:


form_submission(df.rename({'f_id': 'id'}, axis=1), ss, 'submission_final_catboost_500days.csv')


# # BONUS
# #### WRMSSE Cross Validation

# It is generally adviced to perform cross validation to validate machine learning models (to avoid overfitting by only analyzing metric for only one subset of training data)
# 
# Most of the notebooks have been focusing on only computing WRMSSE for the last 28 days of the training dataset.
# 
# We've decided to refactor WRMSSE code from the great public notebook [Fast & Clear WRMSSE 18ms](https://www.kaggle.com/sibmike/fast-clear-wrmsse-18ms) into a function that:
# - performs all the nessesary calculations ("W" and "S" matrixes) only based on the input data to this function 
# - works with the "denormalized" data (row with features for every date of every item - the same shape it is need to fit the machine learning models)

# In[ ]:


def wrmsse(x, y_true, y_pred, feature_cols, forecast_horizon=28):
    sales = pd.concat([x.reset_index(drop=True), 
                       y_true.reset_index(drop=True)], axis=1)
    sales = sales.pivot_table(index=['f_id', 
                               'f_state_id', 
                               'f_store_id', 
                               'f_cat_id',
                               'f_dept_id',
                               'f_item_id'], 
                              columns='date', 
                              values='value',
                              aggfunc='first',
                              fill_value=0).reset_index()
    
    sales_pred = pd.concat([x.reset_index(drop=True), 
                            y_pred.reset_index(drop=True)], axis=1)
    sales_pred = sales_pred.pivot_table(index=['f_id', 
                               'f_state_id', 
                               'f_store_id', 
                               'f_cat_id',
                               'f_dept_id',
                               'f_item_id'], 
                              columns='date', 
                              values='value',
                              aggfunc='first',
                              fill_value=0).reset_index()
    
    # List of categories combinations for aggregations as defined in docs:
    dummies_list = [sales.f_state_id.astype(str), 
                    sales.f_store_id.astype(str), 
                    sales.f_cat_id.astype(str), 
                    sales.f_dept_id.astype(str), 
                    sales.f_state_id.astype(str) +'_'+ sales.f_cat_id.astype(str), 
                    sales.f_state_id.astype(str) +'_'+ sales.f_dept_id.astype(str),
                    sales.f_store_id.astype(str) +'_'+ sales.f_cat_id.astype(str), 
                    sales.f_store_id.astype(str) +'_'+ sales.f_dept_id.astype(str), 
                    sales.f_item_id.astype(str), 
                    sales.f_state_id.astype(str) +'_'+ sales.f_item_id.astype(str), 
                    sales.f_id.astype(str)]

    ## First element Level_0 aggregation 'all_sales':
    dummies_df_list =[pd.DataFrame(np.ones(sales.shape[0]).astype(np.int8), 
                                   index=sales.index, 
                                   columns=['all']).T]

    # List of dummy dataframes:
    for i, cats in enumerate(dummies_list):
        dummies_df_list += [pd.get_dummies(cats, 
                                           drop_first=False, 
                                           dtype=np.int8).T]

    # Concat dummy dataframes in one go:
    roll_mat_df = pd.concat(dummies_df_list, 
                            keys=list(range(12)), 
                            names=['level','id'])#.astype(np.int8, copy=False)

    roll_index = roll_mat_df.index
    roll_mat_csr = csr_matrix(roll_mat_df.values)
    
    # Rollup sales:
    d_cols = [i for i in sales.columns if isinstance(i, pd.Timestamp)]
    sales_train_val = roll_mat_csr * sales[d_cols].values

    no_sales = np.cumsum(sales_train_val, axis=1) == 0
    sales_train_val = np.where(no_sales, np.nan, sales_train_val)

    # Denominator of RMSSE / WRMSSE
    S = np.nanmean(np.diff(sales_train_val,axis=1)**2,axis=1)
    
    # Calculate the total sales in USD for each item id:
    df_for_w = x[['f_id', 'date', 'f_sale_usd']]
    
    d_cols.sort()
    cols_for_w = d_cols[-forecast_horizon:]
    
    df_for_w = df_for_w[df_for_w['date'].isin(cols_for_w)]
    
    total_sales_usd = df_for_w.groupby(
        ['f_id'], sort=False)['f_sale_usd'].apply(np.sum).values
    
    # Roll up total sales by ids to higher levels:
    weight2 = roll_mat_csr * total_sales_usd
    
    numerator = 12*weight2
    denominator = np.sum(weight2)
    # Safe divide to replace divide by 0 infinity results with "0" values
    W = np.divide(numerator, 
                  denominator, 
                  out=np.zeros_like(numerator), 
                  where=denominator!=0)
    
    denominator = np.sqrt(S)
    # Safe divide to replace divide by 0 infinity results with "0" values
    SW = np.divide(W, 
                   denominator, 
                   out=np.zeros_like(W), 
                   where=denominator!=0)
    
    return np.nansum(
                np.sqrt(
                    np.mean(
                        np.square(roll_mat_csr*(sales_pred[d_cols[-forecast_horizon:]].values - 
                                                sales[d_cols[-forecast_horizon:]].values))
                            ,axis=1)) * SW)/12


# ### Function that makes time series specific cross validation by WRMSSE metric
# - Takes an unfitted model object, whole training "data" to perform cross validation on, feature columns to use for training and categorical columns (to use "LabelEncoding on")
# - Computes dates that split dataset in TimeSeriesSplit manner but leave 28 validation days and every step reduces training set by 28 days
# - Performs fit-predict on model object
# - Calculating WRMSSE on the predictions and storing them into the list
# - Returning the resulted list with scores

# In[ ]:


def validate_model(
        model: BaseEstimator,
        data: pd.DataFrame,
        feature_col_names: Iterable[str],
        cat_feature_col_names: Iterable[str]) -> np.array:
    data = data.reset_index(drop=True)
    
    # Splitting dataset
    chunks = []
    for i in range(int((data['date'].max() - data['date'].min()).days / 28)):
        test_date_end = data['date'].max() - timedelta(days=i*28)
        train_date_end = data['date'].max() - timedelta(days=(i+1)*28)
        train_date_start = data['date'].min()

        chunks.append((train_date_start, train_date_end, test_date_end))

    for feature in cat_feature_col_names:
        encoder = preprocessing.LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature])

    estimators = [model]

    scores = []
    for train_date_start, train_date_end, test_date_end in chunks:
        gc.collect()
        
        x_train = data[data['date'] <= train_date_end].drop('value', axis=1)
        y_train = data[data['date'] <= train_date_end]['value']

        x_val = data[(data['date'] > train_date_end) & 
                     (data['date'] <= test_date_end)].drop('value', axis=1)
        y_val = data[(data['date'] > train_date_end) & 
                     (data['date'] <= test_date_end)]['value']
        
        print('train_date_start = ' + str(train_date_start))
        print('train_date_end = ' + str(train_date_end))
        print('test_date_end = ' + str(test_date_end))
        
        pipe = make_pipeline(*estimators)
        
        pipe.fit(x_train[feature_col_names], y_train)
        y_pred = pipe.predict(x_val[feature_col_names])

        wmrsse_score = wrmsse(pd.concat([x_train, x_val]), 
                              pd.concat([y_train, y_val]), 
                              pd.concat([y_train, pd.Series(y_pred, name='value')]), 
                              feature_col_names, 
                              forecast_horizon=28)
        
        scores.append(wmrsse_score)
    
    return scores


# In[ ]:


df = joblib.load('df_for_training.joblib')


# In[ ]:


df['f_sale_usd'] = df['value'] * df['f_sell_price']

feature_cols = [i for i in df.columns if i.startswith('f_')]
feature_cols.remove('f_sale_usd')

categorical_cols = list(df.select_dtypes(include=['int8', 'category']).columns)
num_cols = list(set(feature_cols) - set(categorical_cols))

df = df.sort_values(by=categorical_cols)

for feature in categorical_cols:
    encoder = preprocessing.LabelEncoder()
    df[feature] = encoder.fit_transform(df[feature])

df['value'] = df['value'].astype(np.float32)


# In[ ]:


df_to_train = df[df['date'] <= '2016-04-24']
df_to_train = df_to_train.dropna(subset=num_cols)


# In[ ]:


regr_trans = LinearRegression()


# In[ ]:


scores = validate_model(regr_trans, 
                        df_to_train, 
                        feature_cols, 
                        categorical_cols)


# In[ ]:


scores

