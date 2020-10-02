#!/usr/bin/env python
# coding: utf-8

# # Gradient Boosting Notebook
# 
# Fork from a version using CatBoost. 
# 
# We switch to LightGBM because CatBoost is very RAM-intensive. 
# 
# Used Kaggle sources:
# 
# - [M5 First Public Notebook Under 0.50](https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50#Changes)
# - [m5_catboost (inspired on previous notebook)](https://www.kaggle.com/vgarshin/m5-catboost)
#     * This notebook is *heavily* inspired by this notebook. 
#     * Where sections are almost literally copied, I indicated this in the respective cells as well

# # Submission log
# 
# For reference, the best score with the CatBoost notebook this is a fork from was [0.67] on CPU, with feature engineering (lag + rolling mean + dates); 1000 iterations; depth 8; max_bin 128, max_ctr_complexity 1
# 
# Increase MODEL_VERSION below
# 
# Most recent at the bottom:
# 
# - v0: 0.57666
# - v1: 0.58
#     * Added month and year date feature
#     * Iterations 1000 > 1200
# - v2: CRASH
#     * Remove month and year again
#     * Iterations still 1200
#     * Year more training days
# - v3: 0.55185
#     * BEGIN_DATE '2014-8-01' 
# - v4: 0.76206
#     * Add a zero threshold of 0.01 for very low predict sales
#     * Setting predictions to 0 already during recursive predictions may too strongly influence predictions for the coming days. I could instead only apply the threshold after making all recursive predictions.
# - v5: 0.75
#     * Try a lower threshold of 0.001 
#     * What is strange is that I can't spot anything being set to 0, so this threshold is clearly too low.
#     
# v4 and v5 accidentally used only a few days for training... I changed this for speed at one point and forgot to set it back. Back to the drawing table.
# 
# - v6: 0.57
#     * Fix training data issue of v4 and v5
#     * Added feature importance
#     * Zero threshold at 0.01, but now only apply after all predictions are done
#     * So does not help, unless the optimized prediction loop has some detrimental effect. But it shouldn't.
# - v7: 0.6
#     * Remove rolling_window features on id, I don't think it makes sense
#     * Clean up load_data, remove unused label_encoding
#     * There was double code for dropping everything before BEGIN_DATE that nevertheless increased memory usage. Fixed.
#     * Fix: all event_names were NaN. I forgot to set them to int16 in load_data(). Fixed.
#     * Now we can safely drop NaNs! 
#     * As a result of dropping NaNs, I'll experiment with adding more data. Try with BEGIN_DATE = '2014-01-01'
# - v7: 0.58478
#     * BEGIN_DATE '2013-08-01' (day 915)
# - v8: CRASH
#     * Comment out threshold code; clearly does not help.
#     * BEGIN_DATE: '2012-01-01' (day 337)
# - v9: 0.58478 =v7!
#     * Back to '2013-08-01' (day 915)
#     * Difference with v7 is that in v7 I forgot to comment out the threshold code
# - v10: 0.58478 (dus optimalizatie werkt prima!)
#     * I can't quite get back to the original 0.55 of v3, despite adding more data. What's going on? 
#     * Could be caused by: a) dropping NaN (but we added more data instead!) b) mistake in optimized prediction loop c) due to the change in event_type handling, d) removing wnd_feats (unlikely)
#     * a), b) seem very unlikely to me. D could be, but notebook <0.50 also didn't use that feature. It could be c, if before events were not used in the prediction and now actually slightly hurt the prediction.
#     * I first want to exclude option c, just to be sure. 
# - v11: ZIE MAIN NOTEBOOK 
#     * Use optimized version
# - v11b TEST: random validation points ipv laatste maand.
# - v12: 0.55541
#     * Less iterations: 1200 -> 600
# - v13: 0.55824
#     * Again, 600 iterations, but now also less data (same amount as previous topscore)
#     * From '2014-8-01'
#     * A bit worse, which makes sense! Finally something that makes sense.
# - v14: 0.55194
#     * Back to '2013-08-01'
#     * Adding stopping condition 10 rounds, max. 1000 iterations
# - v15: 
#     * Stricter stopping condition: 5 rounds, max 2000 iterations.
# - v16: 
#     * 10 early stopping rounds, max 2000 iterations
#     
# TODO deze versie heeft nog niet date en year in make_features.

# # Preprocessing

# ## Imports

# In[ ]:


MODEL_VERSION = 'v15_param_tuning'

import os
import gc #garbage collection
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
from datetime import datetime, timedelta, date # handling dates
from tqdm.notebook import tqdm # progress bars

# LightGBM
import lightgbm as lgb


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Global variables
# 
# TODO: END_DAY is een string, maar dat slaat nergens op. Converteer naar int en refactor.
# 
# Daarom gaat vermoedelijk deze lijn in `load_data` fout: `sales_train_validation = sales_train_validation[(sales_train_validation['day'] >= BEGIN_DAY)]`

# In[ ]:


# Set this to true if you want only one iteration to run, for testing purposes.
TEST_RUN = False

# Do not truncate view when max_cols is exceeded
# ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
pd.set_option('display.max_columns', 50) 

KAGGLE_DATA_FOLDER = '/kaggle/input/m5-forecasting-accuracy'
BACKWARD_LAG = 60
END_DAY = 1913
# Use this if you do not want to load in all data
# Full data starts at 2011-01-29
#BEGIN_DATE = '2015-02-11' 
#BEGIN_DATE = '2014-8-01'
BEGIN_DATE = '2013-08-01'
#BEGIN_DATE = '2012-01-01'
BEGIN_DAY = str((datetime.strptime(BEGIN_DATE, '%Y-%m-%d') - datetime.strptime('2011-01-29', '%Y-%m-%d')).days)
TRAIN_SPLIT = '2016-03-27'
EVAL_SPLIT = '2016-04-24' # In this phase of the competition, this is the end date
print(datetime.strptime(EVAL_SPLIT, '%Y-%m-%d'))
TASK_TYPE='CPU'


# ## Loading data and preprocessing

# ### Data types 
# 
# TODO:
# 
# - Willen we 'd' en bijv. event_name_1 objecten houden? Misschien kan die naar int16. 
#     * Nu zijn event type en name bijv. "unknown". Al helemaal type zou met een integer volstaan.
#     * Update: LightGBM weigert 'object' datatype, dus de conversie naar int of float is noodzakelijk
# 
# Zie bijv. [hier](https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50#Changes)
# 
# ```
# cal[col] = cal[col].cat.codes.astype("int16")
# cal[col] -= cal[col].min()
# ```
# 
# - De snaps zijn nu int16, maar zijn float32(!) in bovenstaande notebook. Ik zie echter niet in waarom ze float32 zouden moeten zijn, aangezien ze maar twee waardes hebben geloof ik.

# In[ ]:


# N.B. LightGBM specifically requires the 'category' dtype
# E.g. see https://stackoverflow.com/questions/56070396/why-does-categorical-feature-of-lightgbm-not-work
CALENDAR_DTYPES = {
    'date':             'str',
    'wm_yr_wk':         'int16', 
    'weekday':          'category',
    'wday':             'int16', 
    'month':            'int16', 
    'year':             'int16', 
    'd':                'object',
    'event_name_1':     'category',
    'event_type_1':     'category',
    'event_name_2':     'category',
    'event_type_2':     'category',
    'snap_CA':          'int16', 
    'snap_TX':          'int16', 
    'snap_WI':          'int16'
}
PARSE_DATES = ['date']
SALES_PRICES_DTYPES = {
    'store_id':    'category', 
    'item_id':     'category', 
    'wm_yr_wk':    'int16',  
    'sell_price':  'float32'#,
    #'sales': 'float32'
}


# ### Loading with preprocessing
# 
# All steps are now integrated into load_data; I did them sequentially before but that resulted in too much overhead.
# 
# #### Convert sales dataframe from wide to long format
# 
# Whereas in the "wide" dataframe one row contains columns with the corresponding sales/demand per day (1913 days at the moment), the new "long" dataframe has a new entry for each day. 
# 
# The resulting dataframe (assuming you use all days) will therefore have 1913-1 less "day" columns (1919-1912+1 = 8 columns), and 30490x1913=58.327.370 rows.
# 
# This is achieved with [pandas.melt](https://pandas.pydata.org/docs/reference/api/pandas.melt.html) by specifying identifier variables and measured variables ('day' in this case), as well as the name of the output value.
# 
# Convert sales_train_validation such that it becomes a function of day with output of sales/demand.
# Unpivots everything not set as id_var, so by default value_vars are all day entries.
# 
# #### Merging dataframes
# 
# [pandas merge doc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html)
# 
# - "left": left outer join that uses keys from left dataframe
# - left dataframe is "sales_train_validation", in which we have just defined the "day" column header
# - join "day" (e.g. d_1) on "d" from the calendar dataframe (also of form d_1)

# In[ ]:


def load_data(train=True):
    """
    Load data
    """
    
    ##### SALES_TRAIN_VALIDATION
    
    print("Loading train and validation data")   
    # Dtype magic from https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50#Changes
    # Required to make LightGBM deal with categorical values
    numcols = [f"d_{day}" for day in range(int(BEGIN_DAY),END_DAY+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    
    sales_train_validation = pd.read_csv(os.path.join(KAGGLE_DATA_FOLDER, 'sales_train_validation.csv'),
                                                     usecols=catcols+numcols, dtype=dtype)
    for col in catcols:
        if col != "id":
            sales_train_validation[col] = sales_train_validation[col].cat.codes.astype("int16")
            sales_train_validation[col] -= sales_train_validation[col].min()
    
    if not train:
        # Add columns for future 28 days, 1914-1941
        for day in range(END_DAY+1, END_DAY+28+1):
            sales_train_validation[f"d_{day}"] = 0  # TODO this was np.nan before

        # Then only keep data from the last BACKWARD_LAG days        
        # If we remove the 'd_' prefix, we can compare day numbers
        value_vars = [column for column in sales_train_validation.columns 
                              if (column.startswith('d_') and int(column.replace('d_', ''))>= END_DAY - BACKWARD_LAG)]
    else:
        # Immediately throw away all days before BEGIN_DAY
        # Doing this so early is important because pd.melt increases memory significantly
        value_vars = [col for col in sales_train_validation.columns 
                      if (col.startswith('d_') and (int(col.replace('d_', '')) >= int(BEGIN_DAY)))]
    
    print("Shape:", sales_train_validation.shape )
    print("Memory usage (Mb) before melting:", sales_train_validation.memory_usage().sum() / 1024**2)
    
    sales_train_validation = pd.melt(
        sales_train_validation, 
        id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        value_vars=value_vars,
        var_name = 'day',
        value_name = 'sales')
    print("Completed melting, new shape:", sales_train_validation.shape )
    print("Colums after melting:", sales_train_validation.columns)
    print("Memory usage (Mb) after melting:", sales_train_validation.memory_usage().sum() / 1024**2)
    
    columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    
    ####### CALENDAR
    
    print("Loading calendar")
    # Parse dates parses the dates as datetime objects! Pandas provides some nice functions on datetime objects.
    calendar = pd.read_csv(os.path.join(KAGGLE_DATA_FOLDER, 'calendar.csv'), dtype=CALENDAR_DTYPES, parse_dates=['date'])
    print("Calendar columns: ", calendar.columns)
    print("Memory usage (Mb) calendar: ", calendar.memory_usage().sum() / 1024**2)
    calendar.rename(columns={'d':'day'}, inplace=True)

    for col, col_dtype in CALENDAR_DTYPES.items():
        if col_dtype == "category":
            calendar[col] = calendar[col].cat.codes.astype("int16")
            calendar[col] -= calendar[col].min()
            
    # Merge sales_train_validation and calendar
    sales_train_validation = sales_train_validation.merge(calendar, on="day", copy=False)
    del calendar; gc.collect()
    print("Merged calendar (in place)")
    print("Memory usage (Mb) after merging calendar:", sales_train_validation.memory_usage().sum() / 1024**2)
    print("Colums after merge:", sales_train_validation.columns)
    
    ####### SELL PRICES
    
    sell_prices = pd.read_csv(os.path.join(KAGGLE_DATA_FOLDER, 'sell_prices.csv'), dtype=SALES_PRICES_DTYPES)
    
    # From https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50#Changes
    # TODO investigate normalization step
    for col, col_dtype in SALES_PRICES_DTYPES.items():
        if col_dtype == "category":
            sell_prices[col] = sell_prices[col].cat.codes.astype("int16")
            sell_prices[col] -= sell_prices[col].min()

    print("Memory usage (Mb) sell prices:", sell_prices.memory_usage().sum() / 1024**2)
    
    columns = ['item_id', 'store_id', 'sell_price']
    for feature in columns:
        if feature == 'sell_price':
            sell_prices[feature].fillna(0, inplace=True)
    
    # Merge in sell prices
    sales_train_validation = sales_train_validation.merge(sell_prices, on=["store_id","item_id","wm_yr_wk"], copy=False)
    del sell_prices; gc.collect()
    print("Merged sales prices (in place)")
    print("Memory usage (Mb) after merging sales:", sales_train_validation.memory_usage().sum() / 1024**2)
    print("Colums after merge:", sales_train_validation.columns)
     
    #submission = pd.read_csv(os.path.join(KAGGLE_DATA_FOLDER, 'sample_submission.csv'))  
    #return reduce_mem_usage(calendar), reduce_mem_usage(sell_prices), reduce_mem_usage(sales_train_validation)
    return sales_train_validation


# ## Feature engineering
# 
# Note that the make_features converts 'day' from an object to an integer, so be aware of this side-effect. I should probably move this to the load_data function.

# See [this tutorial of autoregression](https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/)
# 
# Inspiration: from [here](https://www.kaggle.com/vgarshin/m5-catboost), and from [here](https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50#Changes). 
# 
# The first notebook creates the following features:
# 
# - lag_7           float32
# - lag_28          float32
# - rmean_7_7       float32
# - rmean_28_7      float32
# - rmean_7_28      float32
# - rmean_28_28     float32
# - week            int16
# - quarter         int16
# - mday            int16
# 

# ### Q&A
# 
# Q by Blazej: Why are you calculating rolling means of lags instead of rolling means of the actual values?
# 
# A by Vlad-Marius Griguta: 
# 
# > Good question. The reason for using lagged values of the target variable is to reduce the effect of self-propagating errors through multiple predictions of the same model.
# The objective is to predict 28 days in advance in each series. Therefore, to predict the 1st day in the series you can use the whole series of sales (up to lag1). However, to predict the 8th day you only have actual data for up to lag8 and to predict the whole series you have actuals up to lag28. What people have done at the beginning of the competition was to only use features computed from up to lag28 and apply regression (e.g. lightGBM). This is the safest option, as it does not require the use of 'predictions on predictions'. At the same time, it restrains the capacity of the model to learn features closer to the predicted values. I.e., it underperforms at predicting the 1st day, which could use much more of the latest values in the series than lag28. What this notebook is doing is to find a balance between 'predicting on predictions' and using the latest available information. Using features based on a lag that has some seasonal significance (lag7) seems to give positive results, while the fact that only two features (lag7 and rmean7_7) self-propagate errors keep the over-fitting problem under control.

# - Adding month and date seems to lower performance

# In[ ]:


# This function is copied from m5_catboost 
# minor change: I renamed 'd' to 'day'
# minor change: I pass dates as strings, not datetime, so I convert them
# The date_features contain pandas functions defined on datetimeIndex,
# e.g. https://www.geeksforgeeks.org/python-pandas-datetimeindex-weekofyear/
def make_lag_features(strain):    
    """
    N.B. If you adjust this function, make sure to make make_features_for_day() below
    """
    
    # 1. Lagged sales
    print('in dataframe:', strain.shape)
    print("headers:", strain.columns)
    lags = [7, 28]
    lag_cols = ['lag_{}'.format(lag) for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        strain[lag_col] = strain[['id', 'sales']].groupby('id')['sales'].shift(lag)
    print('lag sales done')
    
    # 2. Rolling means
    windows= [7, 28]
    for window in windows:
        for lag, lag_col in zip(lags, lag_cols):
            window_col = f'rmean_{lag_col}_{window}'
            strain[window_col] = strain[['id', lag_col]].groupby('id')[lag_col].transform(
                lambda x: x.rolling(window).mean()
            )
        print(f'Rolling mean sales for done for window {window}')
   
    # ATTENTION
    # Shit creates some NaNs because lags for the initial days cannot be computed
    # I currently just fill them
    #return strain.fillna(0)

    
def make_date_features(dt):
    # 3. New date features (values are corresponding pandas functions)
    date_features = {
        'week': 'weekofyear',
        'quarter': 'quarter',
        'mday': 'day',
        "wday": "weekday"
    }
    
    # Additional potential date features
    # "wday": "weekday",
    # "month": "month",",
    # "year": "year",
    # "ime": "is_month_end",
    # "ims": "is_month_start",
    #id", "date", "sales", "day", "wm_yr_wk", "weekday
    
    for date_feat_name, date_feat_func in date_features.items():
        if not date_feat_name in dt.columns:
            dt[date_feat_name] = getattr(dt['date'].dt, date_feat_func).astype('int16')      
    print('date features done')
    dt['day'] = dt['day'].apply(lambda x: int(x.replace('d_', '')))  
    print('out dataframe:', dt.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'sales_train_validation = load_data()')


# In[ ]:


sales_train_validation["sale"] = ((sales_train_validation['sell_price'] * 100 % 10) < 6).astype('int8')


# In[ ]:


sales_train_validation.info()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'make_lag_features(sales_train_validation)\nmake_date_features(sales_train_validation)')


# In[ ]:


sales_train_validation.info()


# As you can see, events are encoded as integers. 

# In[ ]:


print('Event type 1 unique values: ', sorted(sales_train_validation.event_type_1.unique()))
print('Event name 1 unique values: ', sorted(sales_train_validation.event_name_1.unique()))
print('Event type 2 unique values: ', sorted(sales_train_validation.event_type_2.unique()))
print('Event name 2 unique values: ', sorted(sales_train_validation.event_name_1.unique()))


# Note that for example event type 2 has two unique values, but three integers. 
# 0 can be interpreted as "absent".
# We can verify that this makes sense, because then we expect 0 to be the absolute majority.

# In[ ]:


total = len(sales_train_validation['event_type_2'])
zero = len(sales_train_validation.loc[sales_train_validation['event_type_2'] == 0])
one = len(sales_train_validation.loc[sales_train_validation['event_type_2'] == 1])
two = len(sales_train_validation.loc[sales_train_validation['event_type_2'] == 2])
print('0: '+ str(zero))
print('1: ' + str(one))
print('2: ' + str(two))
print('-'*7 + '\nTotal: ' + str(total))
del total; del zero; del one; del two


# In[ ]:


# Check NaNs
sales_train_validation.isnull().sum()


# ## Drop NaNs
# 
# By dropping NaNs we can hopefully load more data of better quality.

# In[ ]:


before = len(sales_train_validation)
sales_train_validation.dropna(inplace = True)
after = len(sales_train_validation)
print(f"Reduced {(before-after)/before}%")


# In[ ]:


sales_train_validation.head()


# ### Split data into training, validation and evaluation set
# 
# Was: simple split
# 
# Try: random validation data points

# In[ ]:


print("Splitting data into train, validation, evaluation set")
#train = sales_train_validation[(sales_train_validation['date'] >= BEGIN_DATE) & (sales_train_validation['date'] <= TRAIN_SPLIT)]
#validation = sales_train_validation[(sales_train_validation['date'] > TRAIN_SPLIT) & (sales_train_validation['date'] <= EVAL_SPLIT)]
#evaluation = sales_train_validation[(sales_train_validation['date'] > EVAL_SPLIT)] # This is currently empty
#del sales_train_validation; gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "np.random.seed(777)\n\nvalidation_idx = np.random.choice(sales_train_validation.index.values, 2_000_000, replace = False)\ntrain_idx = np.setdiff1d(sales_train_validation.index.values, validation_idx) # set difference\n\ntrain = sales_train_validation.loc[train_idx]\nvalidation = sales_train_validation.loc[validation_idx]\nevaluation = sales_train_validation[(sales_train_validation['date'] > EVAL_SPLIT)]")


# In[ ]:


print("Train:", train.shape)
print("Validation:", validation.shape)
print("Evaluation:", evaluation.shape)


# In[ ]:


train.head()


# ### TODO: datatypes fixes
# 
# - lag* van float64 -> float32
# - sales van int16 -> float32

# In[ ]:


train.info()


# ### Specify categorical features

# In[ ]:


categorical_features = [
    'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id',
    'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
]


# ### Drop columns that are unnecessary for training

# In[ ]:


# Added all columns after weekday for now, becasue these columns get dropped in the dropna() step later in the cell below - Jordy
# TODO remove features after weekday when we want to use these features. For now, these features are not yet correctly implemented
drop = ["id", "date", "sales", "day", "wm_yr_wk", "weekday"]#, 'lag_28', 'lag_7', 'lag_7_id_rmean_7', 'lag_7_item_id_rmean_28', 'event_name_2', 'event_type_1', 'lag_7_item_id_rmean_7', 'lag_28_id_rmean_28', 'lag_28_id_rmean_7', 'lag_28_item_id_rmean_28', 'event_name_1', 'event_type_2', 'lag_7_id_rmean_28', 'lag_28_item_id_rmean_7']
train_columns = train.columns[~train.columns.isin(drop)]
print(train_columns)


# In[ ]:


print("Train shape:", train.shape)
print("Validation shape:", validation.shape)


# # LightGBM Model definition

# In[ ]:



train_pool = lgb.Dataset(
    data=train[train_columns],
    label=train["sales"], 
    categorical_feature=categorical_features,
    free_raw_data=False)
del train; gc.collect()


# In[ ]:


val_pool = lgb.Dataset(
    data=validation[train_columns],
    label=validation["sales"],
    categorical_feature=categorical_features,
    free_raw_data=False
)
del validation; del evaluation; gc.collect()


# ### LightGBM hyperparameters
# 
# Initial setting from [here](https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50#Changes). Update later.
# 

# In[ ]:


if TEST_RUN:
    ITERATIONS=1
else:
    ITERATIONS = 2000

params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.075,
#         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
#         "nthread" : 4
        "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations' : ITERATIONS,
    'num_leaves': 128,
    "min_data_in_leaf": 100,
    'early_stopping': 10
}


# In[ ]:


get_ipython().run_cell_magic('time', '', 'if TASK_TYPE==\'GPU\':\n    model = "todo"\nelse:\n    model = lgb.train(\n        params,\n        train_pool,\n        valid_sets = val_pool,\n        verbose_eval=20)')


# In[ ]:


model.save_model("model_{}.lgb".format(MODEL_VERSION))


# In[ ]:


del train_pool, val_pool; gc.collect()


# # Feature importance
# 
# TODO replace CatBoost code from below

# ```
# feat_importances = sorted(
#     [(f, v) for f, v in zip(train_columns, model.get_feature_importance())],
#     key=lambda x: x[1],
#     reverse=True
# )
# threshold = .25
# labels = [x[0] for x in feat_importances if x[1] > threshold]
# values = [x[1] for x in feat_importances if x[1] > threshold]
# fig, ax = plt.subplots(figsize=(8, 8))
# y_pos = np.arange(len(labels))
# ax.barh(y_pos, values)
# ax.set_yticks(y_pos)
# ax.set_yticklabels(labels)
# ax.invert_yaxis()
# ax.set_xlabel('Performance')
# ax.set_title('feature importances')
# plt.show()
# ```

# In[ ]:


plt.rcParams['figure.figsize'] = (18.0, 4)
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(12,8))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()


# # Prediction

# ### Load data and add future days

# In[ ]:


#%%time
df = load_data(train=False)
make_date_features(df)


# In[ ]:


df["sale"] = ((df['sell_price'] * 100 % 10) < 6).astype('int8')


# In[ ]:


df.info()


# ### Prediction loop
# 
# Apply the "recursive features" approach here:
# 
# - Predict the next day based on last BACKWARD_LAG days
# - Perform feature engineering on those days (same as during training)
# - Repeat, but now include the day for which we just predicted demand
# 
# TODO: [use some weighing scheme](https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50#Changes)
# 
# Optimization: only compute the required features for a prediction day for an immense speedup.

# In[ ]:


# TODO als ik deze functie gebruik gaat iets grandioos fout
# Returnt een numpy ndarray!

# Code to just compute features only for the single prediction day
# Adapted with several changes from https://www.kaggle.com/poedator/m5-under-0-50-optimized#Prediction-stage 
def lag_features_for_day(dt, day):
    print(type(dt))
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        dt.loc[dt['date'] == str(day), lag_col] =             dt.loc[dt['date'] == str(day-timedelta(days=lag)), 'sales'].values
    
    windows = [7, 28]
    for window in windows:
        for lag, lag_col in zip(lags, lag_cols):
            df_window = dt[(dt['date'] <= str(day-timedelta(days=lag))) & (dt['date'] > str(day-timedelta(days=lag+window)))]
            df_window_grouped = df_window.groupby("id").agg({'sales':'mean'}).reindex(dt.loc[dt['date']==str(day),'id'])
            dt.loc[dt['date'] == str(day),f'rmean_{lag_col}_{window}'] = df_window_grouped.sales.values   
    print("Lag features done")
    print(type(dt))


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nEND_DATE = EVAL_SPLIT\nZERO_THRESHOLD = 0.01\nPREDICT_DAYS = 28\n\n# TODO check of fillen met 0 hierboven goed gaat; i.e. of fillna nu nog nodig is\n#df[\'sales\']=df[\'sales\'].fillna(0)\n\n# Predict from 2016-04-25 on\nfor f_day in tqdm(range(1,PREDICT_DAYS+1)):\n    pred_date = (datetime.strptime(END_DATE, \'%Y-%m-%d\') + timedelta(days=f_day)).date()\n    print(f"Forecasting day {END_DAY+f_day}, date: {str(pred_date)}")\n    pred_begin_date = pred_date - timedelta(days=BACKWARD_LAG+1)\n    # Select last BACKWARD_LAG days to use for predicting\n    prediction_data = df[(df[\'date\'] >= str(pred_begin_date)) & (df[\'date\'] <= str(pred_date))].copy()\n    \n    # Repeat feature engineering\n    # Following line does feature engineering on the whole bunch, but in fact we only need features for pred_date\n    #make_lag_features(prediction_data)\n    lag_features_for_day(prediction_data, pred_date)\n    \n    # Only use the columns you trained on before\n    prediction_data = prediction_data.loc[prediction_data[\'date\'] == str(pred_date), train_columns]\n    prediction =  model.predict(prediction_data)   \n    print("Prediction", prediction.size, prediction)\n    df.loc[df[\'date\'] == str(pred_date), \'sales\'] = prediction\n    \n# If predictions are very close to zero, predict a gap day \n#df.loc[df[\'sales\'] < ZERO_THRESHOLD, \'sales\'] = 0  ')


# In[ ]:


del prediction_data
gc.collect()


# # Submission
# 
# Now let's turn the prediction into a submission file. 
# We'll wrangle the long dataframe with the predictions into the correct format.
# 
# cf. https://medium.com/@durgaswaroop/reshaping-pandas-dataframes-melt-and-unmelt-9f57518c7738 
# 
# - Currently we only predict 30490 rows corresponding to the validation set. Later in the competition, another 30490 rows corresponding to the evaluation set will be added. 
# - For now, to get the correct submission format, we simply copy the predictions of the first 30490 rows. 
# 

# In[ ]:


# We are only interested in the predicted days
# We need the id for the row index, the day to calculate F_{x}, and the sales for the prediction values
submission_val = df.loc[df['date'] > END_DATE, ['id', 'day', 'sales']].copy()

# Memory clean-up
#del df; gc.collect()

# Do not make negative predictions
submission_val.loc[submission_val['sales'] < 0, 'sales'] = 0

# Sort on id 
submission_val.sort_values('id', inplace=True)

#submission_val['day'] = submission_val['day'].apply(lambda x: 'F{}'.format(int(x.replace('d_', '')) - END_DAY))
# Use code below if you have used make_features on df, because it replaces day with an int already.
submission_val['day'] = submission_val['day'].apply(lambda x: 'F{}'.format(x - END_DAY))
print(submission_val.columns)


# Now we have a single 'day' column. Instead, we want to have a separate column for each day.
# The reverse of the melt operation is `pivot` .
# An extra 'sales' descriptor is introduced that we remove again.
# 'id' will serve as the index, but we want to reintroduce it as a column for submission with `reset_index()`
# 

# In[ ]:


# This is required to force the correct ordering after reshaping
f_cols = ['F{}'.format(x) for x in range(1, 28 + 1)]

submission_val = submission_val.pivot(index='id', columns='day')['sales'][f_cols].reset_index(level='id')
print(submission_val.columns)


# In[ ]:


# Temporary solution, copy the 28 validation days as the 28 evaluation days
submission_eval = submission_val.copy()

submission_eval['id'] = submission_eval['id'].str.replace('validation', 'evaluation')
submission = pd.concat([submission_val, submission_eval], axis=0, sort=False)
#spred_subm.reset_index(drop=True, inplace=True)
print(submission.columns)
print(submission.head(1))


# In[ ]:


submission.to_csv('submission.csv', index=False)
print('Submission shape', submission.shape)

