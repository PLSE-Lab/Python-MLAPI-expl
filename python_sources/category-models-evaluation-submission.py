#!/usr/bin/env python
# coding: utf-8

# # Category Models Evaluation Submission
# 
# Link to the notebook with submission on validation data: [https://www.kaggle.com/jordai/rausnaus-lightgbm-per-category-best-notebook?scriptVersionId=35174007](https://www.kaggle.com/jordai/rausnaus-lightgbm-per-category-best-notebook?scriptVersionId=35174007)

# # Preprocessing

# ## Imports

# In[ ]:


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

# In[ ]:


# Set this to true if you want only one iteration to run, for testing purposes.
TEST_RUN = False

MODEL_VERSION = 'final'

# Do not truncate view when max_cols is exceeded
# ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
pd.set_option('display.max_columns', 50) 

# Path to Data Folder
KAGGLE_DATA_FOLDER = '/kaggle/input/m5-forecasting-accuracy'
# Path to submission on validation data
SUBMISSION_PATH = "/kaggle/input/rausnaus-lightgbm-per-category-best-notebook/submission.csv"

# Paths to Models per Category
PATH_TO_CAT0 = "/kaggle/input/rausnaus-lightgbm-per-category-best-notebook/model_cat0_v10.lgb"
PATH_TO_CAT1 = "/kaggle/input/rausnaus-lightgbm-per-category-best-notebook/model_cat1_v10.lgb"
PATH_TO_CAT2 = "/kaggle/input/rausnaus-lightgbm-per-category-best-notebook/model_cat2_v10.lgb"
PATH_MODELS = [PATH_TO_CAT0, PATH_TO_CAT1, PATH_TO_CAT2]

KAGGLE_DATA_FOLDER = '/kaggle/input/m5-forecasting-accuracy'
BACKWARD_LAG = 60
END_DAY = 1913+28
# Use this if you do not want to load in all data
# Full data starts at 2011-01-29
BEGIN_DATE = '2013-08-01'
BEGIN_DAY = str((datetime.strptime(BEGIN_DATE, '%Y-%m-%d') - datetime.strptime('2011-01-29', '%Y-%m-%d')).days)
EVAL_SPLIT = '2016-05-22' 
TRAIN_SPLIT = str(datetime.strptime(EVAL_SPLIT, '%Y-%m-%d') - timedelta(days=28))[:10]
print(f"Train split: {datetime.strptime(TRAIN_SPLIT, '%Y-%m-%d')}")
print(f"Eval split: {datetime.strptime(EVAL_SPLIT, '%Y-%m-%d')}")


# ## Loading data and preprocessing

# ### Data types 

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
    'sell_price':  'float32',
#     'sales': 'float32'
}


# ### Loading with preprocessing
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
    
    sales_train_validation = pd.read_csv(os.path.join(KAGGLE_DATA_FOLDER, 'sales_train_evaluation.csv'),
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

# ### Q&A
# 
# Q by Blazej: Why are you calculating rolling means of lags instead of rolling means of the actual values?
# 
# A by Vlad-Marius Griguta: 
# 
# > Good question. The reason for using lagged values of the target variable is to reduce the effect of self-propagating errors through multiple predictions of the same model.
# The objective is to predict 28 days in advance in each series. Therefore, to predict the 1st day in the series you can use the whole series of sales (up to lag1). However, to predict the 8th day you only have actual data for up to lag8 and to predict the whole series you have actuals up to lag28. What people have done at the beginning of the competition was to only use features computed from up to lag28 and apply regression (e.g. lightGBM). This is the safest option, as it does not require the use of 'predictions on predictions'. At the same time, it restrains the capacity of the model to learn features closer to the predicted values. I.e., it underperforms at predicting the 1st day, which could use much more of the latest values in the series than lag28. What this notebook is doing is to find a balance between 'predicting on predictions' and using the latest available information. Using features based on a lag that has some seasonal significance (lag7) seems to give positive results, while the fact that only two features (lag7 and rmean7_7) self-propagate errors keep the over-fitting problem under control.

# - N.B. month and year are already in the data. In this case, the only thing we do is reduce memory to int16. 

# In[ ]:


# This function is copied from m5_catboost 
# minor change: I renamed 'd' to 'day'
# minor change: I pass dates as strings, not datetime, so I convert them
# The date_features contain pandas functions defined on datetimeIndex,
# e.g. https://www.geeksforgeeks.org/python-pandas-datetimeindex-weekofyear/
def make_lag_features(strain):    
    """
    N.B. If you adjust this function, make sure to also adjust make_features_for_day() below
    """
    
    # 1. Lagged sales
    print('in dataframe:', strain.shape)
    print("headers:", strain.columns)
    lags = [7, 28]
    lag_cols = ['lag_{}'.format(lag) for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        # ATTENTIE: hier stond eerst id
        strain[lag_col] = strain[['item_id', 'sales']].groupby('item_id')['sales'].shift(lag)
    print('lag sales done')
    
    # 2. Rolling means for id (so aggregated sales on item, independent of store and state etc.)
    windows= [7, 28]
    #for window in windows:
    #    for lag, lag_col in zip(lags, lag_cols):
    #        window_col = f'id_rmean_{lag_col}_{window}'
    #        strain[window_col] = strain[['id', lag_col]].groupby('id')[lag_col].transform(
    #            lambda x: x.rolling(window).mean()
    #        )
    #    print(f'Rolling mean sales done for window {window} per item')
       
    # 3. Rolling means for item_id (items per state and per category type, so quite specific!)
    for window in windows:
        for lag, lag_col in zip(lags, lag_cols):
            window_col = f'item_id_rmean_{lag_col}_{window}'
            strain[window_col] = strain[['item_id', lag_col]].groupby('item_id')[lag_col].transform(
                lambda x: x.rolling(window).mean()
            )
        print(f'Rolling mean sales done for window {window} per item_id')
    
    # 4. Rolling means for store_id, last week and last month
    #for window in windows:
    #    for lag, lag_col in zip(lags, lag_cols):
    #        window_col = f'store_id_rmean_{lag_col}_{window}'
    #        strain[window_col] = strain[['store_id', lag_col]].groupby('store_id')[lag_col].transform(
    #            lambda x: x.rolling(window).mean()
    #        )
          
    #window = 28
    #lag_col = 'lag_28'
    #window_col = f'store_id_rmean_{lag_col}_{window}'
    #[window_col] = strain[['store_id', lag_col]].groupby('store_id')[lag_col].transform(
    #    lambda x: x.rolling(window).mean())
    #print(f'Rolling mean sales done for lag {lag_col} and window {window} per store_id')
    
def make_date_features(dt):
    # 3. New date features (values are corresponding pandas functions)
    # Again, month and year are already in the original data
    date_features = {
        'week': 'weekofyear',
        'quarter': 'quarter',
        'mday': 'day',
        "wday": "weekday",
        "month": "month",
        "year": "year"
    }
    
    # Additional potential date features
    # "ime": "is_month_end",
    # "ims": "is_month_start",
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt['date'].dt, date_feat_func).astype('int16') 
        
    print('date features done')
    dt['day'] = dt['day'].apply(lambda x: int(x.replace('d_', '')))  
    print('out dataframe:', dt.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'sales_train_evaluation = load_data()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'sales_train_evaluation["sale"] = ((sales_train_evaluation[\'sell_price\'] * 100 % 10) < 6).astype(\'int8\')\nmake_lag_features(sales_train_evaluation)\nmake_date_features(sales_train_evaluation)')


# ### Drop NaNs

# In[ ]:


before = len(sales_train_evaluation)
sales_train_evaluation.dropna(inplace = True)
after = len(sales_train_evaluation)
print(f"Reduced {(before-after)/before}%")


# ## Splitting into train/validation

# In[ ]:


print("Splitting data into train, validation, evaluation set")
train = sales_train_evaluation[(sales_train_evaluation['date'] >= BEGIN_DATE) & (sales_train_evaluation['date'] <= TRAIN_SPLIT)]
validation = sales_train_evaluation[(sales_train_evaluation['date'] > TRAIN_SPLIT) & (sales_train_evaluation['date'] <= EVAL_SPLIT)]
del sales_train_evaluation; gc.collect()


# In[ ]:


print(train["cat_id"].value_counts())
print(validation["cat_id"].value_counts())


# In[ ]:


categorical_features = [
    'item_id', 'dept_id', 'store_id', 'state_id',
    'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
]


# In[ ]:


drop = ["id", "date", "sales", "day", "wm_yr_wk", "weekday", "cat_id"]
train_columns = train.columns[~train.columns.isin(drop)]
print(train_columns)


# In[ ]:


train_sets = [train[train["cat_id"] == category].drop("cat_id", axis = 1) for category in range(3)]
del train; gc.collect()
val_sets = [validation[validation["cat_id"] == category].drop("cat_id", axis = 1) for category in range(3)]
del validation; gc.collect()


# ## LGBM Model Definition

# In[ ]:


train_pools = [lgb.Dataset(
                            data=train_sets[category][train_columns],
                            label=train_sets[category]["sales"], 
                            categorical_feature=categorical_features,
                            free_raw_data=False)
               for category in range(3)]
del train_sets; gc.collect()


# In[ ]:


val_pools = [lgb.Dataset(
                         data=val_sets[category][train_columns],
                         label=val_sets[category]["sales"],
                         categorical_feature=categorical_features,
                         free_raw_data=False)
             for category in range(3)]
del val_sets; gc.collect()


# In[ ]:


if TEST_RUN:
    ITERATIONS=1
else:
    ITERATIONS = 1200

params = {
           "objective" : "poisson",
           "metric" : "rmse",
           "force_row_wise" : True,
           "learning_rate" : 0.075,
           "sub_row" : 0.75,
           "bagging_freq" : 1,
           "lambda_l2" : 0.1,
           'verbosity': 1,
           'num_iterations': ITERATIONS,
           'num_leaves': 128,
           "min_data_in_leaf": 100,
         }


# In[ ]:


get_ipython().run_cell_magic('time', '', 'models = [lgb.train(params,\n                    train_pools[category],\n                    valid_sets = val_pools[category],\n                    verbose_eval=20)\n          for category in range(3)]\nfor cat,model in enumerate(models):\n    model.save_model(f"model_cat{cat}_{MODEL_VERSION}.lgb")')


# In[ ]:


del train_pools, val_pools; gc.collect()


# ### Feature Importance

# In[ ]:


plt.rcParams['figure.figsize'] = (18.0, 4)
get_ipython().run_line_magic('matplotlib', 'inline')

for category in range(3):
    fig, ax = plt.subplots(figsize=(12,8))
    lgb.plot_importance(models[category], max_num_features=50, height=0.8, ax=ax)
    ax.grid(False)
    plt.title("LightGBM - Feature Importance - Category "+str(category), fontsize=15)
    plt.show()


# # Prediction

# ### Load data and add future days

# In[ ]:


#%%time
df = load_data(train=False)

df["sale"] = ((df['sell_price'] * 100 % 10) < 6).astype('int8')
# make_lag_features(df)
make_date_features(df)


# In[ ]:


df.info()


# ### Prediction loop
# 
# Apply the "recursive features" approach here:
# 
# - Predict the next day based on last BACKWARD_LAG days
# - Perform feature engineering on those days (same as during training)
# - Repeat, but now include the day for which we just predicted demand

# In[ ]:


# TODO als ik deze functie gebruik gaat iets grandioos fout
# Returnt een numpy ndarray!

# Code to just compute features only for the single prediction day
# Adapted with several changes from https://www.kaggle.com/poedator/m5-under-0-50-optimized#Prediction-stage 
def lag_features_for_day(dt, day):
    print(type(dt))
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    # 1. Lag sales
    for lag, lag_col in zip(lags, lag_cols):
        dt.loc[dt['date'] == str(day), lag_col] =             dt.loc[dt['date'] == str(day-timedelta(days=lag)), 'sales'].values
    
    windows = [7, 28]
    for window in windows:
        for lag, lag_col in zip(lags, lag_cols):
            df_window = dt[(dt['date'] <= str(day-timedelta(days=lag))) & (dt['date'] > str(day-timedelta(days=lag+window)))]
            
            # 2. Rolling means for id (so aggregated sales on item, independent of store and state etc.)
            #df_window_grouped = df_window.groupby("id").agg({'sales':'mean'}).reindex(dt.loc[dt['date']==str(day),'id'])
            #dt.loc[dt['date'] == str(day),f'id_rmean_{lag_col}_{window}'] = df_window_grouped.sales.values   
            
            # 3. Rolling means for item_id (tems per state and per category type, so quite specific!)
            df_window_grouped = df_window.groupby("item_id").agg({'sales':'mean'}).reindex(dt.loc[dt['date']==str(day),'item_id'])
            dt.loc[dt['date'] == str(day),f'item_id_rmean_{lag_col}_{window}'] = df_window_grouped.sales.values   
            
    # 4. Rolling mean for store_id lag 28 last 28 days
    lag = 28
    lag_col = 'lag_28'
    window = 28
    df_window = dt[(dt['date'] <= str(day-timedelta(days=lag))) & (dt['date'] > str(day-timedelta(days=lag+window)))]
    df_window_grouped = df_window.groupby("store_id").agg({'sales':'mean'}).reindex(dt.loc[dt['date']==str(day),'store_id'])
    dt.loc[dt['date'] == str(day),f'store_id_rmean_{lag_col}_{window}'] = df_window_grouped.sales.values
    print(f"Features done for day {str(day)}") 
        


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nEND_DATE = EVAL_SPLIT\nPREDICT_DAYS = 28\n\n# Predict from 2016-05-22 on\nfor f_day in tqdm(range(1,PREDICT_DAYS+1)):\n    pred_date = (datetime.strptime(END_DATE, \'%Y-%m-%d\') + timedelta(days=f_day)).date()\n    print(f"Forecasting day {END_DAY+f_day}, date: {str(pred_date)}")\n    pred_begin_date = pred_date - timedelta(days=BACKWARD_LAG+1)\n    print(pred_begin_date)\n    # Select last BACKWARD_LAG days to use for predicting\n    prediction_data = df[(df[\'date\'] >= str(pred_begin_date)) & (df[\'date\'] <= str(pred_date))].copy()\n    \n    # Repeat feature engineering\n    lag_features_for_day(prediction_data, pred_date)\n\n    # Make predictions per category\n    for i in range(3):\n        # Only use the columns you trained on before\n        prediction_data_cat = prediction_data.loc[(prediction_data[\'date\'] == str(pred_date)) & (prediction_data["cat_id"] == i), train_columns]\n        prediction = models[i].predict(prediction_data_cat)\n        df.loc[(df[\'date\'] == str(pred_date)) & (df.index.isin(prediction_data_cat.index)), \'sales\'] = prediction\n    \n    print("Prediction", prediction.size, prediction)')


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
# We copy the old validation data from a submission, such that we maintain our current public leaderboard score (instead of getting a score of 0). Then, the newly trained model (also on the public leaderboard data) makes predictions for the private leaderboard data.
# 

# In[ ]:


sales_ = pd.read_csv(os.path.join(KAGGLE_DATA_FOLDER, 'sales_train_evaluation.csv'))
submission = pd.read_csv(os.path.join(SUBMISSION_PATH))


# In[ ]:


# We are only interested in the predicted days
# We need the id for the row index, the day to calculate F_{x}, and the sales for the prediction values
submission_eval = df.loc[df['date'] > END_DATE, ['id', 'day', 'sales']].copy()

# Do not make negative predictions
submission_eval.loc[submission_eval['sales'] < 0, 'sales'] = 0

# Sort on id 
submission_eval.sort_values('id', inplace=True)

submission_eval['day'] = submission_eval['day'].apply(lambda x: 'F{}'.format(x - END_DAY))
print(submission_eval.columns)
print(submission_eval.head(), submission_eval.tail(), sep="\n")


# Now we have a single 'day' column. Instead, we want to have a separate column for each day.
# The reverse of the melt operation is `pivot` .
# An extra 'sales' descriptor is introduced that we remove again.
# 'id' will serve as the index, but we want to reintroduce it as a column for submission with `reset_index()`
# 

# In[ ]:


# This is required to force the correct ordering after reshaping
f_cols = ['F{}'.format(x) for x in range(1, 28 + 1)]

submission_eval = submission_eval.pivot(index='id', columns='day')['sales'][f_cols].reset_index(level='id')
print(submission_eval.head(), submission_eval.shape, submission_eval.columns, sep="\n")


# In[ ]:


submission.iloc[30490:, 1:] = submission_eval.iloc[:,1:].to_numpy()
print(submission.head(), submission.tail(), sep="\n")


# In[ ]:


submission.to_csv('submission.csv', index=False)
print('Submission shape', submission.shape)


# In[ ]:




