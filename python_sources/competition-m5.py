#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import random

import numpy as np
import pandas as pd
import seaborn as sns

from tqdm.notebook import tqdm 
from IPython.display import clear_output

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# In[ ]:


# Loading datasets

# Contains date features, like week of year and holidays
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv', index_col='d')

# Contains item sales by day, the target features 
sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

# Contais item sell prices by departments
sell = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')


# ## Objective
# 
# The detailed competition objective and rules are described in [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy) competition page.
# 
# The summary is that: given a retail product and stores historical sales by day dataset, the goal is forecast the sales by day over 30k products for next 27 days.

# ### Notes
# - `d_` columns are the goal columns in past to use in model.
# - The predictions are about columns `d_1914 - d_1941` 

# ### Parameters

# In[ ]:


# Days in past to retrive data for model training 
LAST_N_DAYS = 365 * 2


# ## Sample from population
# The dataset are huge*. There are over 30k of items and each of them have historical sales data over `LAST_N_DAYS` days. 
# 
# Lets use a sample of data sets for predictions. 
# 
# \* *In this study, its about 22 milion rows of data (30490 items x 730 days)* 

# In[ ]:


# CALCULATE THE SAMPLE SIZE
# Original code from Marek Madejski in his github repository: https://github.com/veekaybee/data/blob/master/samplesize.py
# Extracted from his article for sample size: http://veekaybee.github.io/2015/08/04/how-big-of-a-sample-size-do-you-need/ 
def get_sample_size(population_size, confidence_level, confidence_interval):
    # SUPPORTED CONFIDENCE LEVELS: 50%, 68%, 90%, 95%, and 99%
    confidence_level_constant = [50,.67], [68,.99], [90,1.64], [95,1.96], [99,2.57]
    
    Z = 0.0
    p = 0.5
    e = confidence_interval/100.0
    N = population_size
    n_0 = 0.0
    n = 0.0

    # LOOP THROUGH SUPPORTED CONFIDENCE LEVELS AND FIND THE NUM STD
    # DEVIATIONS FOR THAT CONFIDENCE LEVEL
    for i in confidence_level_constant:
        if i[0] == confidence_level:
            Z = i[1]

    if Z == 0.0:
        return -1

    # CALC SAMPLE SIZE
    n_0 = ((Z**2) * p * (1-p)) / (e**2)

    # ADJUST SAMPLE SIZE FOR FINITE POPULATION
    n = n_0 / (1 + ((n_0 - 1) / float(N)) )

    return int(math.ceil(n)) # THE SAMPLE SIZE


# Calculating the sample size needed to make predictions. At this point in competiton, to improve my leaderboard, I make aggressive parameters.
# 
# In real life (and avoid overfitting too) the good confidence level is about 95 and margin error acceptable is about 2.

# In[ ]:


# Calculating the sample size
population = len(sales)
confidence_level = 99.0
margin_error_acceptable = 1

sample_size = get_sample_size(population, confidence_level, margin_error_acceptable)

print(f"Sample size needed: {sample_size}")


# ### Check items by store
# To define the sample data, we will divide it proportionally according to the quantity of items per store.

# In[ ]:


sales.store_id.value_counts()


# Well, we have the same quantitiy of items in each store. So, we must divide the sample equally for each store.

# In[ ]:


stores_quantity = len(sales.store_id.unique())

store_sample_size = round(sample_size / stores_quantity)

print(f"Each store sample size will have {store_sample_size} items")


# ### Split the store samples

# In[ ]:


def get_item_samples_by_store(store_id, sample_size):
        
    temp = sales.query(f"store_id == '{store_id}'").copy()
    temp['index'] = temp.index
    
    # List that will containing the sample indexes
    items_sample_indexes = []
    
    for sample in range(sample_size):
        row = random.randint(0, len(temp)-1)

        item_index = temp.iloc[row]['index']

        items_sample_indexes.append(item_index)
        
        # Drops the sample drawn 
        temp.drop(item_index, axis=0, inplace=True)
            
    return items_sample_indexes


# Drawing samples for each store
# 

# In[ ]:


stores = sales.store_id.unique()

samples_by_store = {}

# Progress bar params
total = len(stores)
desc = 'Drawing samples by store'
    
for store in tqdm(stores, total=total, desc=desc):
    items_index_list = get_item_samples_by_store(store, store_sample_size)
    samples_by_store.update( {store : items_index_list} )


# ## Preprocessing data
# 
# Cleaning dataframes to preprocessing

# In[ ]:


# Preparing feature name cols

initial_day = 1914 - LAST_N_DAYS

train_indexes = [
    f"d_{x}"
    for x in range(initial_day, 1914)
]

target_indexes = [
    f"d_{x}"
    for x in range(1914, 1970)
]

sales_train_features = list(sales.columns[1:6])
sales_train_features


# In[ ]:


# Dataframes to get features and targets data

# Contains historical sales for each day and item
df_targets = sales[train_indexes].copy()
df_targets


# In[ ]:


# Contains store and item names
df_features = sales[sales_train_features].copy()
df_features


# In[ ]:


# Calendar (empty) dataframe for prediction
last = len(target_indexes) - 1
calendar_predict = calendar.loc[target_indexes[0]:target_indexes[last]]
calendar_predict


# In[ ]:


# Filtering calendar dataset for data from LAST_N_DAYS
last = len(train_indexes) - 1
calendar = calendar.loc[train_indexes[0]:train_indexes[last]]
calendar


# In[ ]:


# Historical sell data
sell = sell.query(f"wm_yr_wk >= {min(calendar.wm_yr_wk)}")
sell


# In[ ]:


# Relevant features in calendar
calendar_cols = [
    'wm_yr_wk',
    'wday',
    'month',
    'year',
    'event_type_1',
    'event_type_2'
]

calendar = calendar[calendar_cols].copy()
calendar_predict = calendar_predict[calendar_cols].copy()

calendar


# ### Encoding string features 

# In[ ]:


from sklearn.preprocessing import LabelEncoder

features_encoder = LabelEncoder()
calendar_encoder = LabelEncoder()

# Encoding sales features
df_features_encoded = df_features.apply(features_encoder.fit_transform)

# Encoding calendar features
calendar_cols_encode = calendar.select_dtypes('object').columns

calendar[calendar_cols_encode] = calendar[calendar_cols_encode].fillna('-')
calendar[calendar_cols_encode] = calendar[calendar_cols_encode].apply(calendar_encoder.fit_transform)

calendar_predict[calendar_cols_encode] = calendar_predict[calendar_cols_encode].fillna('-')
calendar_predict[calendar_cols_encode] = calendar_predict[calendar_cols_encode].apply(calendar_encoder.fit_transform) 

calendar


# In[ ]:


# Sells features
sells_encoder = LabelEncoder()

sells_encode_cols = ['store_id', 'item_id']
sells_features = ['wm_yr_wk', 'sell_price']

sell_encoded = sell[sells_encode_cols].apply(sells_encoder.fit_transform) 
sell_encoded[sells_features] = sell[sells_features]

sell_encoded


# ### Transposing data from `sell` and `target` dataframes
# Data transposing for preparing datasets to made sales predictions

# In[ ]:


# Multidimension dataframes (for each store)
train_data = []

# Just print processing stores rows
current_store = 0
total_stores = len(samples_by_store)

    
for stores_samples in samples_by_store.values():
    current_store += 1
    for index in tqdm(stores_samples, total=len(stores_samples), desc=f'Creating mutidimensional dataframe {current_store}/{total_stores}'):
        # Transposing targets for each item
        # New datafame in multidimension dataframe
        tmp_train_data = pd.DataFrame()
        
        # Calendar features
        for col in calendar_cols:
            tmp_train_data[col] = np.array(calendar[col])

        # Unique id to match sell price in sell dataframe
        item_id = df_features.loc[index, 'item_id']
        store_id = df_features.loc[index, 'store_id']

        # Sales features
        for feature in sales_train_features:
            tmp_train_data[feature] = df_features_encoded.loc[index, feature]

        # Input sell price    
        min_week = min(tmp_train_data.wm_yr_wk)

        sell_cols = ['wm_yr_wk', 'item_id', 'store_id', 'sell_price']
        sell_item = sell_encoded.query(f"item_id == '{item_id}' & store_id == '{store_id}' & wm_yr_wk >= {min_week}")[sell_cols]

        tmp_train_data['sell_price'] = tmp_train_data.merge(sell_item, on='wm_yr_wk', how='left')['sell_price']

        # No sell price indicates no sell made
        tmp_train_data['sell_price'].fillna(0, inplace=True)

        # Target transpose from row to column
        tmp_train_data['target'] = np.array(df_targets.iloc[index].values)
        
        train_data.append(tmp_train_data)
        
    clear_output(wait=True)


# ### Defining `X` and `y` data

# In[ ]:


X = pd.DataFrame()
y = np.array([])

for dataset in train_data:
    X = X.append(dataset.loc[:, dataset.columns != 'target'])
    y = np.append(y, dataset['target'].to_numpy())


# In[ ]:


# year col is not necessary for predicting
# causes a worst prediction
X.drop('year', axis=1, inplace=True) 


# ### Splitting train and test data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


# ### Model evaluation
# 
# The best model prediction was got using `XGBoost`. The evaluation was made by tries with different hyper params, and comparing to `RandomForest` model too.
# 
# I tried to run a RandomGridSearch to find the best params, but this runs hours by all night along and in the end, causes a memory exception, so that's very difficult to use in this case. 

# Making cross validation

# In[ ]:


# Params for evaluation
params_xgb = {
    "objective": "reg:squarederror",
    'max_depth': 50, 
    'min_child_weight': 1,    
    'learning_rate': 0.05,
    'alpha': 10,
    'gamma': 10
}

fit_params = {
    'early_stopping_rounds': 30,
    'eval_metric': 'mae',
    'eval_set': [[X_test, y_test]]
}


# In[ ]:


def get_xg_boost_mae(hyper_params, fit_params, X, y):
    model_xgb = XGBRegressor(
        n_estimators=1000,
        objective=hyper_params['objective'],
        max_depth=hyper_params['max_depth'],
        min_child_weight=hyper_params['min_child_weight'],
        alpha=hyper_params['alpha'],
        gamma=hyper_params['gamma'],
        n_jobs=4
    )
    return cross_val_score(
        model_xgb, 
        X, y, 
        cv=5, 
        scoring='neg_mean_absolute_error', 
        fit_params=fit_params
    ).mean()


# In[ ]:


mae = get_xg_boost_mae(params_xgb, fit_params, X_train, y_train)


# In[ ]:


mae * -1


# ### Train
# 
# XGBoost model with best hyper params

# In[ ]:


def set_xgb_model():
    return XGBRegressor(
        n_estimators=1000,
        objective=params_xgb['objective'],
        max_depth=params_xgb['max_depth'],
        min_child_weight=params_xgb['min_child_weight'],
        alpha=params_xgb['alpha'],
        gamma=params_xgb['gamma'],
        early_stopping_rounds=fit_params['early_stopping_rounds'],
        learning_rate=params_xgb['learning_rate'], 
        n_jobs=4  
    )


# In[ ]:


# Preparing dataset for training model
train_data = X.copy()
train_data['target'] = y


# In[ ]:


# Training a model for each store
models = []

for store in tqdm(X.store_id.unique(), total=len(X.store_id.unique()), desc='Fitting a model for each store...'):
    models.append(set_xgb_model())
    
    # X_store: contains data only for the store model
    X_store = train_data.loc[train_data.store_id == store].copy() 
    
    # Important: drop target for X training
    X_store.drop('target', axis=1, inplace=True)
     
    y_store = train_data.loc[train_data.store_id == store]['target']
    
    models[store].fit(X_store, y_store)   


# ## Predict
# 
# First of all, lets prepare the predict dataframe

# In[ ]:


# Creating dataframe to with features to predict sales 
df_predict = pd.DataFrame()

sell_cols = ['wm_yr_wk', 'item_id', 'store_id', 'sell_price']

for index in tqdm(sales.index, total=len(sales), desc=f'Creating predict dataframe'):
    # Transposing targets for each item
    # New datafame in multidimension dataframe
    tmp_df = pd.DataFrame()

    # Calendar features
    for col in calendar_cols:
        tmp_df[col] = np.array(calendar_predict[col])

    # Sales features
    for feature in sales_train_features:
        tmp_df[feature] = df_features_encoded.loc[index, feature]

    # Input sell price
    min_week = min(tmp_df.wm_yr_wk)
    
    sell_item = sell_encoded.query(f"item_id == '{item_id}' & store_id == '{store_id}' & wm_yr_wk >= {min_week}")[sell_cols]

    tmp_df['sell_price'] = tmp_df.merge(sell_item, on='wm_yr_wk', how='left')['sell_price']

    # No sell price indicates no sell made
    tmp_df['sell_price'].fillna(0, inplace=True)
 
    df_predict = df_predict.append(tmp_df)  

clear_output(wait=True)


# In[ ]:


# Analyzing MAE for different hyper params
# predict with year column causes a worst prediction
# So, let's drop it
df_predict.drop('year', axis=1, inplace=True) 


# In[ ]:


# Creating target column
df_predict['predictions'] = None


# Making predictions

# In[ ]:


# Making predictions for items by each store model
for store in tqdm(X.store_id.unique(), total=len(X.store_id.unique()), desc='Making predictions for each store...'):
    
    # Getting a dataset contaning a single store
    to_predict = df_predict.loc[df_predict.store_id == store].copy()
    
    # Drop null predictions column for this dataset to be predicted 
    to_predict.drop('predictions', axis=1, inplace=True)

    # Making predictions using the specific store model    
    predictions = np.round(models[store].predict(to_predict))

    # Assigning the predictions in result dataset    
    df_predict.loc[df_predict.store_id == store, 'predictions'] = np.absolute(predictions)


# Reversing encoded features

# In[ ]:


# item_id decoded
df_predict['item_id'] = sells_encoder.inverse_transform(df_predict['item_id'])
df_predict.head() 


# In[ ]:


# This inverse_transform for store_id is not working, so its made manually
reverse_encode_dict = {
    key : value
    for key, value in zip(df_features_encoded.store_id.unique(), df_features.store_id.unique())
} 

df_predict.replace({"store_id": reverse_encode_dict}, inplace=True) 


# In[ ]:


# Preparing item name for submission dataset
df_predict['item_validation'] = df_predict.apply(lambda x: x.item_id + '_' + x.store_id , axis=1)
df_predict


# In[ ]:


# Preparing submission dataset
df_predict_submission = df_predict[['item_validation', 'predictions']]


# In[ ]:


submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')


# In[ ]:


for item in tqdm(df_predict.item_validation.unique(), total=len(df_predict.item_validation.unique()), desc='Preparing prediction dataset...'):
    
    item_predictions = df_predict.query(f'item_validation == "{item}"')['predictions']

    submission.loc[submission.id == item+'_validation', 'F1':'F28'] = item_predictions[:28].ravel()
    submission.loc[submission.id == item+'_evaluation', 'F1':'F28'] = item_predictions[28:].ravel()
    
    # Removing already proccessed predictions to optimize execution 
    df_predict = df_predict.iloc[28:]


# In[ ]:


submission.to_csv('submission.csv')  


# In[ ]:


submission

