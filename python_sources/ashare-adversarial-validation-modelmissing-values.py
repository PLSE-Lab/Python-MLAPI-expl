#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import datetime,random
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from scipy.stats import skew, boxcox, mstats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Summary

# 1st aim:
# 
# Find and select training examples most similar to test examples and using them as a validation set. The core of this idea is training a probabilistic classifier to distinguish train/test examples.
# 
# Conclusion:
# 
# From the results of visualize_train_test_feature_distribution we see that there is a good classification based on building_id and site_id. Thus we can select the ids from peaks for validation sets.
# 
# 2nd aim:
# 
# Correlation analysis to find strong linear dependencies.
# Conclusion:
# 
# building_id and site_id have a strong linear dependency. I guess we can eliminate site_id from the models. Same for air_temperature dew_temperature (in addition they have a similar % of missing values) and floor_count and squre_feet.
# 
# 3rd aim:
# 
# Find ways to recover missing values.
# 
# Conclusion:
# 
# floor_count - 76 % of missing values. It can be well recovered by square_feet and year_built
# 
# year_built - 47 % of missing values. Can be well recovered by square_feet and year_built and primary_use
# 
# precip_depth_1_hr - 17 % Can be well recovered by hour air temperature and sea level
# 
# cloud coverage - Can be well discovered by day, hour, temperature
# 
# We can buld models for missing values imputation before training instead of using simple imputer

# ## Setup Environment

# In[ ]:


SEED = 42
FOLDS = 2

def seed_env(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

pd.set_option('display.max_columns', 380)
pd.set_option('display.max_rows', 500)
seed_env()

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
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

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


def download_train_dataframe(csv_sources, full_download = True):
    if full_download:
        train = reduce_mem_usage(pd.read_csv(csv_sources[0]))
    else:
        train = reduce_mem_usage(pd.read_csv(csv_sources[0],nrows = 10000000))
        
    building_metadata_df = reduce_mem_usage(pd.read_csv(csv_sources[1]))
    weather_df = reduce_mem_usage((pd.read_csv(csv_sources[2])))
    
    train = train.merge(building_metadata_df, on="building_id", how="left")
    train.drop("meter_reading", axis=1, inplace=True)
    train = train.merge(weather_df, on=["site_id", "timestamp"], how="left")
    train.reset_index()
    
    del building_metadata_df,weather_df
    gc.collect()

    return train

def download_test_dataframe(csv_sources):
    test = reduce_mem_usage(pd.read_csv(csv_sources[0],skiprows = 800, nrows = 20000000,names = ['row_id','building_id','meter','timestamp']))
    building_metadata_df = reduce_mem_usage(pd.read_csv(csv_sources[1]))
    weather_df = reduce_mem_usage((pd.read_csv(csv_sources[2])))
    
    test = test.merge(building_metadata_df, on="building_id", how="left")
    test.drop("row_id", axis=1, inplace=True)
    test = test.merge(weather_df, on=["site_id", "timestamp"], how="left")
    test.reset_index()
    
    del building_metadata_df,weather_df
    gc.collect()

    return test

def dataframe_num_feature_summary(df):
    summary = pd.DataFrame(df.dtypes,columns = ["dtypes"])
    summary = summary.reset_index()
    summary["Feature"] = summary["index"]
    summary = summary[["Feature","dtypes"]]
    summary["Missed"] = df.isna().sum().values
    df_num_describe = df.describe()
    summary["Count"] = df_num_describe.iloc[0,:].values
    summary["% Missed"] = (summary["Missed"] / df.shape[0]) * 100
    summary["% Missed"] = round(summary["% Missed"], 2)
    summary["Mean"] = df_num_describe.iloc[1,:].values
    summary["Std"] = df_num_describe.iloc[2,:].values
    summary["Min"] = df_num_describe.iloc[3,:].values
    summary["Max"] = df_num_describe.iloc[7,:].values
    summary["25 %"] = df_num_describe.iloc[4,:].values
    summary["75 %"] = df_num_describe.iloc[6,:].values
    
    return summary


# In[ ]:


def visualize_feature_importance(model,columns):
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importance(),columns), reverse=True), columns=["Value","Feature"])
    plt.figure(figsize=(10,10))
    importance_bar = sns.barplot(data=feature_imp, x='Value', y='Feature')
    plt.show()


# In[ ]:


csv_sources = ['../input/ashrae-energy-prediction/train.csv', '../input/ashrae-energy-prediction/building_metadata.csv', '../input/ashrae-energy-prediction/weather_train.csv']
train_df = download_train_dataframe(csv_sources)
print(train_df.shape)

csv_sources = ['../input/ashrae-energy-prediction/test.csv', '../input/ashrae-energy-prediction/building_metadata.csv', '../input/ashrae-energy-prediction/weather_test.csv']
test_df = download_test_dataframe(csv_sources)
print(test_df.shape)


# In[ ]:


train_df["SELECTION"] = 1
test_df["SELECTION"] = 0
train_test_df = pd.concat([train_df,test_df])
train_test_df["SELECTION"] = train_test_df["SELECTION"].astype(np.uint8)
train_test_df.reset_index()
del train_df,test_df
gc.collect()


# ## Adversarial validation

# In[ ]:


train_test_df["primary_use"] = train_test_df["primary_use"].astype(str)
le = LabelEncoder()
train_test_df["primary_use"] = le.fit_transform(train_test_df["primary_use"])
    
train_test_df['timestamp'] = pd.to_datetime(train_test_df['timestamp'], format="%Y-%m-%d %H:%M:%S")
train_test_df['month'] = train_test_df['timestamp'].dt.month.astype(np.int8)
train_test_df['day'] = train_test_df['timestamp'].dt.day.astype(np.int8)
train_test_df['hour'] = train_test_df['timestamp'].dt.hour.astype(np.int8)
train_test_df['weekday'] = train_test_df['timestamp'].dt.weekday.astype(np.int8)
train_test_df.drop('timestamp',axis=1,inplace=True)


# In[ ]:


train_test_df.head()


# In[ ]:


train_test_target = train_test_df["SELECTION"]
train_test_df.drop('SELECTION',axis=1,inplace=True)
categorical_features = ["meter", "primary_use", "cloud_coverage","building_id", "month", "day", "hour","weekday","site_id"]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'params = {\n          "objective" : "binary",\n          "metric" : "auc",\n          "boosting_type": "gbdt",\n          "num_leaves" : 11,\n          "learning_rate" : 0.05,\n          "feature_fraction": 0.85,\n          "random_state" : SEED}\n\ntrain_X, val_X, train_y, val_y = train_test_split(train_test_df, train_test_target, test_size=0.3, random_state=SEED, shuffle = True)\n        \nlgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categorical_features)\nlgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categorical_features)\n\ndel train_X, val_X, train_y, val_y\ngc.collect()\n\nlgbmodel = lgb.train(params, lgb_train, num_boost_round=60, valid_sets=[lgb_train, lgb_eval], early_stopping_rounds=60, verbose_eval=60)')


# In[ ]:


visualize_feature_importance(lgbmodel, train_test_df.columns)


# In[ ]:


def visualize_train_test_feature_distribution(df):
    train = df.iloc[:20216100]
    test = df.iloc[20216100:]
    print(train.shape)
    print(test.shape)
    fig,axis = plt.subplots(3, 3, figsize=(12, 16))
    
    sns.distplot(train["building_id"], ax=axis[0,0], color = 'blue')
    sns.distplot(test["building_id"], ax=axis[0,0], color = 'red')
    
    sns.distplot(train["site_id"], ax=axis[0,1], color = 'blue')
    sns.distplot(test["site_id"], ax=axis[0,1], color = 'red')
    
    sns.distplot(train["day"], ax=axis[0,2], color = 'blue')
    sns.distplot(test["day"], ax=axis[0,2], color = 'red')
    
    sns.distplot(train["month"], ax=axis[1,0], color = 'blue')
    sns.distplot(test["month"], ax=axis[1,0], color = 'red')
    
    sns.distplot(train["air_temperature"].dropna(), ax=axis[1,1], color = 'blue')
    sns.distplot(test["air_temperature"].dropna(), ax=axis[1,1], color = 'red')
    
    sns.distplot(train["sea_level_pressure"].dropna(), ax=axis[1,2], color = 'blue')
    sns.distplot(test["sea_level_pressure"].dropna(), ax=axis[1,2], color = 'red')
    
    sns.distplot(train["year_built"].dropna(), ax=axis[2,0], color = 'blue')
    sns.distplot(test["year_built"].dropna(), ax=axis[2,0], color = 'red')
    
    sns.distplot(train["square_feet"].dropna(), ax=axis[2,1], color = 'blue')
    sns.distplot(test["square_feet"].dropna(), ax=axis[2,1], color = 'red')
    
    sns.distplot(train["primary_use"].dropna(), ax=axis[2,2], color = 'blue')
    sns.distplot(test["primary_use"].dropna(), ax=axis[2,2], color = 'red')
    
    plt.show()
    del train, test
    gc.collect()


# In[ ]:


visualize_train_test_feature_distribution(train_test_df)


# In[ ]:


del train_test_target,lgbmodel,lgb_train,lgb_eval
gc.collect()


# ## Correlation Analysis

# In[ ]:


def display_corr_map(df,cols):
    plt.figure(figsize=(16,16))
    sns.heatmap(df[cols].corr(), cmap='RdBu_r', annot=True, center=0.0)
    plt.title(cols[0]+' - '+cols[-1],fontsize=14)
    plt.show()


# In[ ]:


cols_to_analyze = ["square_feet","year_built", "floor_count","air_temperature", "dew_temperature","precip_depth_1_hr","sea_level_pressure","wind_direction","wind_speed","meter", "primary_use", "cloud_coverage","building_id","site_id","day","hour","weekday"]
display_corr_map(train_test_df,cols_to_analyze)


# In[ ]:


dataframe_num_feature_summary(train_test_df.drop(["primary_use"], axis=1, inplace = False))


# In[ ]:


train_test_df["primary_use"] = train_test_df["primary_use"].astype(str)
print(train_test_df["primary_use"].describe())
print("% Missed = {0}".format((train_test_df["primary_use"].isna().sum() / train_test_df["primary_use"].shape[0]) * 100))


# In[ ]:


del train_test_df
gc.collect()


# In[ ]:


csv_sources = ['../input/ashrae-energy-prediction/train.csv', '../input/ashrae-energy-prediction/building_metadata.csv', '../input/ashrae-energy-prediction/weather_train.csv']
train_df = download_train_dataframe(csv_sources, full_download = False)
print(train_df.shape)


# In[ ]:


train_df.drop("building_id",axis=1,inplace=True)
train_df.drop("site_id",axis=1,inplace=True)
train_df["primary_use"] = train_df["primary_use"].astype(str)
le = LabelEncoder()
train_df["primary_use"] = le.fit_transform(train_df["primary_use"])
train_df["primary_use"] = train_df["primary_use"].astype(np.int8)

train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], format="%Y-%m-%d %H:%M:%S")
train_df['month'] = train_df['timestamp'].dt.month.astype(np.int8)
train_df['day'] = train_df['timestamp'].dt.day.astype(np.int8)
train_df['hour'] = train_df['timestamp'].dt.hour.astype(np.int8)
train_df['weekday'] = train_df['timestamp'].dt.weekday.astype(np.int8)
train_df.drop('timestamp',axis=1,inplace=True)

del le
gc.collect()


# In[ ]:


def adversarial_data(df, feature):
    adv_data = df.dropna(subset=[feature])
    target = adv_data[feature].copy()
    adv_data = adv_data.drop(feature, axis=1)
    return adv_data, target


# In[ ]:


def train_adv_models(data,target):
    params = {
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 40,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "reg_lambda": 2,
    "metric": "rmse"}
    
    kf = KFold(n_splits = FOLDS, shuffle=True, random_state=SEED)
    models = list()
    
    for train_idx, valid_idx in kf.split(data, target):
        train_X = data.iloc[train_idx]
        val_X = data.iloc[valid_idx]
        train_y = target.iloc[train_idx]
        val_y = target.iloc[valid_idx]
        
        lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categorical_features)
        lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categorical_features)
        
        del train_X,val_X,train_y,val_y
        gc.collect()
        
        model = lgb.train(params,
                lgb_train,
                num_boost_round=200,
                valid_sets=(lgb_train, lgb_eval),
                early_stopping_rounds=200,
                verbose_eval = 200)
        
        models.append(model)
        del lgb_train, lgb_eval
        gc.collect()
    
    return models


# First analyze with the biggest missing values

# In[ ]:


def plot_results(target,prediction,models,columns):
    plt.subplots(2, 2, figsize=(10, 10))
    ax1 = plt.subplot(2,2,(1,2))
    sns.distplot(target, ax=ax1, color = 'blue')
    sns.distplot(prediction, ax=ax1, color = 'red')
    ax2 = plt.subplot(223)
    feature_imp1 = pd.DataFrame(sorted(zip(models[0].feature_importance(),columns), reverse=True), columns=["Value","Feature"])
    importance_bar1 = sns.barplot(data=feature_imp1, x='Value', y='Feature', ax=ax2)
    ax3 = plt.subplot(224)
    feature_imp2 = pd.DataFrame(sorted(zip(models[1].feature_importance(),columns), reverse=True), columns=["Value","Feature"])
    importance_bar2 = sns.barplot(data=feature_imp2, x='Value', y='Feature', ax=ax3)
    plt.show()
    del feature_imp1, feature_imp2
    gc.collect()


# ## floor_count

# In[ ]:


get_ipython().run_cell_magic('time', '', 'categorical_features = ["meter", "primary_use", "cloud_coverage","month", "day", "hour","weekday"]\nadv_data, target = adversarial_data(train_df, \'floor_count\')\nmodels = train_adv_models(adv_data, target)\npredictions = sum([model.predict(adv_data, num_iteration=model.best_iteration) for model in models])/FOLDS\nplot_results(target,predictions,models,adv_data.columns)\ndel target,predictions,models,adv_data\ngc.collect()')


# ## year_built

# In[ ]:


adv_data, target = adversarial_data(train_df,"year_built")
models = train_adv_models(adv_data, target)
predictions = sum([model.predict(adv_data, num_iteration=model.best_iteration) for model in models])/FOLDS
plot_results(target,predictions,models,adv_data.columns)
del target,predictions,models,adv_data
gc.collect()


# ## precip_depth_1_hr

# In[ ]:


adv_data, target = adversarial_data(train_df,"precip_depth_1_hr")
models = train_adv_models(adv_data, target)
predictions = sum([model.predict(adv_data, num_iteration=model.best_iteration) for model in models])/FOLDS
plot_results(target,predictions,models,adv_data.columns)
del target,predictions,models,adv_data
gc.collect()


# ## cloud_coverage

# In[ ]:


adv_data, target = adversarial_data(train_df, 'cloud_coverage')
categorical_features = ["meter", "primary_use", "month", "day", "hour","weekday"]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'params = {\n          "objective" : "multiclass",\n          "num_class" : 10,\n          "num_leaves" : 40,\n          "learning_rate" : 0.05,\n          "feature_fraction": 0.85,\n          "bagging_seed" : SEED}\n\ntrain_X, val_X, train_y, val_y = train_test_split(adv_data, target, test_size=0.3, random_state=SEED)\n        \nlgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categorical_features)\nlgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categorical_features)\n\nlgbmodel = lgb.train(params, lgb_train, num_boost_round=200, valid_sets=[lgb_train, lgb_eval], early_stopping_rounds=200, verbose_eval=200)')


# In[ ]:


del lgb_train,lgb_eval,train_X,val_X,train_y, val_y,train_df
gc.collect()


# In[ ]:


predictions = lgbmodel.predict(adv_data, num_iteration=lgbmodel.best_iteration)


# In[ ]:


sns.distplot(target, color = "blue")
sns.distplot(np.argmax(predictions, axis=1), color = "red")


# In[ ]:


feature_imp1 = pd.DataFrame(sorted(zip(lgbmodel.feature_importance(),adv_data.columns), reverse=True), columns=["Value","Feature"])
sns.barplot(data=feature_imp1, x='Value', y='Feature')

