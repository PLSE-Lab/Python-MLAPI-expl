#!/usr/bin/env python
# coding: utf-8

# * Notebook reads in sample of data due to memory limits
# * Training will be MUCH faster with GPU.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

from random import randint
from catboost import CatBoostClassifier
from catboost import Pool, cv
from catboost import CatBoost, Pool, MetricVisualizer

from pprint import pprint
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GroupKFold, GroupShuffleSplit
from pprint import pprint
import shap
from catboost import cv
shap.initjs()
import zipfile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, cross_validate
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report

from copy import deepcopy


pd.options.display.float_format = '{:,.3f}'.format

import os


# In[ ]:


get_ipython().system('unzip /kaggle/input/expedia-personalized-sort/data.zip ')
### ZipFile can't read this proprietary format 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().system('wc -l train.csv')
## 9.9 million rows of train data


# In[ ]:


get_ipython().system('wc -l test.csv')
# 6.6 Million rows


# In[ ]:


def get_target(row):
    """
    0=not clicked at all, 1=clicked but not booked, 5=booked
    """
    if row.booking_bool>0:
        return 1
    if row.click_bool>0 :
        return 0.2
    return 0


def featurize_df(df:pd.DataFrame) ->pd.DataFrame:
    """
    Extract more features
    """
    df["weekday"] = df["date_time"].dt.weekday
    df["week_of_year"] = df["date_time"].dt.week

    df["hour"] = df["date_time"].dt.hour
    df["minute"] = df["date_time"].dt.minute
    ## total time elapsed - allows model to learn continous trend over time to a degree
    df["time_epoch"] = df["date_time"].astype('int64')//1e9
    ## if we were looking at fraud: df["seconds"] = df.timestamp.dt.second
    df["early_night"] = ((df["hour"]>19) | (df["hour"]<3)) # no added value from feature
    
    df["nans_count"] = df.isna().sum(axis=1)
    
    ## we won't make any time series features for now
    ## We could add time series features per property/hotel. We'd need to check for unaries, and to add a shift/offset dependant on forecast horizon

    ## get relative rank of price within group/query (i.e order by  relative price - most expensive)
    df["price_rank"] = df.groupby("srch_id")["price_usd"].rank("dense", ascending=False)
    
    return df


# In[ ]:


# data = zipfile.ZipFile("/kaggle/input/expedia-personalized-sort/data.zip") #  zipped file.
# df = pd.read_csv(data.open('train.csv'))
df = pd.read_csv('train.csv',parse_dates=["date_time"],infer_datetime_format=True,
#                  nrows=3123456
                ) # memory error when reading all data in
print(df.shape)
df.tail()


# In[ ]:


print(list(df.columns))
display(df.nunique())
df.head()


# In[ ]:


# df = df.dropna(how="all")
# float_cols = df.columns[df.dtypes.eq('float')]# float_cols = df.columns.drop(['date_time'])
# for c in float_cols:
#     df[c] = pd.to_numeric(df[c], errors="ignore",downcast="integer")  # parse columns back to integer

print(df.shape)

df["target"] = df.apply(get_target,axis=1)
# featurization must be after leaky cols are dropped, otherwise the nan feature will bea leak!
display(df.describe())
display(df.nunique())

display(df["date_time"].describe())
display(df)

df


# #### test_csv is huge - we'll need to predict in batches

# In[ ]:


# # df_test = pd.read_csv(data.open('test.csv'))
# df_test = pd.read_csv('test.csv',parse_dates=["date_time"],infer_datetime_format=True)
# print(df_test.shape)
# # cols = df_test.columns.drop(['date_time'])

# # float_cols = df_test.columns[df_test.dtypes.eq('float')]# float_cols = df.columns.drop(['date_time'])
# # for c in float_cols:
# #     df_test[c] = pd.to_numeric(df_test[c], errors="ignore",downcast="integer") 


# df_test


# ### EDA & drop bad cols

# In[ ]:


df.drop_duplicates(['click_bool','booking_bool','random_bool'])


# In[ ]:


drop_cols = []

## we see many columns are unary - drop them, barring feature engineering
drop_unary_cols = [c for c
             in list(df)
             if df[c].nunique(dropna=False) <= 1]
target_cols = ["gross_bookings_usd","click_bool","booking_bool"] # leaky column, and original target columns
drop_cols.extend(drop_unary_cols)
drop_cols.extend(target_cols) 

### we'll need to remove datetime from the model, but it may be useful for train/test split before that
# drop_cols.append("date_time")

df = df.drop(columns=drop_cols,errors="ignore")
df_test = df_test.drop(columns=drop_cols,errors="ignore")
print(df.shape)
df


# #### Add features

# In[ ]:


df = featurize_df(df)
df_test = featurize_df(df_test)


# ### Naive feature importance - by rank / interest
# * We could also do by target class (booked vs clicked vs 0), +- p-values

# In[ ]:


## sort by high rank, regardless of booked or not (for easy comp)
df.drop(['comp3_rate',
       'comp3_inv', 'comp3_rate_percent_diff', 'comp4_inv', 'comp5_rate',
       'comp5_inv', 'comp5_rate_percent_diff', 'comp8_rate', 'comp8_inv',
       'comp8_rate_percent_diff'],axis=1).groupby(df["target"]>0).mean()


# #### train /test ("eval") split
# Split by time, and groupwise (by queries). Depends what makes more sense.. Given very small time period covered, learning a model on a time split may cause us to lose out on features relevant to the testing data in this cases
# 
# We can also just use catboost built in Cross Validation (which supports groupwise splits), since I'm not doing any real hyperparameter tuning at this stage.
# 
# We'll split by the lasy ~10% of queries, i.e by group and time for evaluation.

# In[ ]:


cutoff_id = df["srch_id"].quantile(0.94) # 90/10 split
X_train = df.loc[df.srch_id< cutoff_id].drop(["target"],axis=1)
X_eval = df.loc[df.srch_id>= cutoff_id].drop(["target"],axis=1)
y_train = df.loc[df.srch_id< cutoff_id]["target"]
y_eval = df.loc[df.srch_id>= cutoff_id]["target"]


# In[ ]:


print("mean relevancy train",round(y_train.mean(),4))
print("mean relevancy eval",round(y_eval.mean(),4))
print(y_eval.value_counts()) # check we have all 3 "labels" in subset


# In[ ]:


df["target"].value_counts()


# ### Train (Ranking) models
# * explain top features using SHAP
# * Build a classification/AUC model then a ranking model (best fit for data)

# In[ ]:


categorical_cols = ['prop_id',"srch_destination_id", "weekday"] # ,"week_of_year"


# In[ ]:


df.tail()


# In[ ]:


## check for feature/column leaks
set(X_train.columns).symmetric_difference(set(df_test.columns))


# In[ ]:


train_pool = Pool(data=X_train,
                  label = y_train,
                  cat_features=categorical_cols,
                  group_id=X_train["srch_id"]
                 )

eval_pool = Pool(data=X_eval,
                  label = y_eval,
                  cat_features=categorical_cols,
                  group_id=X_eval["srch_id"]
                 )


# In[ ]:


default_parameters  = {
    'iterations': 2000,
    'custom_metric': ['NDCG', "AUC:type=Ranking"], # , 'AverageGain:top=3'# 'QueryRMSE', "YetiLoss" (use with hints)
    'verbose': False,
    'random_seed': 42,
#     "task_type":"GPU",
    "has_time":True,
    "metric_period":5,
    "save_snapshot":False,
    "use_best_model":True, # requires eval set to be set
} 

parameters = {}

def fit_model(loss_function, additional_params=None, train_pool=train_pool, test_pool=eval_pool):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function
    
    if additional_params is not None:
        parameters.update(additional_params)
        
    model = CatBoost(parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)
    print("best results (train on train):")
    print(model.get_best_score()["learn"])
    print("best results (on validation set):")
    print(model.get_best_score()["validation"])
    
    print("(Default) Feature importance (on train pool)")
    display(model.get_feature_importance(data=train_pool,prettified=True).head(15))
    
    try:
        print("SHAP features importance, on all data:")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(pd.concat([X_train,X_eval]),
                                            y=pd.concat([y_train,y_eval]))

        # # summarize the effects of all the features
        shap.summary_plot(shap_values, pd.concat([X_train,X_eval]))
    finally:
        return model


# In[ ]:


## we can try fitting with other losses, but this one worked best for me
# model_PairLogit = fit_model('PairLogit')

model = fit_model('PairLogitPairwise')


# #### Get predictions on "evaluation" data & export
# 
# * For each query, a maximum of 38 hotels may be returned 
# * We use the  "best model" based on those evaluated, and refit it on all data (we could also stick to trained model instead): `PairLogitPairwise`  - based on the NCDG [(and ranking AUC)](https://catboost.ai/docs/concepts/loss-functions-ranking.html) score.

# In[ ]:


# df_test["target"] = model.predict(df_test)
# display(df_test[["srch_id","date_time","target"]])
# df_test[["srch_id","date_time","target"]].to_csv("test_predictions.csv.gz",compression="gzip")

