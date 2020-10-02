#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test_df = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])


# In[ ]:


train_df.head()


# In[ ]:


train_df["feature_1"].value_counts()


# In[ ]:


train_df["feature_2"].value_counts()


# In[ ]:


train_df["feature_3"].value_counts()


# So seems like feature 1,2 and 3 are cateogrical.

# In[ ]:


sns.set_style("ticks")
plt.figure(figsize=[6,4])
sns.distplot(train_df["target"])
plt.show()


# In[ ]:


train_df.isna().sum()


# In[ ]:


test_df.isna().sum()


# In[ ]:


train_df.shape


# In[ ]:


train_df.dtypes


# In[ ]:


# from old fastai old
# https://github.com/fastai/fastai/blob/master/old/fastai/structured.py#L76
import re
def add_datepart(df, fldname, drop=False, time=False):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)


# In[ ]:


add_datepart(train_df, "first_active_month")
add_datepart(test_df, "first_active_month")


# In[ ]:


train_df['elapsed_time'] = (datetime.date(2018, 2, 1) - train_df['first_active_month'].dt.date).dt.days
test_df['elapsed_time'] = (datetime.date(2018, 2, 1) - test_df['first_active_month'].dt.date).dt.days


# In[ ]:


print("Train Data Time Range:",train_df["first_active_month"].min(), "-", train_df["first_active_month"].max())
print("Test Data Time Range:",test_df["first_active_month"].min(), "-",  test_df["first_active_month"].max())


# In[ ]:


train_df.groupby("first_active_month")["target"].mean().plot()


# In[ ]:


set((train_df["first_active_month"])).intersection(set(test_df["first_active_month"]))


# So there are some intersection between train and test data.

# In[ ]:


train_card_id = train_df["card_id"]
test_card_id = test_df["card_id"]


# Sort data by date and get last 10% for validation split

# In[ ]:


train_df.sort_values("first_active_month", inplace=True)


# In[ ]:


train_df.reset_index(drop=True, inplace=True)


# In[ ]:


train_columns = [c for c in train_df.columns if c not in ["first_active_month", "card_id", "target"]]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(train_df[train_columns], train_df["target"], test_size = 0.2, random_state=1001)


# In[ ]:


(X_train.shape, y_train.shape), (X_val.shape, y_val.shape)


# In[ ]:


from sklearn.metrics import mean_squared_error
import lightgbm as lgb


# In[ ]:


params = {
"objective" : "regression",
"metric" : "rmse",
"num_leaves" : 12,
"learning_rate" : 0.01,
"bagging_fraction" : 0.7,
"feature_fraction" : 0.9,
"bagging_frequency" : 4,
"bagging_seed" : 1001,
"verbosity" : -1
}


# In[ ]:


cat_cols = ["feature_1", "feature_2", "feature_3"]


# In[ ]:


dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, free_raw_data=False)


# In[ ]:


evals_result = {}
model_lgb = lgb.train(params, dtrain, 500, valid_sets=[dtrain,dval], valid_names=["train", "val"], early_stopping_rounds=30, verbose_eval=100, evals_result=evals_result)


# In[ ]:


lgb.plot_importance(model_lgb, figsize=(12,10))


# In[ ]:


lgb.plot_importance(model_lgb, figsize=(12,10), importance_type="gain")


# In[ ]:


lgb.plot_metric(evals_result, "rmse", figsize=(12,10))


# In[ ]:


pred_lgb = model_lgb.predict(test_df[train_columns])


# In[ ]:


ss = pd.DataFrame({"card_id":test_card_id, "target":pred_lgb})
ss.to_csv("preds_starter_lgb.csv", index=None)
ss.head()


# #### xgboost

# In[ ]:


import xgboost as xgb


# In[ ]:


dtrain_xgb = xgb.DMatrix(X_train, label=y_train)
dval_xgb = xgb.DMatrix(X_val, y_val)


# In[ ]:


xgb_params = {
        'objective': 'reg:linear',
        'learning_rate': 0.02,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.9,
        'colsample_bylevel': 0.9,
        'n_jobs': -1,
        "silent": 1
    }


# In[ ]:


evallist  = [(dtrain_xgb, 'train'), (dval_xgb, "val")]


# In[ ]:


model_xgb = xgb.train(xgb_params, dtrain_xgb, num_boost_round=500, evals=evallist, early_stopping_rounds=50, verbose_eval=100)


# In[ ]:


xgb.plot_importance(model_xgb)


# In[ ]:


preds_xgb = model_xgb.predict(xgb.DMatrix(test_df[train_columns]))


# In[ ]:


ss = pd.DataFrame({"card_id":test_card_id, "target":preds_xgb})
ss.to_csv("preds_starter_xgb.csv", index=None)
ss.head()


# #### Catboost

# In[ ]:


from catboost import CatBoostRegressor


# In[ ]:


model_cat = CatBoostRegressor(iterations=500,
                             learning_rate=0.02,
                             depth=6,
                             eval_metric='RMSE',
                             bagging_temperature = 0.9,
                             od_type='Iter',
                             metric_period = 100,
                             od_wait=50)


# In[ ]:


model_cat.fit(X_train, y_train,
             eval_set=(X_val,y_val),
             cat_features=np.array([0,1,2]),
             use_best_model=True,
             verbose=100)


# In[ ]:


preds_cat = model_cat.predict(test_df[train_columns])  


# In[ ]:


pd.DataFrame(model_cat.get_feature_importance(), index=X_train[train_columns].columns, columns=["FeatureImportance"]).sort_values("FeatureImportance", ascending=False).plot(kind="barh", legend=False, figsize=(12,10))


# In[ ]:


ss = pd.DataFrame({"card_id":test_card_id, "target":preds_cat})
ss.to_csv("preds_starter_cat.csv", index=None)
ss.head()


# ### Blending

# In[ ]:


ss_blend = pd.DataFrame({"card_id":test_card_id, "target":((0.33 * preds_xgb) + (0.34 * pred_lgb)+ (0.33 * preds_cat))})
ss_blend.to_csv("preds_starter_blend.csv", index=None)
ss_blend.head()


# ### To be continued...
