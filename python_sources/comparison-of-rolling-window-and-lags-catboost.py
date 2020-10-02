#!/usr/bin/env python
# coding: utf-8

# ## Goal
# 
# There are multiple solutions here used rolling mean and lags as features and lots of them turned out pretty useful. So what is the optimal rolling mean and lags in this case?
# 
# Here I used catboost to build a basic model and analyzed it by 
# 
# * Feature Importance
# * SHAP value
# * Model analysis plot
# 
# (Note: to avoid memory error, a few parameters and data scope has been modified. )
# 
# 
# ## Referenced Notebook 

# **Most of the code on feature engineering are from this notebook by @kkiller: **
# https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50
# 
# Also referenced: 
# 
# 
# https://www.kaggle.com/mayer79/m5-forecast-poisson-loss
# 
# 
# https://towardsdatascience.com/deep-dive-into-catboost-functionalities-for-model-interpretation-7cdef669aeed
# 

# In[ ]:


from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
from catboost import Pool, CatBoostRegressor
#import lightgbm as lgb


# In[ ]:


import pandas as pd
import shap


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.style.use('fivethirtyeight')


# In[ ]:


pd.options.display.max_columns = 50


# In[ ]:


CAL_DTYPES={"weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16"}
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }


# In[ ]:


h = 28 
max_lags = 57
tr_last = 1913
fday = datetime(2016,4, 25) 
fday


# In[ ]:


def create_dt(is_train = True, nrows = None, first_day = 1200):
    prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv", 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
    return dt


# In[ ]:


## modified from the original
def create_fea(dt, lags, windows):
    print("creating window: ", windows, " lag: ", lags)
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    for win in windows :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())


    ## drop lag col
    dt.drop(lag_cols, axis=1, inplace=True)
    
    
    date_features = {
        
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
#         "ime": "is_month_end",
#         "ims": "is_month_start",
    }
    
#     dt.drop(["d", "wm_yr_wk", "weekday"], axis=1, inplace = True)
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")


# In[ ]:


#FIRST_DAY = 350 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk !
# I used 350 as the first day 
# To avoid maxing out the memory
FIRST_DAY = 1000


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndf = create_dt(is_train=True, first_day= FIRST_DAY)\ndf.columns')


# In[ ]:


df.info()


# In[ ]:


df.drop(['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI'], axis=1,       inplace=True)


# In[ ]:


gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'create_fea(df, lags=[7,14, 28, 30, 50, 60], windows=[7, 28])\ndf.shape')


# In[ ]:


df.info()


# In[ ]:


# X_train.corr()


# ## Split train and validation

# In[ ]:


df.dropna(inplace = True)
df.shape


# In[ ]:


cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id']
useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
train_cols = df.columns[~df.columns.isin(useless_cols)]
X_train = df[train_cols]
y_train = df["sales"]


# In[ ]:


X_train.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nnp.random.seed(777)\n\nfake_valid_inds = np.random.choice(X_train.index.values, int(X_train.shape[0]*.2), replace = False)\ntrain_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)')


# ## (Auto)correlation between features

# In[ ]:


fake_valid_inds_2 = np.random.choice(X_train.index.values, int(X_train.shape[0]*.002), replace = False)


# In[ ]:


import seaborn as sns


# In[ ]:


plt.figure(figsize=(10, 10))
sns.heatmap(X_train.loc[fake_valid_inds_2].corr())


# In[ ]:


gc.collect()


# ## Catboost

# In[ ]:


## load model
from catboost import Pool, CatBoostRegressor

from_file = CatBoostRegressor()
model = from_file.load_model('../input/catboost-baseline/catboost_model_0430')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n####### I uploaded the model to avoid memory error\ncat_feats_ind = [i for (i, j) in enumerate(X_train.columns) if j in cat_feats]\n\n# # put in pool\ntrain_pool = Pool(X_train.loc[train_inds], y_train.loc[train_inds], cat_features=cat_feats_ind)\ntest_pool = Pool(X_train.loc[fake_valid_inds], y_train.loc[fake_valid_inds], cat_features=cat_feats_ind)\n\n# # train\n# # I used iteration = 80 and depths = 3 at first\n# model =CatBoostRegressor(iterations=80, \\\n#                       depth=3, \\\n#                       learning_rate=1, \\\n#                       loss_function='RMSE',l2_leaf_reg=1)\n\n\n###############################################################\n\n\n#model.fit(train_pool, eval_set=test_pool, verbose=False)\n\n# plot\nplt.style.use('fivethirtyeight')\nplt.figure(figsize=(5,10))\nplt.barh(train_pool.get_feature_names(), model.get_feature_importance())\nplt.xticks(rotation=90)\nplt.show()")


# In[ ]:


gc.collect()


# In[ ]:


feature_df = pd.DataFrame({'Feature': X_train.columns, "Imp": model.get_feature_importance()})


# In[ ]:


feature_df.sort_values('Imp', ascending=False)


# ## Prediction

# In[ ]:


cat_feats_ind = [i for (i, j) in enumerate(X_train.columns) if j in cat_feats]


# In[ ]:


cat_feats_ind


# In[ ]:


gc.collect()


# In[ ]:


def create_lag_features_for_test(dt, day, lags, windows):
    # modified the original 
    # create lag feaures just for single day (faster)
    #lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        dt.loc[dt.date == day, lag_col] =             dt.loc[dt.date ==day-timedelta(days=lag), 'sales'].values  # !!! main

    #windows = [7, 28]
    for window in windows:
        for lag in lags:
            df_window = dt[(dt.date <= day-timedelta(days=lag)) & (dt.date > day-timedelta(days=lag+window))]
            df_window_grouped = df_window.groupby("id").agg({'sales':'mean'}).reindex(dt.loc[dt.date==day,'id'])
            dt.loc[dt.date == day,f"rmean_{lag}_{window}"] =                 df_window_grouped.sales.values     


# In[ ]:


def create_date_features_for_test(dt):
    # copy of the code from `create_dt()` above
    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(
                dt["date"].dt, date_feat_func).astype("int16")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nalphas = [1.028, 1.023, 1.018]\nweights = [1/len(alphas)]*len(alphas)  # equal weights\n\nte0 = create_dt(False)  # create master copy of `te`\ncreate_date_features_for_test (te0)')


# In[ ]:


model = model
for icount, (alpha, weight) in enumerate(zip(alphas, weights)):
    te = te0.copy()  # just copy
    cols = [f"F{i}" for i in range(1, 29)]

    for tdelta in range(0, 28):
        day = fday + timedelta(days=tdelta)
        print(tdelta, day.date())
        tst = te[(te.date >= day - timedelta(days=max_lags))
                 & (te.date <= day)].copy()
#         create_fea(tst)  # correct, but takes much time
        create_lag_features_for_test(tst, day, lags=[7], windows=[7])  # faster  
        tst = tst.loc[tst.date == day, train_cols]
        
        ## put tst in pool
        tst_pool = Pool(data=tst, cat_features=cat_feats_ind)
        te.loc[te.date == day, "sales"] =             alpha * model.predict(tst_pool)  # magic multiplier by kyakovlev

    te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()

    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")[
        "id"].cumcount()+1]
    te_sub = te_sub.set_index(["id", "F"]).unstack()[
        "sales"][cols].reset_index()
    te_sub.fillna(0., inplace=True)
    te_sub.sort_values("id", inplace=True)
    te_sub.reset_index(drop=True, inplace=True)
    te_sub.to_csv(f"submission_{icount}.csv", index=False)
    if icount == 0:
        sub = te_sub
        sub[cols] *= weight
    else:
        sub[cols] += te_sub[cols]*weight
    
sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv("submission_catboost_add_iteration_04_27.csv",index=False)
#print(icount, alpha, weight)


# In[ ]:


del sub,sub2


# In[ ]:





# ## Shap values
# 
# Shape values explain how far each feature 'pushes' the target values to larger or smaller values.
# 
# 
# See [this link](https://github.com/slundberg/shap) for details.

# In[ ]:


gc.collect()


# In[ ]:


shap_values = model.get_feature_importance(tst_pool,type="ShapValues")
expected_value = shap_values[0,-1]
shap_values = shap_values[:,:-1]


# In[ ]:





# In[ ]:


# we used one of the test data for shap value calculation
shap.initjs()
shap.force_plot(expected_value, shap_values[3,:], tst.iloc[3,:])


# In[ ]:


get_ipython().run_cell_magic('time', '', "shap.summary_plot(shap_values, tst, plot_type='bar')")


# You can see the SHAP value has a different interpretation than the feature importance.

# ## Model Analysis Plots
# 
# Catboost has a special way of plotting features to help you understand **how dispersed each feature has and the discrepancy between predicted value and true value**.
# 
# See this [link](https://catboost.ai/docs/concepts/python-reference_catboostregressor_calc_feature_statistics.html) for detail explanation.

# In[ ]:


feature = ['rmean_7_7', 'rmean_7_28', 'rmean_60_7', 'rmean_50_7', 'rmean_50_28',           'rmean_14_7','rmean_14_28', 'rmean_28_7']


# In[ ]:


X_train.shape


# In[ ]:


model.calc_feature_statistics(X_train[23940000:], y_train[23940000:],feature=feature, plot=True)


# ## Summary

# * Feature Importance: rmean_7_28 > rmean_7_7 > rmean_60_7 > rmean_50_7 > rmean_50_28
# * SHAP value: rmean_7_28 > rmean_7_7 > rmean_14_28 > rmean_28_7 > rmean_14_7
# * model plot analysis: rmean_7_28 > rmean_7_7 > rmean_14_28 > rmean_50_28 > rmean_14_7

# In[ ]:





# In[ ]:





# In[ ]:




