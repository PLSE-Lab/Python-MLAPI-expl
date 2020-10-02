#!/usr/bin/env python
# coding: utf-8

# # Project goal is forecasting item sales at stores in various locations for two 28-day time periods.
# 
# Datasets provided for the project are as follows:
#     
# calendar.csv - Contains information about the dates on which the products are sold.
# 
# sales_train_validation.csv - Contains the historical daily unit sales data per product and store [d_1 - d_1913]
# 
# sample_submission.csv - The correct format for submissions. Reference the Evaluation tab for more info.
# 
# sell_prices.csv - Contains information about the price of the products sold per store and date.
# 
# sales_train_evaluation.csv - NOT available, will be made available one month before competition deadline. Will include sales [d_1 - d_1941]
# 
# Data provided cover stores for three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details and has explanatory variables such as price, promotions, day of the week, and special events.

# In[ ]:


import numpy as np 
import pandas as pd


# In[ ]:


from  datetime import datetime, timedelta
import gc
import lightgbm as lgb


# In[ ]:


def reduce_mem_usage(df, verbose=True):
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


# In[ ]:


# historical daily unit sales data per product and store
stv = pd.read_csv("../input/m5-forecasting/sales_train_validation.csv")


# In[ ]:


stv.head()


# In[ ]:


# information about the price of the products sold per store and date
sp = pd.read_csv("../input/m5-forecasting/sell_prices.csv")


# In[ ]:


sp.head()


# In[ ]:


# calendar dates info
cal = pd.read_csv("../input/m5-forecasting/calendar.csv")


# In[ ]:


cal.head()


# In[ ]:


ss = pd.read_csv("../input/m5-forecasting/sample_submission.csv")
ss.head()


# In[ ]:


stv.info()


# In[ ]:


# drop NA values from 'sales_train_validation.csv'
stv.dropna(inplace = True)
stv.shape


# In[ ]:


CAL_DTYPES = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }


# In[ ]:


pd.options.display.max_columns = 50


# In[ ]:


cal.tail()


# In[ ]:


h = 28 
max_lags = 57
train_last = 1913
f_day = datetime(2016,4, 25) 
f_day


# # Formatting datasets

# In[ ]:


def create_stv(is_train = True, nrows = None, first_day = 1500):
    sales_price = pd.read_csv("../input/m5-forecasting/sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            sales_price[col] = sales_price[col].cat.codes.astype("int16")
            sales_price[col] -= sales_price[col].min()
            
    cal = pd.read_csv("../input/m5-forecasting/calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else train_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,train_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    stv = pd.read_csv("../input/m5-forecasting/sales_train_validation.csv", 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
    for col in catcols:
        if col != "id":
            stv[col] = stv[col].cat.codes.astype("int16")
            stv[col] -= stv[col].min()
    
    if not is_train:
        for day in range(train_last+1, train_last+ 28 +1):
            stv[f"d_{day}"] = np.nan
    
    stv = pd.melt(stv,
                  id_vars = catcols,
                  value_vars = [col for col in stv.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    stv = stv.merge(cal, on= "d", copy = False)
    stv = stv.merge(sales_price, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
    return stv


# In[ ]:


def create_features(stv):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        stv[lag_col] = stv[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            stv[f"rmean_{lag}_{win}"] = stv[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())

    
    
    date_features = {
        
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }
    
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in stv.columns:
            stv[date_feat_name] = stv[date_feat_name].astype("int16")
        else:
            stv[date_feat_name] = getattr(stv["date"].dt, date_feat_func).astype("int16")


# In[ ]:


FIRST_DAY = 500


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nstv = create_stv(is_train=True, first_day= FIRST_DAY)\nstv.shape')


# In[ ]:


stv.info()


# In[ ]:


stv.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ncreate_features(stv)\nstv.shape')


# In[ ]:


stv.info()


# In[ ]:


stv.head()


# In[ ]:


stv.dropna(inplace = True)
stv.shape


# In[ ]:


cat_features = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ['event_name_1', 'event_name_2', 'event_type_1', 'event_type_2']
useless_columns = ['id', 'date', 'sales','d', 'wm_yr_wk', 'weekday']
train_columns = stv.columns[~stv.columns.isin(useless_columns)]
X_train = stv[train_columns]
y_train = stv['sales']


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nnp.random.seed(1000)\n\nfake_valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace = False)\ntrain_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)\ntrain_data = lgb.Dataset(X_train.loc[train_inds], label = y_train.loc[train_inds], \n                         categorical_feature=cat_features, free_raw_data=False)\nfake_valid_data = lgb.Dataset(X_train.loc[train_inds], label = y_train.loc[train_inds],\n                              categorical_feature=cat_features,\n                 free_raw_data=False)# This is a random sample.')


# In[ ]:


#train_data.savebinary('train.bin')


# In[ ]:


del stv, X_train, y_train, fake_valid_inds,train_inds ; gc.collect()


# In[ ]:


# Defining parameters for the model
params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.01,
#         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
#         "nthread" : 4
        "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations' : 100,
    'num_leaves': 100,
    "min_data_in_leaf": 100,
}


# In[ ]:


get_ipython().run_cell_magic('time', '', '# We will use LightGBM model\n\n\nm_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=1)')


# <font color = "blue">As we can see the RMSE is about 3.4%. It can be further reduced by increasing the number of itertions.

# In[ ]:


m_lgb.save_model("model.lgb")


# In[ ]:


ss = pd.read_csv("../input/m5-forecasting/sample_submission.csv")


# In[ ]:


ss.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nalphas = [1.028, 1.023, 1.018]\nweights = [1/len(alphas)]*len(alphas)\nsub = 0.\n\nfor icount, (alpha, weight) in enumerate(zip(alphas, weights)):\n\n    te = create_stv(False)\n    cols = [f"F{i}" for i in range(1,29)]\n\n    for tdelta in range(0, 28):\n        day = f_day + timedelta(days=tdelta)\n        print(tdelta, day)\n        tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()\n        create_features(tst)\n        tst = tst.loc[tst.date == day , train_columns]\n        te.loc[te.date == day, "sales"] = alpha*m_lgb.predict(tst) # magic multiplier by kyakovlev\n\n\n\n    te_sub = te.loc[te.date >= f_day, ["id", "sales"]].copy()\n\n    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]\n    te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()\n    te_sub.fillna(0., inplace = True)\n    te_sub.sort_values("id", inplace = True)\n    te_sub.reset_index(drop=True, inplace = True)\n    te_sub.to_csv(f"submission_{icount}.csv",index=False)\n    if icount == 0 :\n        sub = te_sub\n        sub[cols] *= weight\n    else:\n        sub[cols] += te_sub[cols]*weight\n    print(icount, alpha, weight)\n\n\nsub2 = sub.copy()\nsub2["id"] = sub2["id"].str.replace("validation$", "evaluation")\nsub = pd.concat([sub, sub2], axis=0, sort=False)\nsub.to_csv("submission.csv",index=False)')


# In[ ]:


sub.head(10)


# In[ ]:


sub.id.nunique(), sub["id"].str.contains("validation$").sum()


# In[ ]:


sub.shape

