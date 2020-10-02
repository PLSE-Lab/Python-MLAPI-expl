#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## 0. Import libraries and read in data

# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm


# In[ ]:


df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
price_df = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
cal_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")


# In[ ]:


cal_df["d"]=cal_df["d"].apply(lambda x: int(x.split("_")[1]))
price_df["id"] = price_df["item_id"] + "_" + price_df["store_id"] + "_validation"


# In[ ]:


df.head()


# ## 1. Calculate weight for the level 12 series

# In[ ]:


for day in tqdm(range(1886, 1914)):
    wk_id = list(cal_df[cal_df["d"]==day]["wm_yr_wk"])[0]
    wk_price_df = price_df[price_df["wm_yr_wk"]==wk_id]
    df = df.merge(wk_price_df[["sell_price", "id"]], on=["id"], how='inner')
    df["unit_sales_" + str(day)] = df["sell_price"] * df["d_" + str(day)]
    df.drop(columns=["sell_price"], inplace=True)


# In[ ]:


df["dollar_sales"] = df[[c for c in df.columns if c.find("unit_sales")==0]].sum(axis=1)


# In[ ]:


df.drop(columns=[c for c in df.columns if c.find("unit_sales")==0], inplace=True)


# In[ ]:


df["weight"] = df["dollar_sales"] / df["dollar_sales"].sum()


# In[ ]:


df.drop(columns=["dollar_sales"], inplace=True)


# In[ ]:


df["weight"] /= 12


# ## 2. Transform time series

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


# In[ ]:


df.dtypes


# In[ ]:


print('Largest sales value is', df[[c for c in df.columns if c.find('d_')==0]].max().max(), 
      '\nLargest int16 is', np.iinfo(np.int16).max)


# In[ ]:


for d in range(1914, 1942):
    df["d_" + str(d)] = np.nan


# In[ ]:


temp = df.drop(columns = [c for c in df.columns if c.find('d_')==0 and int(c.split('_')[1]) < 1100])


# In[ ]:


df_melted = temp.melt(id_vars=[n for n in temp.columns if n.find("id")!=-1],
       value_vars=[n for n in temp.columns if n.find("d_")==0],
       var_name = 'day', value_name = 'sales')
del temp


# In[ ]:


df[['id', 'd_1100']].head(1)


# In[ ]:


df_melted.head()


# In[ ]:


df_melted["day"]=df_melted["day"].apply(lambda x: int(x.split("_")[1]))


# In[ ]:


df_melted=df_melted.merge(cal_df.drop(columns=["date", "wm_yr_wk", 
                                            "weekday"]), left_on=["day"], right_on=["d"]).drop(columns=["d"])


# In[ ]:


df_melted['event_name_1'].value_counts(dropna=False).head()


# In[ ]:


df_melted['event_name_1'].astype('category').cat.codes.astype("int8").value_counts().head()


# In[ ]:


df_melted["event_name_1"]=df_melted["event_name_1"].astype('category').cat.codes.astype("int8")
df_melted["event_name_2"]=df_melted["event_name_2"].astype('category').cat.codes.astype("int8")
df_melted["event_type_1"]=df_melted["event_type_1"].astype('category').cat.codes.astype("int8")
df_melted["event_type_2"]=df_melted["event_type_2"].astype('category').cat.codes.astype("int8")


# In[ ]:


useful_ids = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
id_encodings = [id_col + '_encoding' for id_col in useful_ids]

for id_col in useful_ids:
    if id_col == 'item_id':
        df_melted[id_col + '_encoding'] = df_melted[id_col].astype('category').cat.codes.astype("int16")
    else: 
        df_melted[id_col + '_encoding'] = df_melted[id_col].astype('category').cat.codes.astype("int8")


# In[ ]:


df_melted.drop(columns=['year'] + useful_ids, inplace=True)


# In[ ]:


reduce_mem_usage(df_melted)


# In[ ]:


df_melted['test'] = df_melted[["id","sales"]].groupby("id")["sales"].shift(1).fillna(-1).astype(np.int16)
print(list(df_melted[df_melted['day']==1101]['test']) == list(df_melted[df_melted['day']==1100]['sales']))


# In[ ]:


df_melted.drop(columns=['test'], inplace=True)


# In[ ]:


# create lags, lags starts from 28 days ago to 77 days ago, spaced by 7 days
for lag in tqdm([28, 35, 42, 49, 56, 63, 70, 77]):
    df_melted["lag_" + str(lag)] = df_melted[["id","sales"]].groupby("id")["sales"].shift(lag).fillna(-1).astype(np.int16)


# In[ ]:


df_melted=df_melted[df_melted['lag_77']!=-1]


# In[ ]:


assert list(df_melted[df_melted['day']==1528]['lag_28']) == list(df_melted[df_melted['day']==1500]['sales'])


# In[ ]:


df_melted.head()


# In[ ]:


price_df.head()


# In[ ]:


df_melted=df_melted.merge(cal_df[['d', 'wm_yr_wk']], left_on=['day'], right_on=['d']).drop(columns=['d'])


# In[ ]:


df_melted=df_melted.merge(price_df[['id', 'sell_price', 'wm_yr_wk']], on=['id', 'wm_yr_wk'], how='inner')


# In[ ]:


del price_df
del cal_df


# In[ ]:


import lightgbm as lgb


# In[ ]:


best_params = {
            "objective" : "poisson",
            "metric" :"rmse",
            "force_row_wise" : True,
            "learning_rate" : 0.05,
    #         "sub_feature" : 0.8,
            "sub_row" : 0.75,
            "bagging_freq" : 1,
            "lambda_l2" : 0.1,
    #         "nthread" : 4
            "metric": ["rmse"],
        'verbosity': 1,
        'num_iterations' : 2048,
        'num_leaves': 64,
        "min_data_in_leaf": 50,
    }


# In[ ]:


del wk_price_df


# In[ ]:


X_train = df_melted[df_melted["day"] < 1886].drop(columns=["sales"])
X_val = df_melted[df_melted["day"].between(1886, 1913)].drop(columns=["sales"])
X_test = df_melted[df_melted["day"] > 1913].drop(columns=["sales"])

y_train = df_melted[df_melted["day"] < 1886]["sales"]
y_val = df_melted[df_melted["day"].between(1886, 1913)]["sales"]


# In[ ]:


get_ipython().run_line_magic('who', 'DataFrame')


# In[ ]:


del df_melted


# In[ ]:


X_train.columns


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nnp.random.seed(777)\n\nfake_valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace = False)\ntrain_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)\ntrain_data = lgb.Dataset(X_train.drop(columns=['id']).loc[train_inds] , label = y_train.loc[train_inds], \n                         categorical_feature=id_encodings, free_raw_data=False)\nfake_valid_data = lgb.Dataset(X_train.drop(columns=['id']).loc[fake_valid_inds], label = y_train.loc[fake_valid_inds],\n                              categorical_feature=id_encodings,\n                 free_raw_data=False)# This is a random sample, we're not gonna apply any time series train-test-split tricks here!")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nm_lgb = lgb.train(best_params, train_data, valid_sets = [fake_valid_data], verbose_eval=1) ')


# In[ ]:


for d in range(1886, 1914):
    df['F_' + str(d)] = m_lgb.predict(X_val[X_val['day']==d].drop(columns=['id']))


# In[ ]:


df.head()


# In[ ]:


h = 28
n = 1885
def rmsse(ground_truth, forecast, train_series, axis=1):
    # assuming input are numpy array or matrices
    assert axis == 0 or axis == 1
    assert type(ground_truth) == np.ndarray and type(forecast) == np.ndarray and type(train_series) == np.ndarray
    
    if axis == 1:
        # using axis == 1 we must guarantee these are matrices and not arrays
        assert ground_truth.shape[1] > 1 and forecast.shape[1] > 1 and train_series.shape[1] > 1
    
    numerator = ((ground_truth - forecast)**2).sum(axis=axis)
    if axis == 1:
        denominator = 1/(n-1) * ((train_series[:, 1:] - train_series[:, :-1]) ** 2).sum(axis=axis)
    else:
        denominator = 1/(n-1) * ((train_series[1:] - train_series[:-1]) ** 2).sum(axis=axis)
    if (numerator < 0).any():
        print('nu')
    elif (denominator < 0).any():
        print(denominator[denominator < 0])
    return (1/h * numerator/denominator) ** 0.5


# In[ ]:


level_groupings = {2: ["state_id"], 3: ["store_id"], 4: ["cat_id"], 5: ["dept_id"], 
              6: ["state_id", "cat_id"], 7: ["state_id", "dept_id"], 8: ["store_id", "cat_id"], 9: ["store_id", "dept_id"],
              10: ["item_id"], 11: ["item_id", "state_id"]}


# In[ ]:


#remake agg_df
new_agg_df = pd.DataFrame(df[[c for c in df.columns if c.find("d_") == 0 or c.find("F_") == 0]].sum()).transpose()
new_agg_df["level"] = 1
new_agg_df["weight"] = 1/12
column_order = new_agg_df.columns

for level in level_groupings:
    temp_df = df.groupby(by=level_groupings[level]).sum().reset_index()
    temp_df["level"] = level
    new_agg_df = new_agg_df.append(temp_df[column_order])
del temp_df

agg_df = new_agg_df

train_series_cols = [c for c in df.columns if c.find("d_") == 0 and int(c.split('_')[1]) < 1886]
ground_truth_cols = [c for c in df.columns if c.find("d_") == 0 and int(c.split('_')[1]) in range(1886, 1914)]
forecast_cols = [c for c in df.columns if c.find("F_") == 0]

df["rmsse"] = rmsse(np.array(df[ground_truth_cols]), 
        np.array(df[forecast_cols]), np.array(df[train_series_cols]))
agg_df["rmsse"] = rmsse(np.array(agg_df[ground_truth_cols]), 
        np.array(agg_df[forecast_cols]), np.array(agg_df[train_series_cols]))

df["wrmsse"] = df["weight"] * df["rmsse"]
agg_df["wrmsse"] = agg_df["weight"] * agg_df["rmsse"]

print(df["wrmsse"].sum() + agg_df["wrmsse"].sum())


# In[ ]:


lgb.plot_importance(m_lgb)


# ###### Make submission file

# In[ ]:


submit_df = df[["id"]]
for i in range(1, 29):
    submit_df["F" + str(i)] = m_lgb.predict(X_test[X_test['day']==i+1913].drop(columns=['id']))


# In[ ]:


submit_df.head()


# In[ ]:


submit_df2 = submit_df.copy()
submit_df2["id"] = submit_df2["id"].apply(lambda x: x.replace('validation',
                                                              'evaluation'))


# In[ ]:


submit_df = submit_df.append(submit_df2).reset_index(drop=True)


# In[ ]:


submit_df.to_csv("submission.csv", index=False)


# In[ ]:




