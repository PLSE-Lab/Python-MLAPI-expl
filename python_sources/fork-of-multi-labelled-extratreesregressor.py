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


from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
price_df = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
cal_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")


# In[ ]:


cal_df["d"]=cal_df["d"].apply(lambda x: int(x.split("_")[1]))
price_df["id"] = price_df["item_id"] + "_" + price_df["store_id"] + "_validation"


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


# ## 2. Infer round truth values, and weights for all the higher level series by aggregating

# In[ ]:


agg_df = pd.DataFrame(df[[c for c in df.columns if c.find("d_") == 0]].sum()).transpose()
agg_df["level"] = 1
agg_df["weight"] = 1/12
column_order = agg_df.columns


# In[ ]:


level_groupings = {2: ["state_id"], 3: ["store_id"], 4: ["cat_id"], 5: ["dept_id"], 
              6: ["state_id", "cat_id"], 7: ["state_id", "dept_id"], 8: ["store_id", "cat_id"], 9: ["store_id", "dept_id"],
              10: ["item_id"], 11: ["item_id", "state_id"]}


# In[ ]:


for level in tqdm(level_groupings):
    temp_df = df.groupby(by=level_groupings[level]).sum().reset_index()
    temp_df["level"] = level
    agg_df = agg_df.append(temp_df[column_order])

del temp_df


# In[ ]:


print(df.shape[0], agg_df.shape[0], df.shape[0] + agg_df.shape[0])


# In[ ]:


agg_df["weight"].sum() + df["weight"].sum()


# ## 3. Multi Label Regression with ExtraTreesRegressor

# In[ ]:


h = 28
def rmsse(ground_truth, forecast, train_series, axis=1, n=1885):
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
    return (1/h * numerator/denominator) ** 0.5


# In[ ]:


pd.get_dummies(df.drop(columns=["id", "item_id", "weight"]))


# In[ ]:


df.head()


# In[ ]:


df = df[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "weight"]].join(pd.get_dummies(df.drop(columns=["id", "item_id", "weight"])))


# In[ ]:


import random


# In[ ]:


best_s = 100
best_m = None
best_start_date = 1000


# In[ ]:


1885 - 28+28


# In[ ]:


for _ in tqdm(range(50)):
    rand_est = random.randint(20, 50)
    rand_depth = random.randint(10, 30)
    rand_start_date = random.randint(1200, 1500)
    
    print(rand_est, rand_depth)
    
    
    average = []
    
    for cv in range(1, 4):
        train_start = rand_start_date - 28 * cv
        train_end = 1885 - 28 * cv
        
        regressor = ExtraTreesRegressor(n_estimators=rand_est, max_depth=rand_depth, random_state=42)
        
        drop_cols = [item for item in [c for c in df.columns if c.find("F_")==0] + ['wrmsse', 'rmsse'] if item in df.columns]
        df.drop(columns=drop_cols, inplace=True)

        regressor.fit(df.drop(columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] +                              [c for c in df.columns if c.find("d_")==0 and int(c.split("_")[1]) not in range(train_start, train_end + 1)]),
              df[[c for c in df.columns if c.find("d_")==0 and int(c.split("_")[1]) in range(train_end + 1, train_end + 28 + 1)]])

        pred_df = pd.DataFrame(regressor.predict(df.drop(columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] +                                       [c for c in df.columns if c.find("d_")==0 and int(c.split("_")[1]) not in range(train_start+28, train_end + 28 + 1)])))
        pred_df.columns = ["F_" + str(d) for d in range(train_end + 28 + 1, train_end + 28 + 28 + 1)]
        df = df.join(pred_df)

        # remake agg_df
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
        
        train_series_cols = [c for c in df.columns if c.find("d_") == 0][:-28]
        ground_truth_cols = [c for c in df.columns if c.find("d_") == 0][-28:]
        forecast_cols = [c for c in df.columns if c.find("F_") == 0]

        df["rmsse"] = rmsse(np.array(df[ground_truth_cols]), 
                np.array(df[forecast_cols]), np.array(df[train_series_cols]))
        agg_df["rmsse"] = rmsse(np.array(agg_df[ground_truth_cols]), 
                np.array(agg_df[forecast_cols]), np.array(agg_df[train_series_cols]))

        df["wrmsse"] = df["weight"] * df["rmsse"]
        agg_df["wrmsse"] = agg_df["weight"] * agg_df["rmsse"]

        print("CV", cv, ":", df["wrmsse"].sum() + agg_df["wrmsse"].sum())

        average.append(df["wrmsse"].sum() + agg_df["wrmsse"].sum())
    
    this_s = np.array(average).mean()
    if this_s < best_s:
        best_s = this_s
        best_m = regressor
        best_start_date = rand_start_date
        
    print(this_s, best_s)


# In[ ]:


# fit the best_m with the closest training set
best_m.fit(df.drop(columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] +                              [c for c in df.columns if c.find("d_")==0 and int(c.split("_")[1]) not in range(best_start_date, 1886)]),
              df[[c for c in df.columns if c.find("d_")==0 and int(c.split("_")[1]) in range(1886, 1914)]])


# ###### Make submission file

# In[ ]:


submit_df = df[["id"]]
pred = best_m.predict(df.drop(columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] +
                               [c for c in df.columns if c.find("d_")==0 and int(c.split("_")[1]) not in range(best_start_date+28, 1914)]))
for i in range(1, 29):
    submit_df["F" + str(i)] = pred[:, i-1]


# In[ ]:


submit_df2 = submit_df.copy()
submit_df2["id"] = submit_df2["id"].apply(lambda x: x.replace('validation',
                                                              'evaluation'))


# In[ ]:


submit_df = submit_df.append(submit_df2).reset_index(drop=True)


# In[ ]:


submit_df.to_csv("submission.csv", index=False)


# In[ ]:


submit_df


# In[ ]:




