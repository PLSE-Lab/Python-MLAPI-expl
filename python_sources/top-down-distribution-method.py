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


# 	1. Calculate weight for the level 12 series
# 	2. Use the naive logic to make forecasts for each of the level 12 series
# 	3. Infer forecast, ground truth values, and weights for all the higher level series by aggregating
# 	4. Calculalte RMSSE for all series using the equation
# 	5. Multiply weight by respective RMSSE and add all these products

# ## 0. Import libraries and read in data

# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm


# In[ ]:


df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')


# In[ ]:


price_df = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")


# In[ ]:


df.head()


# In[ ]:


price_df.head()


# In[ ]:


cal_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")


# In[ ]:


cal_df.head()


# In[ ]:


cal_df["d"]=cal_df["d"].apply(lambda x: int(x.split("_")[1]))
price_df["id"] = price_df["item_id"] + "_" + price_df["store_id"] + "_validation"


# ## 1. Calculate weight for the level 12 series

# In[ ]:


for day in tqdm(range(1858, 1886)):
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
id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
for col in id_cols:
    agg_df[col] = 'all'
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
    for c in column_order:
        if c not in temp_df.columns:
            temp_df[c] = 'all'
    agg_df = agg_df.append(temp_df[column_order])

del temp_df


# In[ ]:


print(df.shape[0], agg_df.shape[0], df.shape[0] + agg_df.shape[0])


# In[ ]:


agg_df["weight"].sum() + df["weight"].sum()


# In[ ]:


agg_df


# ## 3. Top down forecasts from different levels

# In[ ]:


df["last28_mean"] = df[[c for c in df.columns if c.find("d_")==0 and        int(c.split("_")[1]) in range(1858, 1886)] +       ["id"]].set_index("id").transpose().mean().reset_index()[0]


# In[ ]:


for l in tqdm(range(1, 10)):
    # 1. make forecast for this level
    this_level_df = agg_df[agg_df["level"]==l].copy()
    for d in range(1, 29):
        this_level_df["F_" + str(l) + "_" + str(1885 + d)] = this_level_df["d_" + str(1885 + d - 28)]
    
    # from this level, distribute forecast to all level 12 series
    # find columns with non 'all' entries
    important_column_ids = list(this_level_df[id_cols].columns[this_level_df[id_cols].nunique()!=1])
    this_level_df.reset_index(drop=True, inplace=True)
    for i, row in this_level_df.iterrows():
        if len(important_column_ids) == 0:
            level_mean_with_cond = this_level_df[[c for c in df.columns if c.find("d_")==0 and                                   int(c.split("_")[1]) in range(1858, 1886)]].transpose().mean()[0]
            proportion = df["last28_mean"] / level_mean_with_cond 
            for d in range(1, 29):
                df["F_" + str(l) + "_" + str(1885 + d)] = list(this_level_df["F_" + str(l) + "_" + str(1885 + d)])[0] * proportion
        else:
            cond = True
            for col in important_column_ids:
                cond = cond & (df[col] == row[col])
                
            level_mean_with_cond = this_level_df[[c for c in df.columns if c.find("d_")==0 and                                   int(c.split("_")[1]) in range(1858, 1886)]].transpose().mean()[i]
            proportion = df["last28_mean"] / level_mean_with_cond 
            for d in range(1, 29):
                df.loc[cond, "F_" + str(l) + "_" + str(1885 + d)] = row["F_" + str(l) + "_" + str(1885 + d)] * proportion
    
# remake agg_df
new_agg_df = pd.DataFrame(df[[c for c in df.columns if c.find("d_") == 0 or c.find("F_") == 0]].sum()).transpose()
id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
for col in id_cols:
    new_agg_df[col] = 'all'
new_agg_df["level"] = 1
new_agg_df["weight"] = 1/12
column_order = new_agg_df.columns

for level in level_groupings:
    temp_df = df.groupby(by=level_groupings[level]).sum().reset_index()
    temp_df["level"] = level
    for c in column_order:
        if c not in temp_df.columns:
            temp_df[c] = 'all'
    new_agg_df = new_agg_df.append(temp_df[column_order])
del temp_df


# In[ ]:


agg_df = new_agg_df


# ## 4. Calculalte RMSSE for all series using the equation

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
    return (1/h * numerator/denominator) ** 0.5


# In[ ]:


train_series_cols = [c for c in df.columns if c.find("d_") == 0][:-28]
ground_truth_cols = [c for c in df.columns if c.find("d_") == 0][-28:]

forecast_cols_dict = {}
for i in range(1, 10):
    forecast_cols_dict[i] = [c for c in df.columns if c.find("F_"+str(i)+"_") == 0]


# In[ ]:


for i in range(1, 10):
    df["rmsse_" + str(i)] = rmsse(np.array(df[ground_truth_cols]), 
        np.array(df[forecast_cols_dict[i]]), np.array(df[train_series_cols]))
    agg_df["rmsse_" + str(i)] = rmsse(np.array(agg_df[ground_truth_cols]), 
        np.array(agg_df[forecast_cols_dict[i]]), np.array(agg_df[train_series_cols]))


# In[ ]:


for i in range(1, 10):
    df["wrmsse_" + str(i)] = df["weight"] * df["rmsse_" + str(i)]
    agg_df["wrmsse_" + str(i)] = agg_df["weight"] * agg_df["rmsse_" + str(i)]


# In[ ]:


for i in range(1, 10):
    print("Aggregation by level", str(i) + ":")
    print(df["wrmsse_" + str(i)].sum() + agg_df["wrmsse_" + str(i)].sum())
    print()


# ###### Make submission file

# In[ ]:


df["last28_mean"] = df[[c for c in df.columns if c.find("d_")==0 and        int(c.split("_")[1]) in range(1886, 1914)] +       ["id"]].set_index("id").transpose().mean().reset_index()[0]


# In[ ]:


level_1_mean = agg_df[[c for c in df.columns if c.find("d_")==0 and        int(c.split("_")[1]) in range(1886, 1914)]].transpose().mean().reset_index()[0][0]


# In[ ]:


submit_df = df[["id"]]
for i in range(1, 29):
    proportion = df["last28_mean"] / level_1_mean
    submit_df["F" + str(i)] = agg_df[agg_df["level"]==1]["d_" + str(1885 + i)][0] * proportion


# In[ ]:


submit_df


# In[ ]:


submit_df2 = submit_df.copy()
submit_df2["id"] = submit_df2["id"].apply(lambda x: x.replace('validation',
                                                              'evaluation'))


# In[ ]:


submit_df = submit_df.append(submit_df2).reset_index(drop=True)


# In[ ]:


submit_df.to_csv("submission.csv", index=False)


# In[ ]:




