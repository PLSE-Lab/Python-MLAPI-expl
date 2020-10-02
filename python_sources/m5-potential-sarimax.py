#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for Dirname, _, Filenames in os.walk('/kaggle/input'):
    for Filename in Filenames:
        print(os.path.join(Dirname, Filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Memory Check
# https://www.kaggle.com/kyakovlev/m5-simple-fe/output

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import gc, time, pickle, psutil, random

import pandas as pd
import numpy as np
import dask.dataframe as dd
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 300)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from math import ceil
from sklearn.preprocessing import LabelEncoder


# In[ ]:


## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','k','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Y', suffix)


# In[ ]:


## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
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


# # Load and Display Data

# In[ ]:


train_sales = pd.read_csv( Dirname + '/sales_train_evaluation.csv')
sell_prices = pd.read_csv( Dirname + '/sell_prices.csv')
calendar = pd.read_csv( Dirname + '/calendar.csv')
submission_file = pd.read_csv( Dirname + '/sample_submission.csv')


# In[ ]:


print('train_sales.shape: ', train_sales.shape)
print('sell_prices.shape: ', sell_prices.shape)
print('calendar.shape: ', calendar.shape)
print('submission_file.shape: ', submission_file.shape)


# In[ ]:


d_cols = [c for c in train_sales.columns if 'd_' in c] # sales data columns

ItemID = 'FOODS_3_202_CA_3_evaluation'
train_sales.loc[train_sales['id'] == ItemID].set_index('id')[d_cols].T.plot(figsize=(15,5), title=(ItemID + ' sales by "d" number'))
plt.legend('')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
stores = []
ItemID = 'FOODS_3_202'
for store, d in sell_prices.query('item_id == "FOODS_3_202"').groupby('store_id'):
    d.plot(x='wm_yr_wk', y='sell_price', style='.', title=(ItemID + ' sale price over time'), ax=ax, legend=store)
    stores.append(store)
    plt.legend()
plt.legend(stores)
plt.show()


# # Data Transformation and Feature Generation

# In[ ]:


calendar = reduce_mem_usage(calendar)
print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))
sell_prices = reduce_mem_usage(sell_prices)
print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))


# In[ ]:


# number of items, and number of prediction period
NUM_ITEMS = train_sales.shape[0]  # 30490
END_TRAIN  = 1941
DAYS_PRED = 28


# In[ ]:


def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        le = LabelEncoder()
        #not_null = df[col][df[col].notnull()]
        df[col] = df[col].fillna('nan')
        df[col] = pd.Series(le.fit_transform(df[col]), index=df.index)

    return df


calendar = encode_categorical(
    calendar, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
).pipe(reduce_mem_usage)

train_sales_val = encode_categorical(
    train_sales, ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
).pipe(reduce_mem_usage)

sell_prices = encode_categorical(sell_prices, ["item_id", "store_id"]).pipe(
    reduce_mem_usage
)


# In[ ]:


# drop dupulication
product = train_sales_val[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()


# In[ ]:


#nrows = 365 * 2 * NUM_ITEMS
nrows = END_TRAIN * NUM_ITEMS


# In[ ]:


# Before processing
display(train_sales_val.head(5))


# In[ ]:



d_name = ['d_' + str(i+1) for i in range(END_TRAIN)]
train_sales_val_values = train_sales_val[d_name].values

# calculate the start position(first non-zero demand observed date) for each item 
tmp = np.tile(np.arange(1,END_TRAIN+1),(train_sales_val_values.shape[0],1))
df_tmp = ((train_sales_val_values>0) * tmp)

start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)-1

flag = np.dot(np.diag(1/(start_no+1)) , tmp)<1

train_sales_val_values = np.where(flag,np.nan,train_sales_val_values)

train_sales_val[d_name] = train_sales_val_values

del tmp,train_sales_val_values
gc.collect()


# In[ ]:


train_sales_val = pd.melt(train_sales_val, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                          var_name = 'day', value_name = 'demand')


# In[ ]:


# After Processing
display(train_sales_val.head(5))
print('Melted sales train validation has {} rows and {} columns'.format(train_sales_val.shape[0],train_sales_val.shape[1]))


# In[ ]:


train_sales_val = train_sales_val.iloc[-nrows:,:]
train_sales_val = train_sales_val[~train_sales_val.demand.isnull()]


# In[ ]:


# seperate test dataframes

test1_rows = [row for row in submission_file['id'] if 'validation' in row]
test2_rows = [row for row in submission_file['id'] if 'evaluation' in row]

test1 = submission_file[submission_file['id'].isin(test1_rows)]
test2 = submission_file[submission_file['id'].isin(test2_rows)]

test1.columns = ["id"] + [f"d_{d}" for d in range( END_TRAIN+1-DAYS_PRED, END_TRAIN+1 )]
test2.columns = ["id"] + [f"d_{d}" for d in range( END_TRAIN+1, END_TRAIN+1+ DAYS_PRED)]

#test1['id'] = test1['id'].str.replace('_validation','')
#test2['id'] = test2['id'].str.replace('_evaluation','_validation')

test1 = test1.merge(product, how = 'left', on = 'id')
test2 = test2.merge(product, how = 'left', on = 'id')

test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                var_name = 'day', value_name = 'demand')

test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                var_name = 'day', value_name = 'demand')

train_sales_val['part'] = 'train'
test1['part'] = 'test1'
test2['part'] = 'test2'

data = pd.concat([train_sales_val, test1, test2], axis = 0)

del train_sales_val, test1, test2

#data = data[data['part'] != 'test2']

gc.collect()


# In[ ]:


#calendar merge
# drop some calendar features 
calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)

# notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)
data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
data.drop(['d', 'day'], inplace = True, axis = 1)

# memory release
del  calendar
gc.collect()

#sell price merge
# get the sell price data (this feature should be very important)
data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))

# memory release
del  sell_prices
gc.collect()


# In[ ]:


data.head(5)


# In[ ]:


data.shape


# # SARIMAX Forecast Test

# In[ ]:


ItemID = 'FOODS_3_090_CA_3_evaluation'
#ItemID = 'HOBBIES_2_142_TX_2_evaluation'
itemX = data[data['id'] == ItemID]
itemX.head()


# In[ ]:


itemX.shape


# In[ ]:


itemX.plot(x='date', y='demand', figsize=(15,5))
plt.show()


# In[ ]:


X_train = itemX.iloc[:,10:17].astype(np.float32)
y_train = itemX['demand'].astype(np.float32)
X_date = itemX['date']


# In[ ]:


X_train.head()


# In[ ]:


X_train.shape


# ## Find optimal parameter

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Parameter Optimization\nfrom sklearn import metrics\nimport statsmodels.api as sm\nfrom statsmodels.tsa.statespace.sarimax import SARIMAX\nimport itertools as itr\n\np = [1]\nq = [0, 1]\nd = [0, 1]\npdq = list(itr.product(p, d, q))\nsp = [0, 1]\nsq = [0, 1]\nsd = [0, 1]\ns = [7]\nseasonal_pdq = list(itr.product(sp, sd, sq,s))\npattern = len(seasonal_pdq) * len(pdq)\n\nwarnings.filterwarnings(\'ignore\')\n\nmodelSelection = pd.DataFrame(index=range(pattern), columns=["model", "aic",\'rmse\'])\nnum = 0\nfor param in pdq:\n    for param_seasonal in seasonal_pdq:\n        mdl = SARIMAX(endog=y_train, exog=X_train,\n                                   order=param, seasonal_order=param_seasonal, measurement_error = True,\n                                   enforce_stationarity = False, enforce_invertibility = False).fit()\n        modelSelection["model"][num] = "order=(" + str(param) + "), season=("+ str(param_seasonal) + ")"\n        modelSelection["aic"][num] = mdl.aic\n        \n        pred = mdl.predict(start=1, end=len(X_train), exog=X_train.iloc[0,:]) \n        pred[pred < 0] = 0\n        modelSelection["rmse"][num] = np.sqrt(metrics.mean_squared_error(y_train, pred))\n        \n        num = num + 1\n        \n# RMSE Minimum\nmodelSelection[modelSelection.rmse == min(modelSelection.rmse)]')


# In[ ]:


get_ipython().run_cell_magic('time', '', "warnings.filterwarnings('ignore')\n\nmdl = SARIMAX(endog=y_train, exog=X_train,\n                            order=(1,1,0), seasonal_order=(1,0,0,7), measurement_error = True,\n                            enforce_stationarity=False,enforce_invertibility=False)\nresults = mdl.fit()\nresults.plot_diagnostics(figsize=(12,8))\nprint(results.summary())")


# ## Prediction

# In[ ]:


#evaluation test
def evaluation_test(actual, predict, title):
    from sklearn import metrics
    from scipy.stats import pearsonr
    rmse = np.sqrt(metrics.mean_squared_error(actual, predict))
    corr, p = pearsonr(actual, predict)
    print(title, '; RMSE is ', rmse, ',  Correlation is ', corr)

y_pred = results.predict(start=1, end=len(X_train), exog=X_train.iloc[0,:]) 
y_pred[y_pred < 0] = 0
evaluation_test(y_train, y_pred, ItemID)


# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(X_date, y_train, label='Actual')
plt.plot(X_date, y_pred, label='Predict')
plt.legend()
plt.show()


# In[ ]:


y_pred.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Iteration Test\nid_list = data['id'].unique()\nNid = len(id_list)\nNtest = 10\nprediction = np.zeros([Ntest, END_TRAIN+DAYS_PRED])\n\nfor i in range(0,Ntest):\n    ItemID = id_list[i]\n    itemX = data[data['id'] == ItemID]\n    X_train = itemX.iloc[:,10:17].astype(np.float32)\n    y_train = itemX['demand'].astype(np.float32)\n    mdl = SARIMAX(endog=y_train, exog=X_train,\n                                order=(1,1,0), seasonal_order=(1,0,0,7), measurement_error = True,\n                                enforce_stationarity = False, enforce_invertibility = False)\n    results = mdl.fit()\n    y_pred = results.predict(start=1, end=len(X_train), exog=X_train.iloc[0,:]) \n    y_pred[y_pred < 0] = 0\n    prediction[i, :] = y_pred")


# # Conclusion

# SARIMAX shows great prediction with the salling time-series, but it takes long time since the regression takes one by one.
# It is expected to take over 100 hours in the kaggle kernel.
