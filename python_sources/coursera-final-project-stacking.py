#!/usr/bin/env python
# coding: utf-8

# <h1>Part 1: Pre processing</h1>
# 
# Load the libraries and data

# In[ ]:


import gc

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm_notebook
from itertools import product

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate

from statsmodels.tsa.stattools import pacf
from category_encoders import TargetEncoder
from sklearn.ensemble import StackingRegressor,RandomForestRegressor
import xgboost as xgb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
cats = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
train.head()


# In[ ]:


#Save memory
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


# In[ ]:


#Remove outliers
train = train[(train.item_price < 100000 )& (train.item_cnt_day < 1000)]

#Remove negative values
train = train[train.item_price > 0].reset_index(drop = True)
train.loc[train.item_cnt_day < 1, "item_cnt_day"] = 0

#Remove equal shops
train.loc[train.shop_id == 0, "shop_id"] = 57
test.loc[test.shop_id == 0 , "shop_id"] = 57
train.loc[train.shop_id == 1, "shop_id"] = 58
test.loc[test.shop_id == 1 , "shop_id"] = 58
train.loc[train.shop_id == 11, "shop_id"] = 10
test.loc[test.shop_id == 11, "shop_id"] = 10
train.loc[train.shop_id == 40, "shop_id"] = 39
test.loc[test.shop_id == 40, "shop_id"] = 39

X= train.copy(deep=True)
id_column = test['ID']
X_test = test.drop('ID',axis=1).copy(deep=True)
X_test["date_block_num"] = 34

del train, test
gc.collect()


# In[ ]:


#Create a combined dataframe

df = pd.concat([X, X_test], sort=False)
df.fillna(0, inplace = True )

#Delete the unused data
del X
del X_test
gc.collect()


# In[ ]:


# Create "grid" with columns
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = [] 
for block_num in df['date_block_num'].unique():
    cur_shops = df.loc[df['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = df.loc[df['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

# Turn the grid into a dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

# Groupby data to get shop-item-month aggregates
gb = df.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})
# Fix column names
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values] 
# Join it to the grid
all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)

# Same as above but with shop-month aggregates
gb = df.groupby(['shop_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_shop':'sum'}})
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
all_data = pd.merge(all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)

# Same as above but with item-month aggregates
gb = df.groupby(['item_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_item':'sum'}})
gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)

# Downcast dtypes from 64 to 32 bit to save memory
all_data = downcast_dtypes(all_data)

del grid, gb 
gc.collect()


# In[ ]:


all_data


# In[ ]:


#Calculate partial autocorrelations

x_pacf = pacf(all_data['target_item'], nlags =5, method= 'ols' )
x_pacf1 = pacf(all_data['target'], nlags =5, method= 'ols' )
print(x_pacf)
print(x_pacf1)


# In[ ]:


plt.plot(x_pacf1)


# In[ ]:


# List of columns that we will use to create lags
cols_to_rename = list(all_data.columns.difference(index_cols)) 

shift_range = [1, 2, 3, 4]

for month_shift in tqdm_notebook(shift_range):
    train_shift = all_data[index_cols + cols_to_rename].copy()
    
    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
    
    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
    train_shift = train_shift.rename(columns=foo)

    all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)

del train_shift

# Don't use old data from year 2013
all_data = all_data[all_data['date_block_num'] >= 12] 

# List of all lagged features
fit_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range]] 
# We will drop these at fitting stage
to_drop_cols = list(set(list(all_data.columns)) - (set(fit_cols)|set(index_cols))) + ['date_block_num'] 

# Category for each item
item_category_mapping = items[['item_id','item_category_id']].drop_duplicates()

all_data = pd.merge(all_data, item_category_mapping, how='left', on='item_id')
df = downcast_dtypes(all_data)

del all_data
gc.collect();


# In[ ]:


#Returning the data
X = df[df['date_block_num'] <= 33].drop('date_block_num', axis=1)
Y = X['target']
X = X.drop(['target'], axis=1).clip(0,20)
X_test = df[df['date_block_num'] == 34 ].drop(['date_block_num','target'], axis=1).clip(0,20)

del df
gc.collect();


# In[ ]:


#Mean encoding
columns = ['shop_id','item_id', 'item_category_id', 'target_shop_lag_1','target_shop_lag_2','target_shop_lag_3',
           'target_shop_lag_4', 'target_item_lag_1' ,'target_item_lag_2' ,'target_item_lag_3', 'target_item_lag_4']

targ_enc = TargetEncoder(cols=columns).fit(X, Y)

X = targ_enc.transform(X.reset_index(drop=True))
X_test = targ_enc.transform(X_test.reset_index(drop=True))


# # Part 2: Modelling and predicting

# In[ ]:


ss = MinMaxScaler()
X = ss.fit_transform(X)
ss2 = MinMaxScaler()
X_test = ss2.fit_transform(X_test)

ss1 = MinMaxScaler()
Y = ss1.fit_transform(Y.values.reshape(-1, 1))
Y = Y.reshape(Y.shape[0])

del shops
del items
del cats
gc.collect()


# In[ ]:


model1 = CatBoostRegressor(iterations=5, depth=10,# learning_rate=0.01, 
                           loss_function='RMSE', verbose=0, thread_count=-1)# task_type="GPU"

model2 = RandomForestRegressor(n_estimators = 15, max_depth = 10, n_jobs=-1)

model3 = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.01, eval_metric='rmse',
                max_depth = 10, n_estimators = 10, tree_method = 'exact')


# In[ ]:


training, valid, ytraining, yvalid = train_test_split(X,Y, test_size=0.3)

del X
del Y
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nmodel = StackingRegressor(estimators=[('cb',model1), ('xg', model3)], final_estimator=model2, cv=3)\nmodel.fit(training,ytraining)")


# In[ ]:


preds = model.predict(valid)
print(np.sqrt(mean_squared_error(preds, yvalid)))
XX=X_test.copy()
predictions = model.predict(XX)


# # 3 Predict and save

# In[ ]:


predictions = ss1.inverse_transform(predictions.reshape(-1,1)).clip(0,20)
predictions = predictions.reshape(predictions.shape[0])
df = pd.DataFrame({'ID': id_column, 'item_cnt_month': predictions})
df.to_csv('coursera_new.csv', index=False)


# In[ ]:


df


# In[ ]:


df.sort_values(by="item_cnt_month",ascending=False).head(20)


# In[ ]:


from joblib import dump, load
dump(model, 'coursera_model.joblib')

