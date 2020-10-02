#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
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
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def merge_calendar(df, col):
    return pd.merge(df, calendar[[col+'_enc',col]] , how = 'left', on = col + '_enc')
def merge_id(df, col):
    return pd.merge(df, id_df[[col+'_enc',col+'_id']] , how = 'left', on = col + '_enc')


# Hi guys, here i am gonna fix the data into traditional timeseries format with day_num/ all features for each day_num/ sales. i've merged all features into the data set so it should be easy for you guys to do EDA and stuff. I was afraid there would not be enough memory so I only transformed the data of the last 810-ish days, but it turned out my resulting dataframe with 27.8m rows and 19 feature columns is taking up only 3.3 GB of RAM, so you could transform the whole dataset if you want, it'd take 9 GB at most.
# 
# I wrote a merge function for you to merge basic groupby dataframes and be able to look at the string features, since i had to label encoded all of the object features in the matrix to minimize memory consumption.you can write your own merge functions or merge by yourself of course.
# 
# reduce_mem_usage is of course not mine, it's a popular reduce_mem_usage function but i'm too lay to put the source... sorry guys... will do later
# Have fun guys

# In[ ]:


#train = reduce_mem_usage(train)
#train = train.drop(['d_%s' %(i) for i in range(1,1101)], axis = 1)

calendar['date'] = calendar['date'].map(lambda x:pd.to_datetime(x, format = '%Y-%m-%d'))
calendar = calendar.sort_values('date')
calendar['day_num'] = LabelEncoder().fit_transform(calendar['date'])


id_df = train[['id','item_id','dept_id','cat_id','store_id','state_id']]
id_df['id_enc'] = LabelEncoder().fit_transform(id_df['id'])
id_df['item_enc'] = LabelEncoder().fit_transform(id_df['item_id'])
id_df['dept_enc'] = LabelEncoder().fit_transform(id_df['dept_id'])
id_df['cat_enc'] = LabelEncoder().fit_transform(id_df['cat_id'])
id_df['store_enc'] = LabelEncoder().fit_transform(id_df['store_id'])
id_df['state_enc'] = LabelEncoder().fit_transform(id_df['state_id'])


# In[ ]:


from itertools import product
'''
create TimeSeriesMatrx

day_num of matrix = day_num of calendar (sorted data column of calendar to make sure)




'''

matrix = []

id_arr = list(id_df['id_enc'])
for i in range(813):

    matrix.append(list(product([i], id_arr)))
    
matrix = pd.DataFrame(np.vstack(matrix), columns = ['day_num','id'])

matrix[['day_num', 'id']] = matrix[['day_num', 'id']].astype('int16')

matrix['day_num'] = matrix['day_num'] + 1100

matrix['sales'] = np.hstack(train.loc[:, 'd_1101':].values)

matrix['sales'] = matrix['sales'].astype('int16')


# In[ ]:


calendar = calendar.fillna('none')
lol_cols = ['wday', 'month', 'year', 'd',
       'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
       'snap_CA', 'snap_TX', 'snap_WI', 'day_num']
used_cols = []

for col in lol_cols:
    if calendar[col].dtype == object:
        col_name = col + '_enc'
        calendar[col_name] = LabelEncoder().fit_transform(calendar[col])
        used_cols.append(col_name)
    else:
        used_cols.append(col)
        
cal_merge = reduce_mem_usage(calendar[used_cols])
matrix = pd.merge(matrix, cal_merge, how = 'left', on = 'day_num')
id_cols = ['id_enc', 'item_enc', 'dept_enc', 'cat_enc','store_enc', 'state_enc']
matrix = matrix.rename(columns = {"id":"id_enc"})
matrix = pd.merge(matrix, reduce_mem_usage(id_df[id_cols]), how = 'left', on = 'id_enc')


# In[ ]:


matrix.head()


# In[ ]:


lol = matrix.groupby('event_name_1_enc')['sales'].agg(['count','mean']).sort_values('mean').reset_index()
lol = merge_calendar(lol, 'event_name_1')
lol = lol.drop_duplicates().reset_index().drop('index', axis = 1)


# In[ ]:


lol.head(5)


# In[ ]:


lol.tail(20)


# In[ ]:


matrix.head(2)


# In[ ]:


lel = matrix[matrix.event_name_1_enc == 21].groupby('dept_enc')['sales'].agg(['mean','count']).sort_values('mean')
lel = merge_id(lel, 'dept')
lel = lel.drop_duplicates().reset_index().drop('index', axis = 1)


# In[ ]:


lel.head(10)


# In[ ]:


lel.tail(10)


# In[ ]:


lel = matrix[matrix.event_name_1_enc == 21].groupby('dept_enc')['sales'].agg(['mean','count']).sort_values('mean')
lel = merge_id(lel, 'dept')
lel = lel.drop_duplicates().reset_index().drop('index', axis = 1)


# In[ ]:


lel.head()


# In[ ]:


lel


# In[ ]:


id_df.head()


# In[ ]:


def get_name(name, entry):
    dept_dict = id_df[[name + '_id',name + '_enc']].drop_duplicates()
    dept_dict = dict((y,x) for x,y in list(zip(dept_dict[name + '_id'], dept_dict[name + '_enc'])))
    return dept_dict[entry]


# In[ ]:


loldf = matrix.groupby(['day_num','item_enc'])['sales'].sum().reset_index()


# In[ ]:


loldf


# In[ ]:



for item_id in matrix.item_enc.unique()[:10]:
    plt.figure(figsize = (14,4))
    plt.title(get_name('item', item_id))
    plt.plot(loldf[loldf['item_enc'] == item_id].reset_index().['sales'])
    
    


# In[ ]:


for item_id in matrix.item_enc.unique()[:10]:
    plt.figure(figsize = (14,4))
    plt.title(get_name('item', item_id))
    plt.plot(loldf[loldf['item_enc'] == item_id].reset_index()['sales'])


# In[ ]:




