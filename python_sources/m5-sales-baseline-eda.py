#!/usr/bin/env python
# coding: utf-8

# ### Reshape the data, basic EDA & a baseline model
# * Naive baseline - mean on that day of week per store, or the last sale value
# * reshape from columnar (wide format) to long
# 
# * Some eda code borrowed from here: https://www.kaggle.com/rdizzl3/eda-and-baseline-model
# 
# 
# * reshaping note: some items may only start to be sold after a certain date - would be best to cutoff them off before that to avoid noise in the model. e.g. take first index/col >0 as start ? 
# 
# * Issue: currently, notebook crashes when reshaping/pivoting :|. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_sales = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
print(f"train shape {train_sales.shape}")
submission_file = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')
print(f"submission_file shape {submission_file.shape}")


# In[ ]:





# ####### We are given previous data days sales in the sales_train_validation dataset.
# 
# * d_1914 - d_1941 represents the validation row
# * d_1942 - d_1969 represents the evaluation rows.
#     * WE could drop them from the pivoting data (leave ot for testing prediction rows) , or leave them in then split later for easy creation of test set data in smae format

# In[ ]:


days = range(1, 1913 + 1)
time_series_columns = [f'd_{i}' for i in days]

time_series_data = train_sales[time_series_columns]


# In[ ]:


print(train_sales.columns)
id_df_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']


# In[ ]:


train_sales[id_df_columns].nunique()


# In[ ]:


train_sales[id_df_columns + time_series_columns].head(2)


# In[ ]:


##opt - drop out test set rows
# train_sales = train_sales[id_df_columns + time_series_columns]


# In[ ]:


# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type != object:
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
        #else: df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


# ## reduce memory usage. There's an imporved version of this function that also saves data as categoircals type, but that can affect joins if not handled explicitly
# train_sales = reduce_mem_usage(train_sales)

### we know the max range of the sales cols, let's just set them all to int 16 (some are int8 , but that doesn't matter if we 'll cast it)
display(train_sales.info())
train_sales[time_series_columns] = train_sales[time_series_columns].astype(np.int16)


# In[ ]:


# train_sales[id_df_columns] = train_sales[id_df_columns].astype('category')
display(train_sales.info())


# In[ ]:


train_sales.dtypes


# In[ ]:


train_sales


# In[ ]:


submission_file


# In[ ]:


time_series_data


# ## Metadata

# In[ ]:


calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv",parse_dates=["date"])
print(calendar.shape)
prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
print(prices.shape)


# In[ ]:


calendar.tail(3)


# In[ ]:


#no need to keep the textual weekday name, we have it from wday + data. Saturday = 1,Sunday	2, Friday	7
calendar.drop("weekday",axis=1,inplace=True)

## we  drop the prefix from the calendar date/d column for easy merging with sales data. .
calendar["d"] = calendar["d"].replace("d_","",regex=True).astype(int)
calendar


# In[ ]:


prices


# ## Reshape sales data to long format
# * Also join with calendar data
# 
# * The IDs being set to categorical type slows this down immensely
# 
# * due to memory - we may wish to split this into sub frames, them concat them. e.g. split by state or store_id?
# 

# In[ ]:


print(f"After reshaping to 1 row per id per day/date, we would have: {train_sales.shape[0]*time_series_data.shape[1]} rows")
## 58 million rows. many sparse likely


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pd.wide_to_long(train_sales.head(3),"d_",i=id_df_columns,j="sales").reset_index()')


# In[ ]:


stores_list = list(set(train_sales["store_id"]))
stores_list


# In[ ]:


get_ipython().run_cell_magic('time', '', '### reshape incrementally - hopefully this will help with memory errors\ndfs= []\nfor st in stores_list:  \n    df = train_sales.loc[train_sales["store_id"]==st]#.head()\n    dfs.append(pd.wide_to_long(df,"d_",i=id_df_columns,j="day").reset_index())\n    \ndf = pd.concat(dfs)\ndf.rename(columns={"d_":"sales"})\ndel(dfs)\nprint(df.shape)\ndf')


# In[ ]:


df.tail()


# In[ ]:


# %%time
# train_sales = pd.wide_to_long(train_sales,"d_",i=id_df_columns,j="sales").reset_index()
# print(train_sales.shape)
# train_sales


# In[ ]:


df.to_csv("sales_basic_v1_all.csv.gz",index=False,compression="gzip")


# ##### Predictions
# * We need to provide predictions for the next 28 days for each of the series. For the validation series that is days 1914 - 1941 and for the evaluation that is days 1942 - 1969.
# * https://www.kaggle.com/rdizzl3/eda-and-baseline-model

# In[ ]:


validation_ids = train_sales['id'].values
evaluation_ids = [i.replace('validation', 'evaluation') for i in validation_ids]


# In[ ]:


ids = np.concatenate([validation_ids, evaluation_ids])


# In[ ]:


predictions = pd.DataFrame(ids, columns=['id'])
forecast = pd.concat([forecast] * 2).reset_index(drop=True)
predictions = pd.concat([predictions, forecast], axis=1)
predictions.to_csv('submission.csv', index=False)


# In[ ]:




