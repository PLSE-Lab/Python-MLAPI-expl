#!/usr/bin/env python
# coding: utf-8

# Read data

# In[ ]:


import numpy as np
import pandas as pd

def read_data():
    train = pd.read_csv("../input/sales_train.csv", index_col=0)
    test = pd.read_csv("../input/test.csv", index_col=0)
    submission = pd.read_csv("../input/sample_submission.csv", index_col=0)
    return train, test, submission

train, test, submission = read_data()


# In[ ]:


N_SESSIONS = 34


# Load external cython library
# 

# In[ ]:


get_ipython().run_line_magic('load_ext', 'cython')


# Using Cython to make sure the kernel would run in under an hour

# In[ ]:


get_ipython().run_cell_magic('cython', ' ', '\nimport numpy as np # linear algebra\ncimport numpy as np\n\nDTYPE_F = np.float\nctypedef np.float_t DTYPE_F_t\n\nDTYPE_F = np.int\nctypedef np.int_t DTYPE_I_t\n\ncdef int N_SESSIONS = 34\n\ndef create_time_series_train(np.ndarray[DTYPE_F_t, ndim=2] train_shop_item_month_count,\n                             np.ndarray[DTYPE_I_t, ndim=2] test_shop_item):\n    cdef i, shop, item\n    train_time_series = np.zeros((test_shop_item.shape[0], N_SESSIONS), dtype=np.int)  # number of train months\n    for i in range(test_shop_item.shape[0]):\n        if not i % 1000:\n            print(i)\n        # Getting current shop and item from test\n        shop = test_shop_item[i, 0]\n        item = test_shop_item[i, 1]\n        # Filtering middle stage for faster iterations\n        filtered_train = train_shop_item_month_count[np.logical_and(\n            train_shop_item_month_count[:, 0] == shop,\n            train_shop_item_month_count[:, 1] == item)]\n        # Filling time series table\n        for j in range(N_SESSIONS):\n            train_current_month = filtered_train[filtered_train[:, 2] == j]\n            train_time_series[i, j] = np.sum(train_current_month[:, 3])\n    return train_time_series')


# The function's inputs are np.array instead of pd.Dataframe for cython competability.
# The functions takes the data from the train and filters it by the test's indices of store and item.
# for each item and store in test it sum of items in the training months (0-33)

# In[ ]:


train_time_series = create_time_series_train(
    train[["shop_id", "item_id", "date_block_num", "item_cnt_day"]].values,
    test[["shop_id", "item_id"]].values)
train_time_series_df = pd.DataFrame(train_time_series, columns=range(N_SESSIONS))


# train_time_series_df could be saved for later use as it is the basis of a lot of manipulations
# possible and it takes some time to run

# In[ ]:


# train_time_series_df.to_csv("input/train_ts.csv")
# train_time_series_df = pd.read_csv("input/train_ts.csv", index_col=0).values
train_time_series_df.head()


# Creating double exponential filtering over the time series. 
# I found that giving momentum gave worse results

# In[ ]:


ALPHA = 0.5
BETA = 0.0

def create_filtered_prediction(train_ts, alpha, beta):
    train_time_filtered_ts = np.zeros((train_ts.shape[0], N_SESSIONS), dtype=np.float)
    train_time_filtered_ts[0, :] = train_ts[0, :]
    train_memontum_ts = np.zeros((train_ts.shape[0], N_SESSIONS), dtype=np.float)
    prediction_ts = np.zeros((train_ts.shape[0], N_SESSIONS+1), dtype=np.float)
    for i in range(1, N_SESSIONS):
        train_time_filtered_ts[:, i] = (1-alpha) * (train_time_filtered_ts[:, i-1] +                                                     train_memontum_ts[:, i-1]) + alpha * train_ts[:, i]
        train_memontum_ts[:, i] = (1-beta) * train_memontum_ts[:, i-1] +                                   beta * (train_time_filtered_ts[:, i] - train_time_filtered_ts[:, i-1])
        prediction_ts[:, i+1] = train_time_filtered_ts[:, i] + train_memontum_ts[:, i]
    return prediction_ts

train_time_series_df = np.clip(train_time_series_df.values, 0, 20).astype(float)
predictions = create_filtered_prediction(train_time_series_df, ALPHA, BETA)


# In[ ]:


submission["item_cnt_month"] = np.clip(predictions[:, N_SESSIONS], 0, 20)
submission.to_csv("alpha_{0}.csv".format(ALPHA))

