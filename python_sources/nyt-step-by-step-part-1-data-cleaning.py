#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # <center> Data analysis

# Give me data for analysis!!!

# In[ ]:


train_df = pd.read_csv('../input/train.csv', nrows=1000000)
test_df = pd.read_csv('../input/test.csv', nrows=1000000)


# Let's watch on statistics in training data.

# In[ ]:


train_df.describe()


# I'm not an expert in geodata; but I know that the longitude is in the range of 180 to -180, and the latitude is in the range of 90 to -90.  But this border very common. Because it, Let's watch on statistics in test data. But befor,let's extract year from time.

# In[ ]:


test_df['year'] = test_df.loc[:,'pickup_datetime'].apply(lambda x: int(x[:4]))


# In[ ]:


test_df.describe()


# Oh, Watch on this data! It's data with good range of longitude and latitude. Let's save max and min value of longitude, latitude, year and number passenger how the borders for cleaning training data.

# In[ ]:


border = {
   'longitude': [min(test_df.pickup_longitude.min(), test_df.dropoff_longitude.min()),
                max(test_df.pickup_longitude.max(), test_df.dropoff_longitude.max())],
   'latitude': [min(test_df.pickup_latitude.min(), test_df.dropoff_latitude.min()),
                max(test_df.pickup_latitude.max(), test_df.dropoff_latitude.max())],
   'year': [test_df.year.min(), test_df.year.max()],
    'passenger_count': [test_df.passenger_count.min(), test_df.passenger_count.max()]
}
border


# In[ ]:


import gc
del train_df
del test_df
gc.collect()


# # <center> Data Cleaning
#    Clean data and save in <b>.csv file</b> with <b>gzip</b> compression.

# In[ ]:


train_df = None
for i,train_df_i in enumerate(pd.read_csv('../input/train.csv', chunksize=int(10**6))):
    print(f'Start procesing {i} chunk...')
    start_len = len(train_df_i)
    train_df_i['year'] = train_df_i.loc[:,'pickup_datetime'].apply(lambda x: int(x[:4]))
    train_df_i = train_df_i[(train_df_i.pickup_longitude >= border['longitude'][0])&(train_df_i.pickup_longitude <= border['longitude'][1])&
                            (train_df_i.dropoff_longitude >= border['longitude'][0])&(train_df_i.dropoff_longitude <= border['longitude'][1])&
                            (train_df_i.pickup_latitude >= border['latitude'][0])&(train_df_i.pickup_latitude <= border['latitude'][1])&
                            (train_df_i.dropoff_latitude >= border['latitude'][0])&(train_df_i.dropoff_latitude <= border['latitude'][1])&
                            (train_df_i.year >= border['year'][0]) & (train_df_i.year <= border['year'][1]) & 
                            (train_df_i.passenger_count >= border['passenger_count'][0])&((train_df_i.passenger_count <= border['passenger_count'][1]))]
    end_len = len(train_df_i)
    print(f'\t Clean rate: {(start_len-end_len)/start_len}')
    if train_df is None:
        train_df = train_df_i
    else:
        train_df = pd.concat([train_df, train_df_i])
    if i%8 == 0 and i != 0 or i==55:
        train_df.to_csv(f'train_{i//8}.csv', compression='gzip')
        print(f'\t train_{i//8}.csv save...')
        train_df = None
    del train_df_i; gc.collect()


# <p>Clean rate in all chunk was about 0.025. What does it mean?</p>
# <p>It's mean that dataset have about 2.5% uncertain samples. And this samples was removed from dataset and split on 6 new file in csv file, which was saved with gzip compression</p>

# In[ ]:




