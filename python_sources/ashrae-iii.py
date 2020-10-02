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


# **Load the data files into Panda dataframes**

# In[ ]:


building_metadata=pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
weather_train=pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')
train=pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')


# **![](http://)![](http://)Explore Building Meta Data content**

# In[ ]:


#view some records of building_metadata

building_metadata.head()


# In[ ]:


building_metadata.describe()


# Notice here above the statistics are not relevant for site_id and building_id but you can check out the average square feet, year of build and number of floors. You can also check the min and max of these to say a few. Also seems that we have 1449 buildings. There are no statistics available for categorical data.
# 
# Let's see then how the buildings are distributed by primary_use

# **Define memory optimization function**

# In[ ]:


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist


# In[ ]:


reduce_mem_usage(train)


# In[ ]:


reduce_mem_usage(weather_train)


# In[ ]:


building_metadata.loc[: ,['building_id', 'primary_use']].groupby('primary_use').count().plot(kind='bar')


# **Not clear to me why soo big proportion of Education buildings**

# **Explore Train data**

# In[ ]:


#display some rows of train data
train.head()


# In[ ]:


#display information about train data. After a first attempt, added a column meter_reading_int to make it more readble
train['meter_reading_int']=train['meter_reading'].apply(lambda x: int(x))
print(train['meter_reading_int'].mean())


# In[ ]:


train.describe()


# Define ECDF function, can be usefull for data representation

# In[ ]:


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y


# In[ ]:


import matplotlib.pyplot as plt
x, y = ecdf(train["meter_reading"])


# In[ ]:


_ = plt.hist(x, y, bin=50)
plt.show()


# In[ ]:


weather_train.describe()


# In[ ]:


#take a look on how surface, number of floors and building age are correlated
import matplotlib.pyplot as plt 
metadata_pivot = building_metadata.groupby(building_metadata["primary_use"]).count()

plt.xticks(rotation=90)
_ = plt.bar(metadata_pivot.index, metadata_pivot.site_id)


# In[ ]:


#merge the data 
train_details = building_metadata.merge(train, on="building_id")


# In[ ]:


train_details.describe()


# In[ ]:


train_details.head()


# 

# In[ ]:


train_details['timestamp']=pd.to_datetime(['timestamp'])

