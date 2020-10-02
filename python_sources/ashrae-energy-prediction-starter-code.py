#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g.L pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# importing libraries

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# #### Reading the data and performing some basic statistics on given data frame using the methods like pandas describe() and info().
# 
# #### To avoid memory issues or reducing the memory we'll supress the datatypes into 32-bit or 16-bit.

# In[ ]:


# Reading the data

# Reading metadata
# specify column dtypes to save memory (by default pandas reads some columns as floats)
dtypes = {
    "site_id": np.int8,
    "building_id": np.int8,
    "primary_use": np.object,
    "square_feet": np.int8,
    "year_built": np.float16,
    "floor_count": np.float16,
}

# meta data information for metadata dataframe
skiprows = None
nrows = None
columns = ['site_id', 'building_id', 'primary_use', 'square_feet', 'year_built', 'floor_count']
path = "/kaggle/input/ashrae-energy-prediction/building_metadata.csv"

metadata_df = pd.read_csv(path, skiprows=skiprows, nrows=nrows, dtype=dtypes, usecols=columns)


# In[ ]:


# Reading training data

# specify column dtypes to save memory (by default pandas reads some columns as floats)
dtypes = {
    "building_id": np.int8,
    "meter": np.int8,
    "timestamp": np.object,
    "meter_reading": np.float16
}

# meta data information for metadata dataframe
skiprows = None
nrows = None
columns = ['building_id', 'meter', 'timestamp', 'meter_reading']
path = "/kaggle/input/ashrae-energy-prediction/train.csv"

# using chunck size to avoid memory related issues
train_df = pd.read_csv(path, skiprows=skiprows, nrows=nrows, dtype=dtypes, usecols=columns)


# In[ ]:


# Reading testing data

# specify column dtypes to save memory (by default pandas reads some columns as floats)
dtypes = {
    "row_id": np.int8,
    "building_id": np.int8,
    "meter": np.int8,
    "timestamp": np.object
}

# meta data information for metadata dataframe
skiprows = None
nrows = None
columns = ['row_id', 'building_id', 'meter', 'timestamp']
path = "/kaggle/input/ashrae-energy-prediction/test.csv"

# using chunck size to avoid memory related issues
test_df = pd.read_csv(path, skiprows=skiprows, nrows=nrows, dtype=dtypes, usecols=columns)


# In[ ]:


# Reading weather training data

# specify column dtypes to save memory (by default pandas reads some columns as floats)
dtypes = {
    "site_id": np.int8,
    "timestamp": np.object,
    "air_temperature": np.float16,
    "cloud_coverage": np.float16,
    "dew_temperature": np.float16,
    "precip_depth_1_hr": np.float16,
    "sea_level_pressure": np.float16,
    "wind_direction": np.float16,
    "wind_speed": np.float16
}

# meta data information for metadata dataframe
skiprows = None
nrows = None
columns = ['site_id', 'timestamp', 'air_temperature', 'cloud_coverage',
       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']
path = "/kaggle/input/ashrae-energy-prediction/weather_train.csv"

weather_train_df = pd.read_csv(path, skiprows=skiprows, nrows=nrows, dtype=dtypes, usecols=columns)


# In[ ]:


# Reading weather testing data

# specify column dtypes to save memory (by default pandas reads some columns as floats)
dtypes = {
    "site_id": np.int8,
    "timestamp": np.object,
    "air_temperature": np.float16,
    "cloud_coverage": np.float16,
    "dew_temperature": np.float16,
    "precip_depth_1_hr": np.float16,
    "sea_level_pressure": np.float16,
    "wind_direction": np.float16,
    "wind_speed": np.float16
}

# meta data information for metadata dataframe
skiprows = None
nrows = None
columns = ['site_id', 'timestamp', 'air_temperature', 'cloud_coverage',
       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']
path = "/kaggle/input/ashrae-energy-prediction/weather_test.csv"

weather_test_df = pd.read_csv(path, skiprows=skiprows, nrows=nrows, dtype=dtypes, usecols=columns)


# In[ ]:


# describing about the data

metadata_df.head()


# In[ ]:


metadata_df.describe()


# In[ ]:


metadata_df.info()


# In[ ]:


weather_test_df.head()


# In[ ]:


weather_train_df.describe()


# In[ ]:


weather_train_df.info()


# In[ ]:


weather_test_df.head()


# In[ ]:


weather_test_df.describe()


# In[ ]:


weather_test_df.info()


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


train_df.info()


# In[ ]:


test_df.head()


# In[ ]:


test_df.describe()


# In[ ]:


test_df.info()


# ### Basic EDA 
# #### Performing some basic EDA which includes techniques like cleaning, feature engineering, transforming and standardising the data.
# #### Performing univariate analysis and bivariate analysis.

# In[ ]:


# checking the null values on metadata dataset

metadata_df.isnull().sum(axis=0)


# In[ ]:


weather_train_df.isnull().sum(axis=0)


# In[ ]:


weather_test_df.isnull().sum(axis=0)


# In[ ]:


train_df.isnull().sum(axis=0)


# In[ ]:


test_df.isnull().sum(axis=0)


# In[ ]:


# percentage of missing values in weather training dataset

round((weather_train_df.isnull().sum() * 100 / len(weather_train_df)), 2)


# In[ ]:


# percentage of missing values in weather testing dataset

round((weather_test_df.isnull().sum() * 100 / len(weather_test_df)), 2)


# In[ ]:


# Finding no.f unique values in weather training dataset

print("Finding no.f unique values in weather training dataset: {}".format(len(pd.unique(weather_train_df['site_id']))))


# In[ ]:


# Finding no.f unique values in weather testing dataset

print("Finding no.f unique values in weather testing dataset: {}".format(len(pd.unique(weather_test_df['site_id']))))


# In[ ]:


# Finding no.f unique values in training dataset 

print("Finding no.f unique values in training dataset: {}".format(len(pd.unique(train_df["building_id"]))))


# In[ ]:


# Finding no.f unique values in testing dataset

print("Finding no.f unique values in testing dataset: {}".format(len(pd.unique(test_df["building_id"]))))


# In[ ]:


# No.f records in weather training dataset

print("No.f records in weather training dataset: {}".format(len(weather_train_df)))


# In[ ]:


# No.f records in weather testing dataset

print("No.f records in weather testing dataset: {}".format(len(weather_test_df)))


# In[ ]:


# No.f records in trainging dataset

print("No.f records in training dataset: {}".format(len(train_df)))


# In[ ]:


# No.f records in testing dataset

print("No.f records in testing dataset: {}".format(len(test_df)))


# #### Converting data types in the given dataset

# In[ ]:


weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])


# In[ ]:


weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])


# In[ ]:


weather_train_df.head()


# In[ ]:


weather_test_df.head()


# In[ ]:


train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])


# In[ ]:


test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


# removing unnecessary data

import gc
gc.collect()


# In[ ]:




