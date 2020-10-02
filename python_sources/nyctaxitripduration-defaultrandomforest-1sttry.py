#!/usr/bin/env python
# coding: utf-8

# New York City Taxi Trip Duration
# 
# https://www.kaggle.com/c/nyc-taxi-trip-duration

# In[88]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/nyc-taxi-trip-duration"))

# Any results you write to the current directory are saved as output.


# In[89]:


# !ls ../input/nytaxi-clean-pairplot


# In[90]:


get_ipython().system('ls ../input/nytaxi-clean-pairplot/pairplot.png')


# In[103]:


# Import libraries
import datetime
import math

import geopy.distance

import seaborn as sns
import matplotlib.pyplot as plt


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Assess data

# Let's first start by loading the training data and inspect it

# In[92]:


train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv',
                   parse_dates=['pickup_datetime', 'dropoff_datetime'],
                   dtype={'store_and_fwd_flag':'category'})


# In[93]:


train.info()


# In[8]:


train.head()


# In[9]:


train.shape


# ### Quality & Tidiness
# 
# #### Categorical column (store_and_fwd_flag)
# * This column must be converted to a numerical value by using** cat.codes and cast it to int
# 
# #### Datetime columns (pickup_datetime)
# * datetime columns which are **pickup_datetime** should be split to 'dayofweek', 'dayofyear', 'weekofyear', 'month', 'hour' (We should not need to do this for and **dropoff_datetime** because they are likely to be the same and we already have the tripduration which we try to predict
# 
# 
# #### Location columns (pickup_longitude	pickup_latitude	dropoff_longitude	dropoff_latitude)
# * To create a train data, create a new column **distance_km** to store a distance value in km computed from (pickup_longitude	pickup_latitude	dropoff_longitude	dropoff_latitude) and drop those data that exceed .99 quantile
# * Do nothing for the validation data
# 
# #### Drop unused columns
# * **id** column can be dropped because we do not need it in training
# * **pickup_datetime** and **dropoff_datetime** must be dropped after all above are done

# 

# ## Data Wrangling
# 
# Transform the original data to follow our requirements in the Quality & Tidiness section above

# In[99]:


def create_datetime_columns(df, column_list):
    
    for col_name in column_list:
        df[col_name+ '_' + 'dayofweek'] = df[col_name].dt.dayofweek
        df[col_name+ '_' + 'dayofyear'] = df[col_name].dt.dayofyear
        df[col_name+ '_' + 'weekofyear'] = df[col_name].dt.weekofyear
        df[col_name+ '_' + 'month'] = df[col_name].dt.month
        df[col_name+ '_' + 'hour'] = df[col_name].dt.hour
        
    return df


# In[100]:


def get_distance_km(row):
    coords_1 = (row.pickup_latitude, row.pickup_longitude)
    coords_2 = (row.dropoff_latitude, row.dropoff_longitude)
    
    return geopy.distance.geodesic(coords_1, coords_2).km


# In[177]:


def transform_data(df, cleanData=False):
    
    data_clean = df.copy()
    
    #### Categorical column (store_and_fwd_flag)
    # This column must be converted to a numerical value by 
    # using cat.codes and cast it to int
    data_clean['store_and_fwd_flag'] = data_clean['store_and_fwd_flag'].cat.codes

    #### Datetime columns (pickup_datetime)
    # datetime columns which is **pickup_datetime**
    # should be split to 'dayofweek', 'dayofyear', 'weekofyear', 'month', 'hour'
#     data_clean = create_datetime_columns(data_clean, 
#                                          ['pickup_datetime', 'dropoff_datetime'])
    # Only do get additional column for pickup_datetime should be enought because
    # They are typically on the same day
    data_clean = create_datetime_columns(data_clean, 
                                         ['pickup_datetime'])

    #### Location columns (pickup_longitude	pickup_latitude	dropoff_longitude	dropoff_latitude)
    # Create a new column **distance_km** to store a distance value in km computed from (pickup_longitude	pickup_latitude	dropoff_longitude	dropoff_latitude)
    data_clean['distance_km'] = data_clean.apply(lambda row: get_distance_km(row), axis=1)
       
    if cleanData:
        # After doing the exploratory analysis, I found that there are outliers in the dataset
        # (there are trips that have 1k km) that could potentially cause an unexpected behavior
        # Hence, remove those outlier data before proceeding         
        data_clean = data_clean[data_clean.distance_km < data_clean.distance_km.quantile(0.99)]
    
    
    #### Drop unused columns
    # **id** column can be dropped because we do not need it in training
    # **pickup_datetime** and **dropoff_datetime** must be dropped after all above are done

    data_clean = data_clean.drop(['id', 
                                  'pickup_datetime'
                                 ], axis=1)
    
    # Test data does not have dropof_datetime column. Hence, skip it
    if data_clean.columns.contains('dropoff_datetime'):
        data_clean = data_clean.drop(['dropoff_datetime'], axis=1)
    
    return data_clean


# In[102]:


get_ipython().run_line_magic('time', 'data_clean = transform_data(train, cleanData=True)')
data_clean.reset_index().to_feather('data_clean')


# In[35]:


# # # We will use a saved clean data from the previous session here
# data_clean = pd.read_feather('../input/nytaxi-clean-feather/data_clean')


# In[111]:


data_clean.head()


# In[112]:


# Inspect the output dataframe
data_clean.sample(20)


# In[113]:


data_clean.info()


# ## Exploratory Data Analysis (EDA)

# Before starting the modeling process, let's explore the data a little more to better understand it

# ### Correlation

# In[114]:


corr = data_clean.corr()


# In[115]:


plt.figure(figsize=(8,6))

sns.heatmap(corr);


# In[116]:


corr.style.background_gradient(cmap='coolwarm')


# In[117]:


# # sns_plot = sns.pairplot(df)
# # sns_plot.show()

# # Load a picture of the pairplot generated from the command above instead since
# # it takes a long time (around two hours) to create it
# from IPython.display import Image
# Image("../input/nytaxi-clean-pairplot/pairplot.png")


# In[118]:


# Get all column names
data_clean.columns.tolist()


# ### vendor_id

# In[119]:


data_clean['vendor_id'].value_counts()


# In[120]:


data_clean['passenger_count'].hist();


# In[121]:


data_clean['passenger_count'].plot.box();


# In[122]:


data_clean['passenger_count'].describe()


# In[123]:


sns.countplot(x="passenger_count", hue="vendor_id", data=data_clean);


# ### Trip duration

# In[130]:


data_clean.distance_km.describe()


# In[131]:


data_clean.shape


# ## Use Default RandomForestRegressor Model

# In[132]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# In[133]:


X = data_clean.drop(['trip_duration'], axis=1)
y = data_clean['trip_duration']


# In[134]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)


# In[136]:


X_train.shape, X_valid.shape


# In[137]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m, X_train, X_valid, y_train, y_valid):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[138]:


m = RandomForestRegressor(n_jobs=-1)

print('[{}] Start'.format(datetime.datetime.now()))

get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')


# In[141]:


print('[{}] Start'.format(datetime.datetime.now()))


get_ipython().run_line_magic('time', 'print_score(m, X_train, X_valid, y_train, y_valid)')


# Let's try to predict **trip_duration** of the **train****** data

# In[150]:


y_pred_train = m.predict(X_train)


# Let's try to predict **trip_duration** of the **validation** data

# In[143]:


y_pred = m.predict(X_valid)


# Now, let's find of Root Mean Squared Logarithmic Error of the predicted data.
# This is an evaluation metrics defined in Kaggle
# 
# https://www.kaggle.com/c/nyc-taxi-trip-duration/overview/evaluation

# In[147]:


# From https://stackoverflow.com/questions/46202223/root-mean-log-squared-error-issue-with-scitkit-learn-ensemble-gradientboostingre
def rmsle(y, y0):
    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))


# Here is RMSLE of the train data

# In[152]:


rmsle(y_train, y_pred_train)


# Here is RMSLE of the validation data

# In[149]:


rmsle(y_valid, y_pred)


# ### Just for fun

# Let submit this simple model and submit it for the competition to see where we are in the leadership board!

# In[161]:


test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv',
                   parse_dates=['pickup_datetime'],
                   dtype={'store_and_fwd_flag':'category'})


# Inspect the test data first. Notice that the **dropoff_datetime** and **trip_duration** columns are not included in this dataset!

# In[162]:


test.head()


# In[166]:


test.shape


# We need to transform data before passing it to the model and this can be done by using the **transform_data** function

# In[178]:


get_ipython().run_line_magic('time', 'test_clean = transform_data(test)')


# Inspect data to ensure that we get all appropriate columns and no rows are removed

# In[180]:


test_clean.head()


# In[181]:


test_clean.info()


# In[182]:


test_clean.shape


# In[183]:


X_sub = test_clean.copy()


# In[184]:


y_sub = m.predict(X_sub)


# Now, replace data in the **trip_duration** column of a dataframe created from the **sample_submission.csv** with our predicted data, and save it to a csv file for submission

# In[185]:


df_sub = pd.read_csv('../input/nyc-taxi-trip-duration/sample_submission.csv')


# In[186]:


df_sub.head()


# In[187]:


df_sub['trip_duration'] = y_sub
df_sub.head()


# In[189]:


df_sub.shape


# In[188]:


df_sub.to_csv('submission_default_randomforest.csv', index=False)


# In[ ]:




