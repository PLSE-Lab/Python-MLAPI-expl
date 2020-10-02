#!/usr/bin/env python
# coding: utf-8

# # The code is using feature engineering from github

# **Learning Objectives:**
#   * Use a pre-definfed DNNLinearCombinedRegressor  estimator of the `Estimator` class in TensorFlow to predict taxi rides
# 

# The data is based on taxi fares. This code was set up as a pipeline in the GCP code. I've refactored it into a Jupyter notebook
# <p>
# Using code from the coursera class which is at github: https://github.com/kariato/training-data-analyst With the actual code being at https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/feateng/taxifare
# 
# <p>
# This is my first python note books so it has bugs please fork and tell me were I went wrong

# ## Set Up
# In this first cell, we'll load the necessary libraries.

# In[ ]:


import math
import shutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from dateutil.parser import parse
from pytz import timezone
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


# # Constants and Hyper-parameters

# In[ ]:


#Constants
MIN_LONG = -74.3
MAX_LONG = -73.0
MIN_LAT = 40.6
MAX_LAT = 41.7
MAX_PASSENGER = 10 
MIN_FARE = 0.0 
output_dir = "."
OUTDIR = "."
OUTPUT_RESULT="submission.csv"
#Hyper prameters
BUCKETS=20
HIDDEN_UNITS = "128 32 4"
SCALE = 10
BATCH_SIZE=32
ROWS_TO_READ=40000
ROWS_TO_SKIP=10
LEARNING_RATE=0.04
STEPS_TO_PROCESS=40000


# Next, we'll load our data set.

# In[ ]:


df = type('', (), {})()
print(datetime.now())
df.train = pd.read_csv('../input/train.csv', sep=",", skiprows=range(1,ROWS_TO_SKIP),nrows=ROWS_TO_READ)
print(datetime.now())
df.test = pd.read_csv('../input/test.csv', sep=",")
print(datetime.now())
#df.train.head(10)
df.train.describe()


# ## Examine the data
# 
# It's a good idea to get to know your data a little bit before you work with it.
# 
# We'll print out a quick summary of a few useful statistics on each column.
# 
# This will include things like mean, standard deviation, max, min, and various quantiles.

# In[ ]:


df.train.head()


# We need to clean up the data to match the test data

# In[ ]:


#The test data is clean the training data has quite a bit of not from new york locations
#I know google thinks you should not do this since the bad data mean something but try
def clean(dfn):
    dfn=dfn[  ((MIN_LONG) <= dfn['pickup_longitude']) & (dfn['pickup_longitude'] <= (MAX_LONG)) ]
    dfn=dfn[ (MIN_LAT <= dfn['pickup_latitude']) & (dfn['pickup_latitude'] <= MAX_LAT) ]
    dfn=dfn[ ((MIN_LONG) <= dfn['dropoff_longitude']) & (dfn['dropoff_longitude'] <= (MAX_LONG) )]
    dfn=dfn[ (MIN_LAT <= dfn['dropoff_latitude'])  & (dfn['dropoff_latitude'] <= MAX_LAT) ]
    dfn=dfn[dfn['passenger_count'] <= MAX_PASSENGER ]
    dfn=dfn[dfn['fare_amount'] >= MIN_FARE ]
    return dfn
df.train=clean(df.train)
df.train.describe()


# # Calculate Time of each ride
# The calcuating the hour and week day for millions of rows is costly so we pre-calcualte all possible values

# In[ ]:


##calculate times
df.train['nyctime'] = df.train.apply(lambda row: row['pickup_datetime'][:14]+'00:00 UTC', axis=1)
df.test['nyctime'] = df.test.apply(lambda row: row['pickup_datetime'][:14]+'00:00 UTC', axis=1)

nycTimes = []
def findTimes(timeStr, nycDict, field):
    if not(timeStr[:14]+'00:00 UTC' in nycDict):
        nycTime = {}
        nycTime['time'] = parse(timeStr).astimezone(timezone('US/Eastern'))
        nycTime['weekday'] = int(nycTime['time'].weekday())
        nycTime['hour'] = int(nycTime['time'].hour)
        nycTime['hourSince2000'] = int(((nycTime['time'].year-2009)*366+int(nycTime['time'].strftime("%j")))*25+nycTime['time'].hour)
        nycTime['nyctime'] = timeStr[:14]+'00:00 UTC'
        nycTimes.append(nycTime)
    return 

minDate=parse(df.train['pickup_datetime'].min())
maxDate=parse(df.train['pickup_datetime'].max())
while (minDate < maxDate):
    findTimes(minDate.strftime("%Y-%m-%d %H:%M:%S%z"),nycTimes,'time')
    minDate = minDate + timedelta(hours=1)

df.times = pd.DataFrame(nycTimes)


# Now join the data frames on the hourly time key

# In[ ]:


df.train=df.train.join(df.times.set_index('nyctime'), on='nyctime')
df.test=df.test.join(df.times.set_index('nyctime'), on='nyctime')
df.times.info()


# In[ ]:


df.train.describe()


# # Feature Engineering on data set

# In[ ]:


# Create feature engineering function that will be used in the input and serving input functions
def add_engineered(features):
    # this is how you can do feature engineering in TensorFlow
    lat1 = features['pickup_latitude']
    lat2 = features['dropoff_latitude']
    lon1 = features['pickup_longitude']
    lon2 = features['dropoff_longitude']
    latdiff = (lat1 - lat2)
    londiff = (lon1 - lon2)
    
    # set features for distance with sign that indicates direction
    features['latdiff'] = latdiff
    features['londiff'] = londiff
    dist = (latdiff * latdiff + londiff * londiff)**(0.5)
    features['euclidean'] = dist
    features['cityBlockDist'] = abs(latdiff) + abs(londiff)
    return features

df.train = add_engineered(df.train)
df.test = add_engineered(df.test)

df.train.head()


# This is the measure used to see how close the data is to actual taxi fares

# In[ ]:


def rmse(labels, predictions):
    pred_values = tf.cast(predictions['predictions'],tf.float64)
    return {'rmse': tf.metrics.root_mean_squared_error(labels*SCALE, pred_values*SCALE)}


# # Build an estimator starting from INPUT COLUMNS.
#      These include feature transformations and synthetic features.
#      The model is a wide-and-deep model.

# In[ ]:


# These are the raw input columns, and will be provided for prediction also
INPUT_COLUMNS = [
    # Define features
    
    # Numeric columns
    tf.feature_column.numeric_column('weekday'),
    tf.feature_column.numeric_column('hour'),
    tf.feature_column.numeric_column('pickup_latitude'),
    tf.feature_column.numeric_column('pickup_longitude'),
    tf.feature_column.numeric_column('dropoff_latitude'),
    tf.feature_column.numeric_column('dropoff_longitude'),
    tf.feature_column.numeric_column('passenger_count'),
    #tf.feature_column.numeric_column('hourSince2000'),
    
    # Engineered features that are created in the input_fn
    tf.feature_column.numeric_column('latdiff'),
    tf.feature_column.numeric_column('londiff'),
    tf.feature_column.numeric_column('euclidean'),
    tf.feature_column.numeric_column('cityBlockDist')
]
# Build the estimator
def build_estimator(model_dir, nbuckets, hidden_units):
    """
     
  """

    # Input columns   hourSince2000,
    (dayofweek, hourofday, plat, plon, dlat, dlon, pcount, latdiff, londiff, euclidean,cityBlockDist) = INPUT_COLUMNS

    # Bucketize the times 
    hourbuckets = np.linspace(0.0, 23.0, 24).tolist()
    b_hourofday = tf.feature_column.bucketized_column(hourofday, hourbuckets)
    weekdaybuckets = np.linspace(0.0, 6.0, 7).tolist()
    b_dayofweek = tf.feature_column.bucketized_column(dayofweek, weekdaybuckets)
    #since2000buckets = np.linspace(0.0, 599999, 60000).tolist()
    #b_hourSince2000 = tf.feature_column.bucketized_column(hourSince2000, since2000buckets)
    
    # Bucketize the lats & lons
    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-76.0, -72.0, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)
   
    # Feature cross
    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets * nbuckets)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets * nbuckets)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets ** 4 )
    day_hr =  tf.feature_column.crossed_column([b_dayofweek, b_hourofday], 24 * 7)

    # Wide columns and deep columns.
    wide_columns = [
        # Feature crosses
        dloc, ploc, pd_pair,
        day_hr,

        # Sparse columns
        b_dayofweek, b_hourofday,
        #b_hourSince2000,

        # Anything with a linear relationship
        pcount 
    ]

    deep_columns = [
        # Embedding_column to "group" together ...
        tf.feature_column.embedding_column(pd_pair, 10),
        tf.feature_column.embedding_column(day_hr, 10),
        #tf.feature_column.embedding_column(b_hourSince2000, 60000),
        # Numeric columns
        plat, plon, dlat, dlon,
        latdiff, londiff, euclidean,cityBlockDist
    ]
    
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir = model_dir,
        linear_feature_columns = wide_columns,
        dnn_feature_columns = deep_columns,
        dnn_hidden_units = hidden_units)

    # add extra evaluation metric for hyperparameter tuning
      
    estimator = tf.contrib.estimator.add_metrics(estimator, rmse)
    return estimator


# ## **Build a neural network model**
# 
# In this exercise, we'll be trying to predicttaxi fares. Ok get all the features into a dictionary

# In[ ]:


feature_columns={}
for i in INPUT_COLUMNS:
    feature_columns[i.key]=i
list(feature_columns.keys())


# Take the panda data and use the estimator functions to turn it into processed data

# In[ ]:


# Split into train and eval and create input functions
msk = np.random.rand(len(df.train)) < 0.8
traindf = df.train[msk]
evaldf = df.train[~msk]

train_input_fn = tf.estimator.inputs.pandas_input_fn(x = traindf[list(feature_columns.keys())],
                                                    y = traindf["fare_amount"] / SCALE,
                                                    num_epochs = 1,
                                                    batch_size = BATCH_SIZE,
                                                    shuffle = True)
eval_input_fn = tf.estimator.inputs.pandas_input_fn(x = evaldf[list(feature_columns.keys())],
                                                    y = evaldf["fare_amount"] / SCALE,  # note the scaling
                                                    num_epochs = 1, 
                                                    batch_size = len(evaldf), 
                                                    shuffle=False)
predict_input_fn = tf.estimator.inputs.pandas_input_fn(x = df.test[list(feature_columns.keys())],
                                                    y = None,  # note the scaling
                                                    num_epochs = 1, 
                                                    batch_size = len(df.test), 
                                                    shuffle=False)


# In[ ]:





# In[ ]:


tf.logging.set_verbosity(tf.logging.INFO)
myopt = tf.train.FtrlOptimizer(learning_rate = LEARNING_RATE) # note the learning rate
estimator = estimator = build_estimator(OUTDIR, BUCKETS, HIDDEN_UNITS.split(' '))
  
estimator = tf.contrib.estimator.add_metrics(estimator,rmse)

train_spec=tf.estimator.TrainSpec(
                     input_fn = train_input_fn,max_steps = STEPS_TO_PROCESS)
eval_spec=tf.estimator.EvalSpec(
                     input_fn = eval_input_fn,
                     steps = None,
                     start_delay_secs = 1, # start evaluating after N seconds
                     throttle_secs = 10,  # evaluate every N seconds
                     )
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# # Predict using the estimator

# In[ ]:


predictions=estimator.predict(input_fn=predict_input_fn)
pred = pd.DataFrame({'fare_amount':[i['predictions'][0]*SCALE for i in predictions]})
submission=pd.concat([df.test['key'],pred],axis=1)
submission.to_csv(OUTPUT_RESULT,index=False)


# In[ ]:




