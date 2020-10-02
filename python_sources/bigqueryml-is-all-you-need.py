#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# Google released BigQuery ML.
# 
# - https://ai.googleblog.com/2018/07/machine-learning-in-google-bigquery.html
# - https://cloud.google.com/bigquery/docs/bigqueryml
# 
# Let's try BigQuery ML !!!

# Before start.  
# Sorry... This kernel shows only SQL.  
# If you want to try BigQuery ML, upload dataset own BigQuery Project.  
# 
# If you run this code, model will be created within 10 minutes.  
# And, prediction will finish within 3 seconds.  

# In[ ]:


create_model_sql='''
CREATE MODEL
  `nyc_taxi.mymodel`
OPTIONS
  ( model_type='linear_reg',
    ls_init_learn_rate=.15,
    l1_reg=1,
    max_iterations=5 ) AS
SELECT
    fare_amount label,
    ABS(pickup_longitude - dropoff_longitude) diff_longitude,
    ABS(pickup_latitude - dropoff_latitude) diff_latitude,
    SQRT(POW(ABS(pickup_longitude - dropoff_longitude),2) + POW(ABS(pickup_latitude - dropoff_latitude),2)) distance,
    EXTRACT(DAYOFWEEK from pickup_datetime) the_dayofweek,
    EXTRACT(DAY from pickup_datetime) the_day,
    EXTRACT(DAYOFYEAR from pickup_datetime) the_dayofyear,
    EXTRACT(WEEK from pickup_datetime) the_week,
    EXTRACT(MONTH from pickup_datetime) the_month,
    EXTRACT(QUARTER from pickup_datetime) the_quater,
    EXTRACT(YEAR from pickup_datetime) the_year,
    passenger_count
FROM
  `nyc_taxi.train`
'''


# In[ ]:


predict_sql='''
SELECT
  key,
  predicted_label fare_amount
FROM
  ML.PREDICT(MODEL `nyc_taxi.mymodel`,
    (
    SELECT
    key,
    ABS(pickup_longitude - dropoff_longitude) diff_longitude,
    ABS(pickup_latitude - dropoff_latitude) diff_latitude,
    SQRT(POW(ABS(pickup_longitude - dropoff_longitude),2) + POW(ABS(pickup_latitude - dropoff_latitude),2)) distance,
    EXTRACT(DAYOFWEEK from pickup_datetime) the_dayofweek,
    EXTRACT(DAY from pickup_datetime) the_day,
    EXTRACT(DAYOFYEAR from pickup_datetime) the_dayofyear,
    EXTRACT(WEEK from pickup_datetime) the_week,
    EXTRACT(MONTH from pickup_datetime) the_month,
    EXTRACT(QUARTER from pickup_datetime) the_quater,
    EXTRACT(YEAR from pickup_datetime) the_year,
    passenger_count
    FROM
      `nyc_taxi.test`))
'''


# In[ ]:


get_ipython().system('ls ../input/bigquery-prediction')


# In[ ]:


# pseudo code (export bigquery results as prediction)
subs = pd.read_csv("../input/bigquery-prediction/results-20180727-011423.csv")
subs.to_csv("subs.csv",index=False)


# In[ ]:




