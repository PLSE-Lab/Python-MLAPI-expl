#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# # Read files

# In[ ]:


df = pd.read_csv('bike-rental-train.csv')
dt = pd.read_csv('bike-rental-test.csv')
dt.head()


# # Prepare train and test data

# In[ ]:


continuous = ['temp', 'atemp', 'hum', 'windspeed']
#continuous = ['cnt','temp', 'atemp', 'hum', 'windspeed', 'season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'year']
scaler = StandardScaler()
for var in continuous:
  df[var] = df[var].astype('float64')
  dt[var] = dt[var].astype('float64')
  df[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))
  dt[var] = scaler.fit_transform(dt[var].values.reshape(-1, 1))

dt.head()


# In[ ]:


x_df = df.drop(['instant', 'dteday','casual', 'registered', 'cnt'], axis=1)
x_df


# In[ ]:


y_df = df['cnt']
y_df


# In[ ]:


#x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=1)
x_train = x_df
y_train = y_df
x_train


# # Create and Train Model

# In[ ]:


season = tf.feature_column.categorical_column_with_vocabulary_list("season", range(1,5))
year = tf.feature_column.categorical_column_with_vocabulary_list("year", [2011,2012])
mnth = tf.feature_column.categorical_column_with_vocabulary_list("mnth", range(1,13))
holiday = tf.feature_column.categorical_column_with_vocabulary_list("holiday", [0,1])
weekday = tf.feature_column.categorical_column_with_vocabulary_list("weekday", range(0,7))
workingday = tf.feature_column.categorical_column_with_vocabulary_list("workingday", [0,1])
hr = tf.feature_column.categorical_column_with_vocabulary_list("hr", range(0,24))
weathersit = tf.feature_column.categorical_column_with_vocabulary_list("weathersit", range(1,5))

temp = tf.feature_column.numeric_column("temp")
atemp = tf.feature_column.numeric_column("atemp")
hum = tf.feature_column.numeric_column("hum")
windspeed = tf.feature_column.numeric_column("windspeed")

feat_cols = [season, year, mnth, holiday, weekday, workingday, hr, weathersit, temp, atemp, hum, windspeed]

input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=100, num_epochs=None, shuffle=True)

model = tf.estimator.LinearRegressor(    
        feature_columns=feat_cols,   
        model_dir="train4")	


# In[ ]:


model.train(input_fn=input_func, max_steps=1000000)


#  # Make prediction

# In[ ]:


pred_test_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=dt, batch_size=100, shuffle=False)


# In[ ]:


predictions = pd.DataFrame(model.predict(input_fn=pred_test_func))


# In[ ]:


predictions


#  # Transform predictions

# In[ ]:


dt_with_predictions = dt.merge(predictions, how='outer',left_index=True, right_index=True)
dt_with_predictions


# In[ ]:


dt_predictions = dt_with_predictions[['instant', 'predictions']]
dt_predictions


# In[ ]:


dt_predictions['cnt'] = dt_predictions['predictions'].apply(lambda x : int(round(x[0])))
dt_predictions = dt_predictions.drop(['predictions'], axis=1)
dt_predictions


# In[ ]:


def make_non_negative(x):
  if x<0:
   return 0
  else:
   return x


# In[ ]:


dt_predictions['cnt'] = dt_predictions['cnt'].apply(lambda x : make_non_negative(x))
dt_predictions


# # Save predictions

# In[ ]:


from google.colab import drive
drive.mount('drive')


# In[ ]:


dt_predictions.to_csv('bike-pred.csv')


# In[ ]:


cp bike-pred.csv "drive/My Drive"

