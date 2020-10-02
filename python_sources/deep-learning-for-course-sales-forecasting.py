#!/usr/bin/env python
# coding: utf-8

# ## Sales Forecasting using MLP, CNN and LSTM
# 
# Quick foreword: This is the first notebook I am publishing! I've been learning ML, DL and Python over 2 years, I have been participating in few hackathons lately. So any feedback is more than welcome, thanks!
# 
# This Neural network is built by creating lag features for the hackathon https://datahack.analyticsvidhya.com/contest/women-in-the-loop-a-data-science-hackathon-by-bain/. The problem statement is to forecast 60 days sales of 600 courses from LearnX which belongs to four domains (Development, Software marketing, Finance & Accounting and Business) given ~882 days of sales of each course. I got private leaderboard score of 126.2250155814
# 
# This Notebook is inspired by https://www.kaggle.com/dimitreoliveira/deep-learning-for-time-series-forecasting. I tried to implement similar strategy for this course sales forecasting. Will explain the approach in the following
# 

# Let's import essential libraries (Here I am using tensorflow 1.14)

# In[ ]:


get_ipython().system('pip install tensorflow==1.14')


# In[ ]:


import pandas as pd
pd.set_option("display.max_columns", 200)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='darkgrid')

import warnings
from keras import optimizers
from keras.utils import plot_model

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout, Activation
from keras.layers import LeakyReLU

import tensorflow as tf

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[ ]:


train_df = pd.read_csv('../input/womenintheloop-data-science-hackathon/train.csv')
test_df = pd.read_csv('../input/womenintheloop-data-science-hackathon/test_QkPvNLx.csv')


# In[ ]:


print("Train shape",train_df.shape)
print("Test shape",test_df.shape)


# In[ ]:


train_df.head(3)


# In[ ]:


train_df.info()


# Replacing *Competition_Metric* nan values to 0

# In[ ]:


train_df.Competition_Metric = train_df.Competition_Metric.fillna(0)
test_df.Competition_Metric = test_df.Competition_Metric.fillna(0)


# In[ ]:


train_df['Course'] = "course_" + train_df['Course_ID'].astype(str)
test_df['Course'] = "course_" + test_df['Course_ID'].astype(str)


# > ## Feature generation and Preprocessing
# 
#    In this example I am considering only selecting *Sales* for creating lag feature. I am only considering 5 days lag features. The whole idea is for each Course take 5 days lag features, current day feature and predict for 5th day from now.
# 
# For this competition, I have considred 60 days of lag features and current day's feature and predicting for 60th day in the future for each course ID

# In[ ]:


def create_lag_features(df, sales_cols, columns_list, lag_days):
    temp = df.copy()
    for i in range(lag_days, 0, -1):
        temp = pd.concat([temp[columns_list],df[sales_cols].shift(i)], axis=1)
        columns_list = columns_list +[sales_col+'_t_'+str(i) for sales_col in sales_cols]
        temp.columns = columns_list
    return temp


# *lag_cols* is the list of columns to create lag features

# In[ ]:


original_column_list = ['ID', 'Day_No', 'Course_ID', 'Course_Domain', 'Course_Type',
                'Short_Promotion', 'Public_Holiday', 'Long_Promotion', 'Competition_Metric']
lag_cols = ['Short_Promotion', 'Long_Promotion', 'Public_Holiday', 'Sales', 'User_Traffic']
lag_days = 60
train_lag_df = pd.DataFrame()
for course_id in train_df.Course_ID.unique():
    column_list = original_column_list.copy()
    temp_df = create_lag_features(train_df.loc[train_df.Course_ID ==course_id], lag_cols, column_list, lag_days)
    train_lag_df = train_lag_df.append(temp_df)
    print("Finished creating lag for course : " + str(course_id))


# We have to drop the rows which doesn't have all lag featues

# In[ ]:


train_lag_df = train_lag_df.dropna()
train_lag_df['User_Traffic'] = train_df['User_Traffic']
train_lag_df['Sales_Today'] = train_df['Sales']


# In[ ]:


derived_test_df = pd.DataFrame()
actual_training_df = pd.DataFrame()
train_target_columns = ['Short_Promotion', 'Public_Holiday', 'Long_Promotion', 'Competition_Metric', 'Sales']
train_target_append_columns = [col+'_t_+60' for col in train_target_columns if 'Sales' not in col]
for course_id in train_df.Course_ID.unique():
    
    train_lag_course_df = train_lag_df.loc[train_lag_df.Course_ID==course_id]
    train_course_df = train_df[train_df.Course_ID==course_id]
#     print("Created df of shape " + str(train_lag_course_df.shape))
    train_target_df = train_course_df[train_target_columns].shift(-60)
    train_target_df.columns = train_target_append_columns + ['Sales']
    temp_actual_training_df = pd.concat([train_lag_course_df, train_target_df], axis=1)
    derived_test_df = derived_test_df.append(temp_actual_training_df[temp_actual_training_df['Sales'].isna()],
                                            verify_integrity=True)
    actual_training_df = actual_training_df.append(temp_actual_training_df.dropna(), verify_integrity=True)
    del temp_actual_training_df
    del train_target_df
    del train_course_df
    del train_lag_course_df


# In[ ]:


# Checking whether the derived test Course_ID is same as test Course_ID
(derived_test_df.sort_values(by=['Course_ID','Day_No'])['Course_ID'].reset_index(drop=True)==test_df.sort_values(by=['Course_ID','Day_No'])['Course_ID'].reset_index(drop=True)).value_counts()


# In[ ]:


model_train_df = actual_training_df.reset_index(drop = True)
model_test_df = derived_test_df.reset_index(drop= True)


# In[ ]:


def overall_preprocessing(df, is_test=False):
    df.Competition_Metric = df.Competition_Metric.fillna(0)
    df['Competition_Metric_t_+60'] = df['Competition_Metric_t_+60'].fillna(0)
    course_type = pd.get_dummies(df['Course_Type'])
    course_domain = pd.get_dummies(df['Course_Domain'])
    
    user_traffic_columns = [col for col in df.columns if 'User_Traffic' in col]
    
    df[user_traffic_columns] = df[user_traffic_columns]/100
    df_processed = pd.concat([df, course_type, course_domain], axis=1)
    df_processed['Day_No'] = df_processed['Day_No'].mod(365)
    df_processed = df_processed.drop(columns = ['ID','Course_Type','Course_Domain'])
    if is_test:
        del df_processed['Sales']
        print("Test shape: " + str(df_processed.shape))
        return df_processed
    else:
        target = df_processed[['Sales']]
        del df_processed['Sales'] 
        print("Train shape: "+str(df_processed.shape))
        return df_processed, target


# In[ ]:


model_encoded_train_df, model_target_df = overall_preprocessing(model_train_df)
model_encoded_test_df = overall_preprocessing(model_test_df, True)


# In[ ]:


X_train, X_valid, Y_train, Y_valid = train_test_split(model_encoded_train_df, model_target_df.values, test_size=0.3, random_state=45)
print('Train set shape', X_train.shape)
print('Validation set shape', X_valid.shape)


# In[ ]:


X_train.isnull().any().value_counts()


# > ## MLP for Time Series Forecasting
# 
# * First we will use a Multilayer Perceptron model or MLP model, here our model will have input features equal to the window size.
# * The thing with MLP models is that the model don't take the input as sequenced data, so for the model, it is just receiving inputs and don't treat them as sequenced data, that may be a problem since the model won't see the data with the sequence patter that it has.
# * Input shape **[samples, timesteps]**.

# In[ ]:


batch = 64
lr = 0.0003
adam = optimizers.Adam(lr)
epochs = 5
leaky_relu_alpha =0.05


# In[ ]:


model_mlp = Sequential()
model_mlp.add(Dense(512, input_dim=X_train.shape[1]))
model_mlp.add(LeakyReLU(alpha=leaky_relu_alpha))
model_mlp.add(Dense(128, kernel_initializer='normal'))
model_mlp.add(LeakyReLU(alpha=leaky_relu_alpha))
model_mlp.add(Dense(32, kernel_initializer='normal'))
model_mlp.add(LeakyReLU(alpha=leaky_relu_alpha))
model_mlp.add(Dense(1))
model_mlp.compile(loss='mse', optimizer=adam, metrics=['msle'])
model_mlp.summary()


# In[ ]:


mlp_history = model_mlp.fit(X_train.values, Y_train,
                            validation_data=(X_valid.values, Y_valid),
                            epochs=epochs, verbose=1, batch_size=batch)


# In[ ]:


def save_submission(df_pred_ID, prediction, filename):
    result = pd.concat([df_pred_ID,pd.DataFrame({'Sales':list(prediction)})],axis=1)
    #result.to_csv('submissions/' + filename + '.csv', index=False)
    return result


# In[ ]:


test_df['ID'].shape


# In[ ]:


df_pred_ID = test_df['ID']
result = save_submission(df_pred_ID, model_mlp.predict(model_encoded_test_df).flatten(),'MLP_Time_series')


# > ## CNN for Time Series Forecasting
# 
# * For the CNN model we will use one convolutional hidden layer followed by a max pooling layer. The filter maps are then flattened before being interpreted by a Dense layer and outputting a prediction.
# * The convolutional layer should be able to identify patterns between the timesteps.
# * Input shape **[samples, timesteps, features]**.
# 
# #### Data preprocess
# * Reshape from [samples, timesteps] into [samples, timesteps, features].
# * This same reshaped data will be used on the CNN and the LSTM model.

# In[ ]:


epochs = 3


# In[ ]:


X_train['Padding'] = 0
X_valid['Padding'] = 0
model_encoded_test_df['Padding'] = 0 


# In[ ]:


subsequences = 5
timesteps = X_train.shape[1]//subsequences
X_train_series = X_train.values.reshape((X_train.shape[0], timesteps, subsequences))
X_valid_series = X_valid.values.reshape((X_valid.shape[0], timesteps, subsequences))
X_test_series = model_encoded_test_df.values.reshape((model_encoded_test_df.shape[0], timesteps, subsequences))
print('Train series shape', X_train_series.shape)
print('Validation series shape', X_valid_series.shape)
print('Test series shape', X_test_series.shape)


# In[ ]:


model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(64, kernel_initializer='normal', activation='relu'))
model_cnn.add(Dense(32, kernel_initializer='normal', activation='relu'))
model_cnn.add(Dense(1))
model_cnn.compile(loss='mse', optimizer=adam, metrics=['msle'])
model_cnn.summary()


# In[ ]:


cnn_history = model_cnn.fit(X_train_series, Y_train,
                            validation_data=(X_valid_series, Y_valid),
                            epochs=epochs, verbose=1, batch_size=batch)


# In[ ]:


result_cnn = save_submission(df_pred_ID, model_cnn.predict(X_test_series).flatten(),'CNN_Time_series')


# > ## LSTM for Time Series Forecasting
# 
# * Now the LSTM model actually sees the input data as a sequence, so it's able to learn patterns from sequenced data (assuming it exists) better than the other ones, especially patterns from long sequences.
# * Input shape **[samples, timesteps, features]**.

# In[ ]:


epochs = 1
batch = 128
lr = 0.0076
adam = optimizers.Adam(lr)
leaky_relu_alpha =0.05


# In[ ]:


model_lstm = Sequential()
model_lstm.add(LSTM(64, input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
model_lstm.add(LeakyReLU(alpha=leaky_relu_alpha))
model_lstm.add(Dense(32, kernel_initializer='normal'))
model_lstm.add(LeakyReLU(alpha=leaky_relu_alpha))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mse', optimizer=adam, metrics=['msle'])
model_lstm.summary()


# In[ ]:


lstm_history = model_lstm.fit(X_train_series, Y_train,
                            validation_data=(X_valid_series, Y_valid),
                            epochs=epochs, verbose=1, batch_size=batch)


# In[ ]:


result_lstm = save_submission(df_pred_ID, model_lstm.predict(X_test_series).flatten(),'LSTM_Time_series')

