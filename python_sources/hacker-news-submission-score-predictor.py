#!/usr/bin/env python
# coding: utf-8

# # Hacker News Submission Score Predictor w/ Keras and TensorFlow
# 
# by Max Woolf ([@minimaxir](https://minimaxir.com))
# 
# A model of a Hacker News post predictor, using a large number of Keras tricks with a TensorFlow backend.
# 
# This notebook requires a GPU instance. (for the very-fast `CuDNNLSTM` to handle text data)

# In[ ]:


import pandas as pd
import numpy as np
import keras
from google.cloud import bigquery


# BigQuery:
# 
# ```sql
# #standardSQL
# SELECT
#   id,
#   title,
#   REGEXP_REPLACE(NET.HOST(url), 'www.', '') AS domain,
#   FORMAT_TIMESTAMP("%Y-%m-%d %H:%M:%S", timestamp, "America/New_York") AS created_at,
#   score,
#   TIMESTAMP_DIFF(LEAD(timestamp, 30) OVER (ORDER BY timestamp), timestamp, SECOND) as time_on_new
# FROM
#   `bigquery-public-data.hacker_news.full`
# WHERE
#   DATETIME(timestamp, "America/New_York") BETWEEN '2017-01-01 00:00:00' AND '2018-08-01 00:00:00'
#   AND type = "story"
#   AND url != ''
#   AND deleted IS NULL
#   AND dead IS NULL
# ORDER BY
#   created_at DESC
# ```

# Use the query above to get it from BigQuery. (via Kaggle tutorial: https://www.kaggle.com/mrisdal/mentions-of-kaggle-on-hacker-news) Outside of Kaggle, you can get the data using `pandas-gbq`.
# 
# The return data is also randomized; this allows us to use the last 20% as a test set without introducing temporal dependencies.

# In[ ]:


query = '''
#standardSQL
SELECT
  id,
  title,
  REGEXP_REPLACE(NET.HOST(url), 'www.', '') AS domain,
  FORMAT_TIMESTAMP("%Y-%m-%d %H:%M:%S", timestamp, "America/New_York") AS created_at,
  score,
  TIMESTAMP_DIFF(LEAD(timestamp, 30) OVER (ORDER BY timestamp), timestamp, SECOND) as time_on_new
FROM
  `bigquery-public-data.hacker_news.full`
WHERE
  DATETIME(timestamp, "America/New_York") BETWEEN '2017-01-01 00:00:00' AND '2018-08-01 00:00:00'
  AND type = "story"
  AND url != ''
  AND deleted IS NULL
  AND dead IS NULL
ORDER BY
  created_at DESC
'''

client = bigquery.Client()

query_job = client.query(query)

iterator = query_job.result(timeout=30)
rows = list(iterator)

df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

df = df.sample(frac=1, random_state=123).dropna().reset_index(drop=True)
df.head(10)


# ## Feature Engineering
# 
# * Text, w/ sequences of length 15 (HN titles can be from 3 - 80 characters; since words are 5-6 characters)
# * Post domain (if in Top 100 by count; 0 otherwise)
# * Day of Week of Submission
# * Hour of Submission
# 
# Other features I tried but did not use (since using them prevents forecasting, and they did not help improve the model):
# 
# * Trend (time from first submission, scaled to `[0-1]`)
# * Time on `/new` page (scaled to `[0-1]`)
# 
# Score is unmodified. Normally you'd `log` transform a skewed independent variable for a OLS, but that's not necessary for deep learning.

# ### Text
# 
# Use a RNN to encode the title. Since we'll be using an unmasked RNN, length of the submission can be implied from the number of padding characters.

# In[ ]:


from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence, Tokenizer

num_words = 20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(df['title'].values)


# In[ ]:


maxlen = 15

titles = tokenizer.texts_to_sequences(df['title'].values)
titles = sequence.pad_sequences(titles, maxlen=maxlen)
print(titles[0:5,])


# ### Top Domains
# 
# Identify the top *n* domains by count (in this case *n* = 100), then transform it to a *n*D vector for each post.

# In[ ]:


num_domains = 100

domain_counts = df['domain'].value_counts()[0:num_domains]

print(domain_counts)


# In[ ]:


from sklearn.preprocessing import LabelBinarizer

top_domains = np.array(domain_counts.index, dtype=object)

domain_encoder = LabelBinarizer()
domain_encoder.fit(top_domains)

domains = domain_encoder.transform(df['domain'].values.astype(str))
domains[0]


# ### Day-of-Week and Hour
# 
# Convert day-of-week to a 7D vector and hours to a 24D vector. Both pandas and keras have useful functions for this workflow.

# In[ ]:


from keras.utils import to_categorical

dayofweeks = to_categorical(pd.to_datetime(df['created_at']).dt.dayofweek)
hours = to_categorical(pd.to_datetime(df['created_at']).dt.hour)

print(dayofweeks[0:5])
print(hours[0:5])


# ## Sample Weights
# 
# Weight `score=1` samples lower so model places a higher importance on atypical submissions.

# In[ ]:


weights = np.where(df['score'].values == 1, 0.5, 1.0)
print(weights[0:5])


# ## Trend and Time on New
# 
# Unused in final model, but kept here for reference.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

trend_encoder = MinMaxScaler()
trends = trend_encoder.fit_transform(pd.to_datetime(df['created_at']).values.reshape(-1, 1))
trends[0:5]


# In[ ]:


newtime_encoder = MinMaxScaler()
newtimes = trend_encoder.fit_transform(df['time_on_new'].values.reshape(-1, 1))
newtimes[0:5]


# ## Build the Model Prototype

# Add R^2 as a performance metric: https://jmlb.github.io/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/

# In[ ]:


from keras import backend as K

def r_2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# Minimizing `mse` loss as typical for regression problems will not work, as the model will realize that selecting 1 unilaterally accomplishes this task the best.
# 
# Instead, create a hybrid loss of `mae`, `msle`, and `poisson` (see Keras's docs for more info: https://github.com/keras-team/keras/blob/master/keras/losses.py) The latter two losses can account for very high values much better; perfect for the hyper-skewed data.

# In[ ]:


def hybrid_loss(y_true, y_pred):
    weight_mae = 0.1
    weight_msle = 1.
    weight_poisson = 0.1
    
    mae_loss = weight_mae * K.mean(K.abs(y_pred - y_true), axis=-1)
    
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    msle_loss = weight_msle * K.mean(K.square(first_log - second_log), axis=-1)
    
    poisson_loss = weight_poisson * K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)
    return mae_loss + msle_loss + poisson_loss


# In[ ]:


from keras.models import Input, Model
from keras.layers import Dense, Embedding, CuDNNGRU, CuDNNLSTM, LSTM, concatenate, Activation, BatchNormalization
from keras.layers.core import Masking, Dropout, Reshape, SpatialDropout1D
from keras.regularizers import l1, l2

input_titles = Input(shape=(maxlen,), name='input_titles')
input_domains = Input(shape=(num_domains,), name='input_domains')
input_dayofweeks = Input(shape=(7,), name='input_dayofweeks')
input_hours = Input(shape=(24,), name='input_hours')
# input_trend = Input(shape=(1,), name='input_trend')
# input_newtime = Input(shape=(1,), name='input_newtime')

embedding_titles = Embedding(num_words + 1, 50, name='embedding_titles', mask_zero=False)(input_titles)
spatial_dropout = SpatialDropout1D(0.2, name='spatial_dropout')(embedding_titles)
rnn_titles = CuDNNLSTM(128, name='rnn_titles')(spatial_dropout)

concat = concatenate([rnn_titles, input_domains, input_dayofweeks, input_hours], name='concat')

num_hidden_layers = 3

hidden = Dense(128, activation='relu', name='hidden_1', kernel_regularizer=l2(1e-2))(concat)
hidden = BatchNormalization(name="bn_1")(hidden)
hidden = Dropout(0.5, name="dropout_1")(hidden)

for i in range(num_hidden_layers-1):
    hidden = Dense(256, activation='relu', name='hidden_{}'.format(i+2), kernel_regularizer=l2(1e-2))(hidden)
    hidden = BatchNormalization(name="bn_{}".format(i+2))(hidden)
    hidden = Dropout(0.5, name="dropout_{}".format(i+2))(hidden)
    
output = Dense(1, activation='relu', name='output', kernel_regularizer=l2(1e-2))(hidden)

model = Model(inputs=[input_titles,
                      input_domains,
                      input_dayofweeks,
                      input_hours],
                      outputs=[output])

model.compile(loss=hybrid_loss,
              optimizer='adam',
              metrics=['mse', 'mae', r_2])

model.summary()


# The model uses a linear learning rate decay to allow it to learn better once it starts converging.
# 
# Note: in this Kaggle Notebook, the training times out after 33 epochs when committing, so I set it to 25 here. You should probably train for longer. (50+ epochs)

# In[ ]:


from keras.callbacks import LearningRateScheduler, Callback

base_lr = 1e-3
num_epochs = 25
split_prop = 0.2

def lr_linear_decay(epoch):
            return (base_lr * (1 - (epoch / num_epochs)))
    
model.fit([titles, domains, dayofweeks, hours], [df['score'].values],
          batch_size=1024,
          epochs=num_epochs,
          validation_split=split_prop,
          callbacks=[LearningRateScheduler(lr_linear_decay)],
          sample_weight=weights)


# ## Check Predictions Against Validation Set
# 
# Predicting against data that was not trained in the model: the model does this poorly. :(

# In[ ]:


val_size = int(split_prop * df.shape[0])

predictions = model.predict([titles[-val_size:],
                             domains[-val_size:],
                             dayofweeks[-val_size:],
                             hours[-val_size:]])[:, 0]

predictions


# In[ ]:


df_preds = pd.concat([pd.Series(df['title'].values[-val_size:]),
                      pd.Series(df['score'].values[-val_size:]),
                      pd.Series(predictions)],
                     axis=1)
df_preds.columns = ['title', 'actual', 'predicted']
# df_preds.to_csv('hn_val.csv', index=False)
df_preds.head(50)


# ## Check Predictions Against Training Set
# 
# The model should be able to predict these better.

# In[ ]:


train_size = int((1-split_prop) * df.shape[0])

predictions = model.predict([titles[:train_size],
                             domains[:train_size],
                             dayofweeks[:train_size],
                             hours[:train_size]])[:, 0]

df_preds = pd.concat([pd.Series(df['title'].values[:train_size]),
                      pd.Series(df['score'].values[:train_size]),
                      pd.Series(predictions)],
                     axis=1)
df_preds.columns = ['title', 'actual', 'predicted']
# df_preds.to_csv('hn_train.csv', index=False)
df_preds.head(50)


# # LICENSE
# 
# MIT License
# 
# Copyright (c) 2018 Max Woolf
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
