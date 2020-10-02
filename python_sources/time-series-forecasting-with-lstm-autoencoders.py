#!/usr/bin/env python
# coding: utf-8

# <h1><center>Time-series forecasting with deep learning & LSTM autoencoders</center></h1>
# 
# * The purpose of this work is to show one way time-series data can be effiently encoded to lower dimensions, to be used into non time-series models.
# * Here I'll encode a time-series of size 12 (12 months) to a single value and use it on a MLP deep learning model, instead of using the time-series on a LSTM model that could be the regular approach.
# * The first part of the data preparation is from my other kernel [Model stacking, feature engineering and EDA](https://www.kaggle.com/dimitreoliveira/model-stacking-feature-engineering-and-eda).
# * This work was inspired by this Machinelearningmastery post [A Gentle Introduction to LSTM Autoencoders](https://machinelearningmastery.com/lstm-autoencoders/), make sure to check out.

# <h2><center>Predict future sales</center></h2>
# 
# We are asking you to predict total sales for every product and store in the next month. By solving this competition you will be able to apply and enhance your data science skills.
# 
# You are provided with daily historical sales data. The task is to forecast the total amount of products sold in every shop for the test set. Note that the list of shops and products slightly changes every month. Creating a robust model that can handle such situations is part of the challenge.
# 
# #### Data fields description:
# * ID - an Id that represents a (Shop, Item) tuple within the test set
# * shop_id - unique identifier of a shop
# * item_id - unique identifier of a product
# * item_category_id - unique identifier of item category
# * date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
# * date - date in format dd/mm/yyyy
# * item_cnt_day - number of products sold. You are predicting a monthly amount of this measure
# * item_price - current price of an item
# * item_name - name of item
# * shop_name - name of shop
# * item_category_name - name of item category
# 
# ### Dependencies

# In[ ]:


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings("ignore")

# Set seeds to make the experiment more reproducible.
from numpy.random import seed
seed(0)
from tensorflow import set_random_seed
set_random_seed(0)


# ### Loading data

# In[ ]:


test = pd.read_csv('../input/test.csv', dtype={'ID': 'int32', 'shop_id': 'int32', 
                                                  'item_id': 'int32'})
item_categories = pd.read_csv('../input/item_categories.csv', 
                              dtype={'item_category_name': 'str', 'item_category_id': 'int32'})
items = pd.read_csv('../input/items.csv', dtype={'item_name': 'str', 'item_id': 'int32', 
                                                 'item_category_id': 'int32'})
shops = pd.read_csv('../input/shops.csv', dtype={'shop_name': 'str', 'shop_id': 'int32'})
sales = pd.read_csv('../input/sales_train.csv', parse_dates=['date'], 
                    dtype={'date': 'str', 'date_block_num': 'int32', 'shop_id': 'int32', 
                      'item_id': 'int32', 'item_price': 'float32', 'item_cnt_day': 'int32'})


# ### Join data sets

# In[ ]:


train = sales.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id', rsuffix='_').join(item_categories, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)


# ### Let's take a look at the raw data

# In[ ]:


print('Train rows: ', train.shape[0])
print('Train columns: ', train.shape[1])


# In[ ]:


train.head().T


# In[ ]:


train.describe()


# ### Time period of the dataset

# In[ ]:


print('Min date from train set: %s' % train['date'].min().date())
print('Max date from train set: %s' % train['date'].max().date())


# I'm leaving only the "shop_id" and "item_id" that exist in the test set to have more accurate results.

# In[ ]:


test_shop_ids = test['shop_id'].unique()
test_item_ids = test['item_id'].unique()
# Only shops that exist in test set.
train = train[train['shop_id'].isin(test_shop_ids)]
# Only items that exist in test set.
train = train[train['item_id'].isin(test_item_ids)]


# ### Data preprocessing
# * I'm dropping all features but "item_cnt_day" because I'll be using only it as a univariate time-series.
# * We are asked to predict total sales for every product and store in the next month, and our data is given by day, so let's aggregate the data by month.
# * Also I'm leaving only monthly "item_cnt" >= 0 and <= 20, as this seems to be the distributions of the test set.

# In[ ]:


train_monthly = train[['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day']]
train_monthly = train_monthly.sort_values('date').groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)
train_monthly = train_monthly.agg({'item_cnt_day':['sum']})
train_monthly.columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt']
train_monthly = train_monthly.query('item_cnt >= 0 and item_cnt <= 20')
# Label
train_monthly['item_cnt_month'] = train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_id'])['item_cnt'].shift(-1)
train_monthly.head(10).T


# In[ ]:


train_monthly.describe().T


# ### Time-series processing
# * As I only need the "item_cnt" feature as a series, I can get that easily by just using a pivot operation.
# * This way I'll also get the missing months from each "shop_id" and "item_id", and then replace them with 0 (otherwise would be "nan"). 

# In[ ]:


monthly_series = train_monthly.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num',values='item_cnt', fill_value=0).reset_index()
monthly_series.head()


# #### Currently I have one series (33 months) for each unique pair of "shop_id" and "item_id", but probably would be better to have multiple smaller series for each unique pair, so I'm generating multiple series of size 12 (one year) for each unique pair.

# In[ ]:


first_month = 20
last_month = 33
serie_size = 12
data_series = []

for index, row in monthly_series.iterrows():
    for month1 in range((last_month - (first_month + serie_size)) + 1):
        serie = [row['shop_id'], row['item_id']]
        for month2 in range(serie_size + 1):
            serie.append(row[month1 + first_month + month2])
        data_series.append(serie)

columns = ['shop_id', 'item_id']
[columns.append(i) for i in range(serie_size)]
columns.append('label')

data_series = pd.DataFrame(data_series, columns=columns)
data_series.head()


# #### Dropping identifier columns as we don't need them anymore.

# In[ ]:


data_series = data_series.drop(['item_id', 'shop_id'], axis=1)


# ### Train and validation sets.

# In[ ]:


labels = data_series['label']
data_series.drop('label', axis=1, inplace=True)
train, valid, Y_train, Y_valid = train_test_split(data_series, labels.values, test_size=0.10, random_state=0)


# In[ ]:


print("Train set", train.shape)
print("Validation set", valid.shape)
train.head()


# ### Reshape data.
# * Time-series shape **(data points, time-steps, features)**.

# In[ ]:


X_train = train.values.reshape((train.shape[0], train.shape[1], 1))
X_valid = valid.values.reshape((valid.shape[0], valid.shape[1], 1))
print("Train set reshaped", X_train.shape)
print("Validation set reshaped", X_valid.shape)


# #### First let's begin with how a regular RNN time-series approach could be.
# 
# ### Regular LSTM model.

# In[ ]:


serie_size =  X_train.shape[1] # 12
n_features =  X_train.shape[2] # 1

epochs = 15
batch = 64
lr = 0.0001

lstm_model = Sequential()
lstm_model.add(LSTM(10, input_shape=(serie_size, n_features), return_sequences=True))
lstm_model.add(LSTM(6, activation='relu', return_sequences=True))
lstm_model.add(LSTM(1, activation='relu'))
lstm_model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
lstm_model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
lstm_model.add(Dense(1))
lstm_model.summary()

adam = optimizers.Adam(lr)
lstm_model.compile(loss='mse', optimizer=adam)
plot_model(lstm_model, show_shapes=True, to_file='regular_lstm.png')


# In[ ]:


lstm_history = lstm_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, validation_data=(X_valid, Y_valid), verbose=2)


# ### Autoencoder
# * Now we will build an autoencoder to learn how to reconstruct the input, this way it internally learns the best way to represent the input in lower dimensions.
# * The reconstruct model is composed of an encoder and a decoder, the encoder is responsible for learning how to represent the input into lower dimensions and the decoder learns how to rebuild the smaller representations into the input again.
# * Here is a structural representations of an autoencoder:
#  <img src="https://raw.githubusercontent.com/dimitreOliveira/MachineLearning/master/Kaggle/Predict%20Future%20Sales/Autoencoder_structure.png" width="400">
# * After the models is trained we can keep only the encoder part and we'll have a model that is able to do what we want.
# 
# ### LSTM Autoencoder.

# In[ ]:


encoder_decoder = Sequential()
encoder_decoder.add(LSTM(serie_size, activation='relu', input_shape=(serie_size, n_features), return_sequences=True))
encoder_decoder.add(LSTM(6, activation='relu', return_sequences=True))
encoder_decoder.add(LSTM(1, activation='relu'))
encoder_decoder.add(RepeatVector(serie_size))
encoder_decoder.add(LSTM(serie_size, activation='relu', return_sequences=True))
encoder_decoder.add(LSTM(6, activation='relu', return_sequences=True))
encoder_decoder.add(TimeDistributed(Dense(1)))
encoder_decoder.summary()

adam = optimizers.Adam(lr)
encoder_decoder.compile(loss='mse', optimizer=adam)


# In[ ]:


encoder_decoder_history = encoder_decoder.fit(X_train, X_train, epochs=epochs, batch_size=batch, verbose=2)


# #### You should be aware that the better the autoencoder is able to reconstruct the input the better it internally encodes the input, in other words if we have a good autoencoder we probably will have an equally good encoder.
# #### Let's take a look at the layers of the encoder_decoder model:

# In[ ]:


rpt_vector_layer = Model(inputs=encoder_decoder.inputs, outputs=encoder_decoder.layers[3].output)
time_dist_layer = Model(inputs=encoder_decoder.inputs, outputs=encoder_decoder.layers[5].output)
encoder_decoder.layers


# #### About the autoencoder layers
# #### LSTM
# * This is just a regular LSTM layer, a layer that is able to receive sequence data and learn based on it nothing much to talk about.

# #### RepeatVector layer
# * Here is something we don't usually see, this layers basically repeats it's input "n" times, the reason to use it is because the last layers from the encoder part (the layer with one neuron) don't return sequences, so it does not outputs a sequenced data, this way we can't just add another LSTM layer after it, we need a way to turn this output into a sequence of the same time-steps of the model input, this is where "RepeatVector" layers comes in.
# * Let's see what it outputs.

# In[ ]:


rpt_vector_layer_output = rpt_vector_layer.predict(X_train[:1])
print('Repeat vector output shape', rpt_vector_layer_output.shape)
print('Repeat vector output sample')
print(rpt_vector_layer_output[0])


# As you can see this is just the same value repeated some times to match the same shape of the model input.
# 
# #### TimeDistributed layer
# * This layer is more common, sometimes is used when you want to mix RNN layers with other kind of layers.
# * We could output the model with another LSTM layer with one neuron and "return_sequences=True" parameter, but using a "TimeDistributed" layer wrapping a "Dense" layer we will have the same weights for each outputted time-step.

# In[ ]:


time_dist_layer_output = time_dist_layer.predict(X_train[:1])
print('Time distributed output shape', time_dist_layer_output.shape)
print('Time distributed output sample')
print(time_dist_layer_output[0])


# [Another good explanation about the used layers](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)

# #### Defining the encoding model.
# * What I want is to encode the whole series into a single value, so I need the output from the layer with a single neuron (in this case it's the third LSTM layer).
# * I'll take only the encoding part of the model and define it as a new one.

# In[ ]:


encoder = Model(inputs=encoder_decoder.inputs, outputs=encoder_decoder.layers[2].output)
plot_model(encoder_decoder, show_shapes=True, to_file='encoder_decoder_reconstruct_lstm.png')
plot_model(encoder, show_shapes=True, to_file='encoder_lstm.png')


# #### Now let's encode the train and validation time-series.

# In[ ]:


train_encoded = encoder.predict(X_train)
validation_encoded = encoder.predict(X_valid)
print('Encoded time-series shape', train_encoded.shape)
print('Encoded time-series sample', train_encoded[0])


# #### Add new encoded features to the train and validation sets.

# In[ ]:


train['encoded'] = train_encoded
train['label'] = Y_train

valid['encoded'] = validation_encoded
valid['label'] = Y_valid

train.head(10)


# #### Now we can use the new encoded feature that is a representation of the whole time-series and train a "less complex" model that does not receives sequenced data as input.
# 
# ### MLP with LSTM encoded feature
# * For the MLP model I'm only using the current month "item_count" and the encoded time-series feature from our LSTM encoder model, the idea is that we won't need the whole series because we already have a column that represents the whole series into a single value (it's like a dimensionality reduction).

# In[ ]:


last_month = serie_size - 1
Y_train_encoded = train['label']
train.drop('label', axis=1, inplace=True)
X_train_encoded = train[[last_month, 'encoded']]

Y_valid_encoded = valid['label']
valid.drop('label', axis=1, inplace=True)
X_valid_encoded = valid[[last_month, 'encoded']]

print("Train set", X_train_encoded.shape)
print("Validation set", X_valid_encoded.shape)


# In[ ]:


X_train_encoded.head()


# In[ ]:


mlp_model = Sequential()
mlp_model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu', input_dim=X_train_encoded.shape[1]))
mlp_model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
mlp_model.add(Dense(1))
mlp_model.summary()

adam = optimizers.Adam(lr)
mlp_model.compile(loss='mse', optimizer=adam)
plot_model(mlp_model, show_shapes=True, to_file='mlp.png')


# In[ ]:


mlp_history = mlp_model.fit(X_train_encoded.values, Y_train_encoded.values, epochs=epochs, batch_size=batch, validation_data=(X_valid_encoded, Y_valid_encoded), verbose=2)


# ### Comparing models
# * As you can see I tried to build both models with a similar topology (type/number of layers and neurons), so it could make more sense to compare them.
# * The results are pretty close, also they may change a bit depending on the random initialization of the networks weights, so I would say they are very similar in terms of performance.
# 
# #### Model training

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(22,7))

ax1.plot(lstm_history.history['loss'], label='Train loss')
ax1.plot(lstm_history.history['val_loss'], label='Validation loss')
ax1.legend(loc='best')
ax1.set_title('Regular LSTM')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('MSE')

ax2.plot(mlp_history.history['loss'], label='Train loss')
ax2.plot(mlp_history.history['val_loss'], label='Validation loss')
ax2.legend(loc='best')
ax2.set_title('MLP with LSTM encoder')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('MSE')

plt.show()


# #### Regular LSTM on train and validation.

# In[ ]:


lstm_train_pred = lstm_model.predict(X_train)
lstm_val_pred = lstm_model.predict(X_valid)
print('Train rmse:', np.sqrt(mean_squared_error(Y_train, lstm_train_pred)))
print('Validation rmse:', np.sqrt(mean_squared_error(Y_valid, lstm_val_pred)))


# #### MLP with LSTM encoder on train and validation.

# In[ ]:


mlp_train_pred2 = mlp_model.predict(X_train_encoded.values)
mlp_val_pred2 = mlp_model.predict(X_valid_encoded.values)
print('Train rmse:', np.sqrt(mean_squared_error(Y_train_encoded, mlp_train_pred2)))
print('Validation rmse:', np.sqrt(mean_squared_error(Y_valid_encoded, mlp_val_pred2)))


# ### Built test set
# #### Since we have two models I'll build test sets to apply on both of them.

# In[ ]:


latest_records = monthly_series.drop_duplicates(subset=['shop_id', 'item_id'])
X_test = pd.merge(test, latest_records, on=['shop_id', 'item_id'], how='left', suffixes=['', '_'])
X_test.fillna(0, inplace=True)
X_test.drop(['ID', 'item_id', 'shop_id'], axis=1, inplace=True)
X_test.head()


# ### Regular LSTM model test predictions
# * For the regular LSTM model we just need the last 12 months, because that's our series input size.

# In[ ]:


X_test = X_test[[(i + (34 - serie_size)) for i in range(serie_size)]]
X_test.head()


# ### Reshape data.
# * Time-series shape **(data points, time-steps, features)**.

# In[ ]:


X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
print(X_test_reshaped.shape)


# #### Making predictions.

# In[ ]:


lstm_test_pred = lstm_model.predict(X_test_reshaped)


#  ### MLP with LSTM encoded feature test predictions
# * For the MLP model with the encoded features I'm only using the current month "item_count" and the encoded time-series feature from our LSTM encoder model.
# 
# #### Encoding the time-series

# In[ ]:


test_encoded = encoder.predict(X_test_reshaped)


# #### Add encoded features to the test set.

# In[ ]:


X_test['encoded'] = test_encoded
X_test.head()


# In[ ]:


X_test_encoded = X_test[[33, 'encoded']]
print("Train set", X_test_encoded.shape)
X_test_encoded.head()


# #### Making predictions.

# In[ ]:


mlp_test_pred = mlp_model.predict(X_test_encoded)


# #### Predictions from the regular LSTM model.

# In[ ]:


lstm_prediction = pd.DataFrame(test['ID'], columns=['ID'])
lstm_prediction['item_cnt_month'] = lstm_test_pred.clip(0., 20.)
lstm_prediction.to_csv('lstm_submission.csv', index=False)
lstm_prediction.head(10)


# #### Predictions from the MLP model with LSTM encodded feature .

# In[ ]:


mlp_prediction = pd.DataFrame(test['ID'], columns=['ID'])
mlp_prediction['item_cnt_month'] = mlp_test_pred.clip(0., 20.)
mlp_prediction.to_csv('mlp_submission.csv', index=False)
mlp_prediction.head(10)


# Just a disclaimer, you absolutely can get better results on any of the used models,  I did not spent too much time tuning the models hyper parameters, as this is just for demonstration purpose, so if you want to give the code a try, you should surely tune a little more, if you get better results or any good insight about the models or architecture please let me know.
# 
# If you want to check out some interesting different approaches on time series problems take a look at this kernel [Deep Learning for Time Series Forecasting](https://www.kaggle.com/dimitreoliveira/deep-learning-for-time-series-forecasting).
