#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[6]:


bits = pd.read_csv('../input/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
bits.head()


# We will be using weighted price of bitcoin for prediction. Currently the time is in unix timestamp and has data for every minute. Lets average the price over a day and forecast for next 30 days.

# In[7]:


bits['date'] = pd.to_datetime(bits['Timestamp'],unit='s').dt.date
group = bits.groupby('date')
Real_Price = group['Weighted_Price'].mean()


# In[8]:


Real_Price[0:5]


# In[9]:


# split data
prediction_days = 30
df_train= Real_Price[:len(Real_Price)-prediction_days]
df_test= Real_Price[len(Real_Price)-prediction_days:]


# ## One Shot LSTM/RNN

# In[10]:


# hyperparameters
n_steps = 10
nb_epochs=20
lr=0.001


# In[11]:


# Data preprocess
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
test_set = df_test.values
test_set = np.reshape(test_set, (len(test_set), 1))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
test_set = sc.transform(test_set)


# In[12]:


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


# In[13]:


#Prepare batches bs*seq_len*1
X_train,y_train = split_sequence(training_set, n_steps)
print(X_train.shape)
print(len(test_set))
X_test, y_test = split_sequence(test_set, n_steps)
print(X_test.shape)


# In[14]:


import tensorflow as tf


# In[15]:


mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)


# In[16]:


# define model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(50, activation='relu', input_shape=(n_steps, 1)))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[17]:


model.fit(X_train, y_train, epochs=nb_epochs, verbose=1, validation_data=(X_test, y_test),callbacks=[mc, es])


# In[18]:


# load the saved model
saved_model = tf.keras.models.load_model('best_model.h5')
# evaluate the model
train_loss = saved_model.evaluate(X_train, y_train, verbose=0)
test_loss = saved_model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_loss, test_loss))


# ## Inference

# In[19]:


train_predictions = saved_model.predict(X_train)
test_predictions = saved_model.predict(X_test)


# In[20]:


train_p = sc.inverse_transform(train_predictions)
test_p = sc.inverse_transform(test_predictions)


# ## Training fit

# In[21]:


import math
t = np.linspace(0, 2*math.pi, 1425)
plt.plot(t, sc.inverse_transform(y_train),'r')
plt.plot(t,train_p,'g')
plt.show()


# ## Test fit

# In[22]:


t = np.linspace(0, 2*math.pi, 20)
plt.plot(t, sc.inverse_transform(y_test),'r')
plt.plot(t,test_p,'g')
plt.show()


# Seems like the model is overfitting, we can add dropout and gradient clipping to improve results. Lets look at second architecture.

# ## Bidirectional LSTM Model

# In[23]:


# define model
bi_model = tf.keras.models.Sequential()
bi_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, activation='relu'), input_shape=(n_steps, 1)))
bi_model.add(tf.keras.layers.Dense(1))
bi_model.compile(optimizer='adam', loss='mse')


# In[24]:


bi_model.fit(X_train, y_train, epochs=nb_epochs, verbose=1, validation_data=(X_test, y_test),callbacks=[mc, es])


# In[25]:


# load the saved model
saved_model2 = tf.keras.models.load_model('best_model.h5')
# evaluate the model
train_loss = saved_model2.evaluate(X_train, y_train, verbose=0)
test_loss = saved_model2.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_loss, test_loss))


# ## Inference

# In[26]:


train_predictions = saved_model2.predict(X_train)
test_predictions = saved_model2.predict(X_test)


# In[27]:


train_p = sc.inverse_transform(train_predictions)
test_p = sc.inverse_transform(test_predictions)


# ## Training fit

# In[28]:


import math
t = np.linspace(0, 2*math.pi, 1425)
plt.plot(t, sc.inverse_transform(y_train),'r')
plt.plot(t,train_p,'g')
plt.show()


# ## Test fit

# In[29]:


t = np.linspace(0, 2*math.pi, 20)
plt.plot(t, sc.inverse_transform(y_test),'r', label="True label")
plt.plot(t,test_p,'g', label="Predictions")
plt.legend()
plt.show()


# ## Encoder Decoder Model For Multiple Predictions

# ### Data Processing

# First we need to prepare the data in such a way that multiple steps can go in as input and multiple steps can come as output.

# In[30]:


def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


# In[31]:


# hyperparameters
n_steps_out = 3


# In[32]:


#Prepare batches bs*seq_len*1
X_train,y_train = split_sequence(training_set, n_steps, n_steps_out)
print(X_train.shape, y_train.shape)
X_test, y_test = split_sequence(test_set, n_steps, n_steps_out)
print(X_test.shape, y_test.shape)


# In[33]:


# define model
ed_model = tf.keras.models.Sequential()
ed_model.add(tf.keras.layers.LSTM(100, activation='relu', input_shape=(n_steps, 1)))
ed_model.add(tf.keras.layers.RepeatVector(n_steps_out))
ed_model.add(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True))
ed_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
ed_model.compile(optimizer='adam', loss='mse')


# In[34]:


nb_epochs = 50


# In[35]:


history = ed_model.fit(X_train, y_train, epochs=nb_epochs, verbose=1, validation_data=(X_test, y_test),callbacks=[mc, es])


# In[36]:


# plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.title("Training Losses")
plt.show()


# ### Inference For test set

# In[37]:


test_predictions = ed_model.predict(X_test)
test_predictions.shape


# In[38]:


y_pred = test_predictions[:,2,:]


# In[39]:


y_true = y_test[:,2,:]
y_true.shape


# In[40]:


y_pred.shape


# In[41]:


t = np.linspace(0, 2*math.pi, 18)
plt.plot(t, sc.inverse_transform(y_true),'r', label="True Data")
plt.plot(t,sc.inverse_transform(y_pred),'g', label = "Prediction")
plt.legend()
plt.show()


# ## Visualize the model

# In[42]:


ed_model.summary()

