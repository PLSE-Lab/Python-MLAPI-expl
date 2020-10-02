#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Libraries Imported
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import keras

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# # A glance at the data

# In[ ]:


df_train = pd.read_csv('/kaggle/input/carsales/train.csv')
df_test = pd.read_csv('/kaggle/input/carsales/test.csv')


# In[ ]:


display(df_train.info())
display(df_train.head())
display(df_train.describe())


# Since, date is a string, better to convert into datetime format for better analysis.

# In[ ]:


df_train['date'] = pd.to_datetime(df_train['date'], format = '%d/%m/%Y')
df_test['date'] = pd.to_datetime(df_test['date'], format = '%d/%m/%Y')


# In[ ]:


print('Minimum date from training set: {}'.format(df_train['date'].min()))
print('Maximum date from training set: {}'.format(df_train['date'].max()))


# In[ ]:


print('Minimum date from test set: {}'.format(df_test['date'].min()))
print('Maximum date from test set: {}'.format(df_test['date'].max()))


# # Exploratory Data Analysis

# In[ ]:


plt.figure(figsize=(12,6))
ax = sns.lineplot(x="date", y="private_orders", data=df_train)
ax.set_title('Private Orders per Day', fontsize=30)
ax.set_xlabel('Date', fontsize=20)
ax.set_ylabel('Number', fontsize=20)
plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))
ax = sns.lineplot(x="week_id", y="private_orders", data=df_train)
ax.set_title('Private Orders per Day', fontsize=30)
ax.set_xlabel('Week', fontsize=20)
ax.set_ylabel('Number', fontsize=20)
plt.tight_layout()
plt.show()


# >On studying the variation of number of private orders on both granularities: daily and weekly, It can be concluded that it has seasonality and noise. It doen't seem to have some sort of trend.

# In[ ]:


sns.heatmap(df_train.corr(), cmap='YlOrRd', annot_kws={'size': 20});


# > The private_orders seems to be most positively correlated with sessions, total_configs_finished and dealer_locator. private_orders don't seem to be negatively correlated with none of the features.

# # Preparing the data

# In[ ]:


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[ ]:


#convert to an array
values = df_train.iloc[:,2:].values
#convert all columns to float
#values.astype('float32')
#normalize featues
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 7, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[range(56,63)], axis=1, inplace=True)
print(reframed.head())


# # Defining and Fitting the model

# Since, we have the training data for 79 weeks. Lets, take 64 weeks of data for training the model and remaining 15 weeks for finding the best parameters(Cross-validation).

# In[ ]:


# split into train and test sets
values = reframed.values
n_train_days = 7*64 
train = values[:n_train_days, :]
dev = values[n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
dev_X, dev_y = dev[:, :-1], dev[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 7, 8))
dev_X = dev_X.reshape((dev_X.shape[0], 7, 8))
print(train_X.shape, train_y.shape, dev_X.shape, dev_y.shape)


# For my model, I'll be an LSTM with 100 neurons in the first layer and a single neuron in the output layer for predicting the number of private orders per day. The model will be fit for 100 training epochs with a batch size of 128.
# 
# >All these values for epochs, batch_size and mae loss function are not taken randomly. I trained my model on multiple values and after hit and trial these are the best values i could get.

# In[ ]:


# design network
model = keras.models.Sequential()
model.add(keras.layers.LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(keras.layers.Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=20, batch_size=128, validation_data=(dev_X, dev_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='dev')
plt.legend()
plt.show()


# # Predicting the values

# Since, We are using the past day to predict the next day private orders, I'll add the last record from training set to help in predicting the values in the test set.

# In[ ]:


#take a backup for the submission file
submit = df_test.copy()
#Insert all zeroes for the values to be predicted.
df_test.insert(loc=9, column='private_orders', value=0)
df = df_test.append(df_train[-7:], ignore_index=True)
df.sort_values('week_id',inplace=True)
df = df.reset_index(drop=True)


# Apply the same transformations to the test data as applied to the training data.

# In[ ]:


#convert to an array
values_test = df.iloc[:,2:].values
#convert all columns to float
#values_test.astype('float32')
#normalize featues
scaler_test = MinMaxScaler(feature_range=(0,1))
scaled_test = scaler_test.fit_transform(values_test)
# frame as supervised learning
reframed_test = series_to_supervised(scaled_test, 7, 1)
# drop columns we don't want to predict
reframed_test.drop(reframed_test.columns[range(56,63)], axis=1, inplace=True)
print(reframed_test.head())


# In[ ]:


values_test = reframed_test.values
# split into input and outputs
test_X, test_y = values_test[:, :-1], values_test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
test_X = test_X.reshape((test_X.shape[0], 7, 8))
print(test_X.shape, test_y.shape)


# In[ ]:


# make a prediction
yhat = model.predict(test_X)


# In[ ]:


test_X = test_X.reshape((test_X.shape[0], 7*8))
test_X.shape


# In[ ]:


# invert scaling for forecast
inv_yhat = np.concatenate((test_X[:, -8:-1],yhat), axis=1)
inv_test = scaler_test.inverse_transform(inv_yhat)
inv_yhat = inv_test[:,-1]


# In[ ]:


np.around(inv_yhat)


# In[ ]:


#map all the values to integers
y = list(map(int,inv_yhat))


# In[ ]:


#Insert the predicted values
submit.insert(loc=9, column='private_orders', value=y)


# In[ ]:


#download the csv
submit.to_csv('csv_to_submit.csv', index = False)

