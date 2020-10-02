#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Loading the Data

# In[ ]:


df = pd.read_csv("../input/BSESN.csv")


# In[ ]:


df.head(3)


# # Loading Libraries

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers import TimeDistributed
from keras.layers import Bidirectional


# In[ ]:


import pandas as pd
import numpy as np
from fbprophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#setting figure size
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 20,10

# Plot styles
sns.set_style("whitegrid")
plt.style.use('fivethirtyeight')

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


np.random.seed(42)


# # Basic checks on Data

# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.head(3)


# In[ ]:


df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']


# In[ ]:


plt.figure(figsize=(16,8))
plt.plot(df['Adj Close'], label='Close Price history')


# # Preparing Data for LSTM

# In[ ]:


#setting index as date values
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#sorting
data = df.sort_index(ascending=True, axis=0)

#creating a separate dataset
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Adj Close'][i]


# In[ ]:


new_data.head(3)


# In[ ]:


new_data.index = new_data['Date']
new_data.drop('Date', inplace=True, axis=1)
new_data.head(2)


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(new_data)


# In[ ]:


# 2006-2008 plotting
plt.figure(figsize=(10,5))
plt.plot(new_data.ix['2006':'2010'])


# In[ ]:


# 2007-2009 plotting
plt.figure(figsize=(10,5))
plt.plot(new_data.ix['2007':'2009'])


# # Train and Test

# In[ ]:


train_data = new_data.ix[:'2014']
test_raw_data = new_data.ix['2015':]


# In[ ]:


train_data.head(3)


# In[ ]:


test_raw_data.head(3)


# In[ ]:


train_data.tail(3)


# In[ ]:


test_raw_data.tail(3)


# In[ ]:


train_data.shape, test_raw_data.shape


# # Scaling the Data

# In[ ]:


train_data.head(5)


# In[ ]:


train_data = scaler.fit_transform(train_data[['Close']])
test_data = scaler.fit_transform(test_raw_data[['Close']])


# In[ ]:


train_data


# In[ ]:


scaler.inverse_transform(train_data)


# In[ ]:


print("Shape of train data: " + str(train_data.shape))
print("Shape of test data: " + str(test_data.shape))


# # Univariate Time-series (LSTM)
# 
# 
# 
# In this section, we will review few methods of LSTM and see which one performs the best on the Univariate time-series where only time is the feature in our data.

# In[ ]:


## Create Dataset for LSTM

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    
    return np.array(dataX), np.array(dataY)  


# In[ ]:


look_back = 3

trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)


# In[ ]:


pd.DataFrame(trainX).head(5)


# In[ ]:


pd.DataFrame(trainY).head(5)


# In[ ]:


print("Shape of train input: " + str(trainX.shape))
print("Shape of train labels: " + str(trainY.shape))
print("Shape of test input: " + str(testX.shape))
print("Shape of test labels: " + str(testY.shape))


# In[ ]:


trainX.shape


# In[ ]:


testX.shape


# # LSTM (time-step = 1)

# In[ ]:


## Reshaping the Data for LSTM
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[ ]:


print("Shape of train input: " + str(trainX.shape))
print("Shape of train labels: " + str(trainY.shape))
print("Shape of test input: " + str(testX.shape))
print("Shape of test labels: " + str(testY.shape))


# In[ ]:


get_ipython().run_cell_magic('time', '', "model = Sequential()\nmodel.add(LSTM(100, activation='relu', input_shape=(1, look_back)))\nmodel.add(Dense(1, activation='linear'))\nmodel.compile(loss='mean_squared_error', optimizer='adam')\nmodel.summary()")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(trainX, trainY, epochs=1000, batch_size=5, validation_data=(testX, testY), verbose=2,\n                   shuffle=False)')


# In[ ]:


model.evaluate(testX, testY)


# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[ ]:


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[ ]:


trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[ ]:


from sklearn.metrics import mean_absolute_error

trainScore_1_mae = mean_absolute_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f MAE' % (trainScore_1_mae))
testScore_1_mae = mean_absolute_error(testY[0], testPredict[:,0])
print('Test Score: %.2f MAE' % (testScore_1_mae))


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


trainScore_1_mape = mean_absolute_percentage_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f MAPE' % (trainScore_1_mape))
testScore_1_mape = mean_absolute_percentage_error(testY[0], testPredict[:,0])
print('Test Score: %.2f MAPE' % (testScore_1_mape))


# # LSTM (time-step = no. of columns)

# In[ ]:


## Reshaping the Data for LSTM
trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

print("Shape of train input: " + str(trainX.shape))
print("Shape of train labels: " + str(trainY.shape))
print("Shape of test input: " + str(testX.shape))
print("Shape of test labels: " + str(testY.shape))

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(look_back, 1)))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(trainX, trainY, epochs=1000, batch_size=5, validation_data=(testX, testY),verbose=2, shuffle=False)')


# In[ ]:


model.evaluate(testX, testY)


# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[ ]:


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[ ]:


trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[ ]:


from sklearn.metrics import mean_absolute_error

trainScore_2_mae = mean_absolute_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f MAE' % (trainScore_2_mae))
testScore_2_mae = mean_absolute_error(testY[0], testPredict[:,0])
print('Test Score: %.2f MAE' % (testScore_2_mae))


# In[ ]:


trainScore_2_mape = mean_absolute_percentage_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f MAPE' % (trainScore_2_mape))
testScore_2_mape = mean_absolute_percentage_error(testY[0], testPredict[:,0])
print('Test Score: %.2f MAPE' % (testScore_2_mape))


# # LSTM Stacked

# In[ ]:


## Reshaping the Data for LSTM
trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

print("Shape of train input: " + str(trainX.shape))
print("Shape of train labels: " + str(trainY.shape))
print("Shape of test input: " + str(testX.shape))
print("Shape of test labels: " + str(testY.shape))

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(trainX, trainY, epochs=1000, batch_size=5, validation_data=(testX, testY),verbose=2, shuffle=False)')


# In[ ]:


model.evaluate(testX, testY)


# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[ ]:


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[ ]:


trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[ ]:


from sklearn.metrics import mean_absolute_error

trainScore_3_mae = mean_absolute_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f MAE' % (trainScore_3_mae))
testScore_3_mae = mean_absolute_error(testY[0], testPredict[:,0])
print('Test Score: %.2f MAE' % (testScore_3_mae))


# In[ ]:


trainScore_3_mape = mean_absolute_percentage_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f MAPE' % (trainScore_3_mape))
testScore_3_mape = mean_absolute_percentage_error(testY[0], testPredict[:,0])
print('Test Score: %.2f MAPE' % (testScore_3_mape))


# # Bidirectional LSTM

# In[ ]:


## Reshaping the Data for LSTM
trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

print("Shape of train input: " + str(trainX.shape))
print("Shape of train labels: " + str(trainY.shape))
print("Shape of test input: " + str(testX.shape))
print("Shape of test labels: " + str(testY.shape))

model = Sequential()
model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True),  input_shape=(look_back, 1)))
model.add(Bidirectional(LSTM(50, activation='relu')))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(trainX, trainY, epochs=1000, batch_size=5, validation_data=(testX, testY),verbose=2, shuffle=False)')


# In[ ]:


model.evaluate(testX, testY)


# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[ ]:


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[ ]:


trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[ ]:


from sklearn.metrics import mean_absolute_error

trainScore_4_mae = mean_absolute_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f MAE' % (trainScore_4_mae))
testScore_4_mae = mean_absolute_error(testY[0], testPredict[:,0])
print('Test Score: %.2f MAE' % (testScore_4_mae))


# In[ ]:


trainScore_4_mape = mean_absolute_percentage_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f MAPE' % (trainScore_4_mape))
testScore_4_mape = mean_absolute_percentage_error(testY[0], testPredict[:,0])
print('Test Score: %.2f MAPE' % (testScore_4_mape))


# In[ ]:


type(trainScore_1_mae)


# In[ ]:


mae_score = [trainScore_1_mae, trainScore_2_mae, trainScore_3_mae, trainScore_4_mae]
df_mae_train = pd.DataFrame(mae_score, index=['trainScore_1_mae', 'trainScore_2_mae', 'trainScore_3_mae', 'trainScore_4_mae'], 
            columns =['values'])


# In[ ]:


mape_score = [trainScore_1_mape, trainScore_2_mape, trainScore_3_mape, trainScore_4_mape]
df_mape_train = pd.DataFrame(mape_score, index=['trainScore_1_mape', 'trainScore_2_mape', 'trainScore_3_mape', 'trainScore_4_mape'], 
            columns =['values'])


# In[ ]:


mae_score = [testScore_1_mae, testScore_2_mae, testScore_3_mae, testScore_4_mae]
df_mae_test = pd.DataFrame(mae_score, index=['testScore_1_mae', 'testScore_2_mae', 'testScore_3_mae', 'testScore_4_mae'], 
            columns =['values'])


# In[ ]:


mape_score = [testScore_1_mape, testScore_2_mape, testScore_3_mape, testScore_4_mape]
df_mape_test = pd.DataFrame(mape_score, index=['testScore_1_mape', 'testScore_2_mape', 'testScore_3_mape', 'testScore_4_mape'], 
            columns =['values'])


# In[ ]:


df_all_train = df_mae_train.append(df_mape_train)
df_all_test = df_mae_test.append(df_mape_test)


# In[ ]:


df_all_train


# In[ ]:


df_all_test


# This is clear from above training and test scores that LSTM with timesteps as no. of columns is the best LSTM model (model number 2).
# 
# There are few clear observations:
# 
# 1. No. of time steps as 1 is clearly not a preferable approach.
# 2. Stacked LSTM is performing well than non-stacked LSTM (in terms of closeness between training loss and test loss)
# 3. Bidirection LSTM performance is equally better
# 
# 

# # Multivariate Time Series
# ## Adding Features and Running LSTM model

# In[ ]:


# time-series to supervised (www.machinelearningmastery.com)
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


new_data.head()


# In[ ]:


new_data.shape


# In[ ]:


new_data_date = new_data.copy()
new_data_date.reset_index(inplace=True)


# In[ ]:


# We will create a number of features on the Dates
new_data_date['year'] = new_data_date['Date'].map(lambda x : x.year)
new_data_date['month'] = new_data_date['Date'].map(lambda x : x.month)
new_data_date['day_week'] = new_data_date['Date'].map(lambda x : x.dayofweek)
new_data_date['quarter'] = new_data_date['Date'].map(lambda x : x.quarter)
new_data_date['week'] = new_data_date['Date'].map(lambda x : x.week)
new_data_date['quarter_start'] = new_data_date['Date'].map(lambda x : x.is_quarter_start)
new_data_date['quarter_end'] = new_data_date['Date'].map(lambda x : x.is_quarter_end)
new_data_date['month_start'] = new_data_date['Date'].map(lambda x : x.is_month_start)
new_data_date['month_end'] = new_data_date['Date'].map(lambda x : x.is_month_end)
new_data_date['year_start'] = new_data_date['Date'].map(lambda x : x.is_year_start)
new_data_date['year_end'] = new_data_date['Date'].map(lambda x : x.is_year_end)
new_data_date['week_year'] = new_data_date['Date'].map(lambda x : x.weekofyear)
new_data_date['quarter_start'] = new_data_date['quarter_start'].map(lambda x: 0 if x is False else 1)
new_data_date['quarter_end'] = new_data_date['quarter_end'].map(lambda x: 0 if x is False else 1)
new_data_date['month_start'] = new_data_date['month_start'].map(lambda x: 0 if x is False else 1)
new_data_date['month_end'] = new_data_date['month_end'].map(lambda x: 0 if x is False else 1)
new_data_date['year_start'] = new_data_date['year_start'].map(lambda x: 0 if x is False else 1)
new_data_date['year_end'] = new_data_date['year_end'].map(lambda x: 0 if x is False else 1)
new_data_date['day_month'] = new_data_date['Date'].map(lambda x: x.daysinmonth)
# Create a feature which could be important - Markets are only open between Monday and Friday.
mon_fri_list = [0,4]
new_data_date['mon_fri'] = new_data_date['day_week'].map(lambda x: 1 if x in mon_fri_list  else 0)
# It has been proved in many studies worldwide that winters are better for return on stocks than summers. 
# We will see how true is this in this case.
second_half = [7, 8, 9, 10, 11, 12]
new_data_date['half_year'] = new_data_date['month'].map(lambda x: 1 if x in second_half  else 0)
# Election Years
elec_year = [1998, 1999, 2004, 2009, 2014]
new_data_date['elec_year'] = new_data_date['year'].map(lambda x: 1 if x in elec_year  else 0)


# In[ ]:


new_data_date.head()


# In[ ]:


new_data_date.shape


# In[ ]:


new_data_date.index = new_data_date['Date']
new_data_date.drop('Date', axis=1, inplace=True)
new_data_date.head()


# In[ ]:


new_data_date.shape


# In[ ]:


columns_to_encode = ['year', 'month', 'day_week', 
                    'quarter', 'week',  'week_year', 'day_month']

columns_to_scale  = ['Close']

other_cols = ['quarter_start', 'quarter_end', 'month_start', 'month_end',
                    'year_start', 'year_end', 'mon_fri', 'half_year', 'elec_year']

from sklearn.preprocessing import OneHotEncoder
scaler = MinMaxScaler(feature_range=(0,1))
ohe    = OneHotEncoder(sparse=False)

scaled_columns  = scaler.fit_transform(new_data_date[columns_to_scale]) 
encoded_columns = ohe.fit_transform(new_data_date[columns_to_encode])

rev_new_data = np.concatenate([scaled_columns, new_data_date[other_cols], encoded_columns], axis=1)


# In[ ]:


rev_new_data.shape


# In[ ]:


# specify the number of lag days
n_days = 3
n_features = rev_new_data.shape[1]
# frame as supervised learning
reframed = series_to_supervised(rev_new_data, n_days, 1)


# In[ ]:


reframed.head()


# In[ ]:


reframed.shape


# In[ ]:


rev_new_data.shape


# In[ ]:


cols_to_remove = rev_new_data.shape[1]-1
print(cols_to_remove)
reframed_new = reframed.iloc[:, :-cols_to_remove]


# In[ ]:


reframed_new.head()


# In[ ]:


reframed_new.shape


# In[ ]:


values = reframed_new.values
n_train_time = 365*10

train = values[:n_train_time, :]
test = values[n_train_time:, :]

n_obs = n_days * n_features
train_X, train_y = train[:, :n_obs], train[:, -(n_features+1)]
test_X, test_y = test[:, :n_obs], test[:, -(n_features+1)]
print(train_X.shape, len(train_X), train_y.shape)

#train_X, train_y = train[:, :-1], train[:, -1]
#test_X, test_y = test[:, :-1], test[:, -1]

#train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
#test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


# In[ ]:


pd.DataFrame(train_X)[:5]


# In[ ]:


train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
test_X = test_X.reshape((test_X.shape[0], n_days, n_features))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[ ]:


print(n_obs)
print(n_days)
print(n_features)


# In[ ]:


test_y


# In[ ]:


import keras
from keras.layers import Activation, BatchNormalization
model = Sequential()

#Input
model.add(Bidirectional(LSTM(500, activation='relu', return_sequences=True), 
                        input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(BatchNormalization())
model.add(Dropout(0.3))

# middle
model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True)))
#model.add(BatchNormalization())
model.add(Dropout(0.3))

# middle
model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True)))
#model.add(BatchNormalization())
model.add(Dropout(0.3))

# middle
model.add(Bidirectional(LSTM(50, activation='relu')))
#model.add(BatchNormalization())
model.add(Dropout(0.3))

#Output
model.add(Dense(1, activation='linear'))
#model.add(BatchNormalization())

optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=optimizer)
model.summary()


# In[ ]:


history = model.fit(train_X, train_y, epochs=1500, batch_size=len(train_X), 
                    validation_data=(test_X, test_y), verbose=2, shuffle=False)


# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[ ]:


np.mean(history.history['val_loss'])


# In[ ]:


model.evaluate(test_X, test_y, verbose=2, batch_size=92)

