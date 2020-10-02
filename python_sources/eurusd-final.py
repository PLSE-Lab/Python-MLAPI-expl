#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook trains a LSTM model that predicts the bid price of **EURUSD** 15 minutes in the future by looking at last five hours of data. While there is no requirement for the input to be contiguous, it's been empirically observed that having the contiguous input does improve the accuracy of the model. I suspect that having *day of the week* and *hour of the day* as the features mitigates some of the seasonality and contiguousness problems.
# 
# **Disclaimer**: This exercise has been carried out using a small sample data which only contains 14880 samples (2015-12-29 00:00:00 to 2016-05-31 23:45:00) and lacks ASK prices. Which restricts the ability for the model to approach a better accuracy. 
# 
# **Improvements**
# * To tune the model further, I would recommend having at least 5 years worth of data, have ASK price (so that you can compute the spread), and increasing the epoch to 3000.
# * Adding more cross-axial features. Such as *spread*.
# * If you are looking into classification approach (PASS, BUY, SELL), consider adding some technical indicators that is more sensitive to more recent data. 
# * Consider adding non-numerical data, e.g. news, Tweets. The catch is that you have to get the data under one minute for trading, otherwise the news will be reflected before you even make a trade. If anybody knows how to get the news streamed really fast, please let me know.
# 
# **Credits** : Dave Y. Kim, Mahmoud Elsaftawy,

# In[ ]:


import numpy as np
import pandas as pd
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


# Load sample data
#df = pd.read_csv('../input/EURUSD_Candlestick_15_M_BID_31.12.2016-23.08.2019.csv.csv')
df = pd.read_csv('../input/eurusd1h/EURUSD_Candlestick_1_Hour_BID_31.12.2009-30.08.2019.csv')


# In[ ]:


df.head()


# In[ ]:


df.index.min(), df.index.max()


# In[ ]:


# FULL DATA (takes too long)
# df = pd.read_csv('../input/EURUSD_15m_BID_01.01.2010-31.12.2016.csv')


# In[ ]:


# Rename bid OHLC columns
from datetime import datetime, timedelta
df.rename(columns={'Gmt time' : 'timestamp', 'Open' : 'open', 'Close' : 'close', 
                   'High' : 'high', 'Low' : 'low', 'Close' : 'close', 'Volume' : 'volume'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)  - timedelta(hours=4,minutes=0,seconds=0)

df.set_index('timestamp', inplace=True)
df = df.astype(float)
df.head()
from datetime import datetime, timedelta
df['hour'] = df.index.hour
df['day']  = df.index.weekday
df['week'] = df.index.week
df['momentum']  = df['volume'] * (df['open'] - df['close'])

df['avg_price'] = (df['low'] + df['high'])/2
df['range']     = df['high'] - df['low']
df['ohlc_price'] = (df['low'] + df['high'] + df['open'] + df['close'])/4
df['oc_diff']    = df['open'] - df['close']
df.head()


# In[ ]:


def heikin_ashi(df):
    heikin_ashi_df = pd.DataFrame(index=df.index.values, columns=['open', 'high', 'low', 'close'])
    
    heikin_ashi_df['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    for i in range(len(df)):
        if i == 0:
            heikin_ashi_df.iat[0, 0] = df['open'].iloc[0]
        else:
            heikin_ashi_df.iat[i, 0] = (heikin_ashi_df.iat[i-1, 0] + heikin_ashi_df.iat[i-1, 3]) / 2
        
    heikin_ashi_df['high'] = heikin_ashi_df.loc[:, ['open', 'close']].join(df['high']).max(axis=1)
    
    heikin_ashi_df['low'] = heikin_ashi_df.loc[:, ['open', 'close']].join(df['low']).min(axis=1)
    
    return heikin_ashi_df

df1=heikin_ashi(df)

df1['volume']=df['volume']
df1['ema8'] = pd.Series.ewm(df1['close'], span=8).mean()
df1['ema20'] = pd.Series.ewm(df1['close'], span=20).mean()
df1['ema50'] = pd.Series.ewm(df1['close'], span=50).mean()
df1['ema200'] = pd.Series.ewm(df1['close'], span=200).mean()

df1['hour'] = df1.index.hour
df1['day']  = df1.index.weekday
df1['week'] = df1.index.week
df1['momentum']  = df1['volume'] * (df1['open'] - df1['close'])

df1['avg_price'] = (df1['low'] + df1['high'])/2
df1['range']     = df1['high'] - df1['low']
df1['ohlc_price'] = (df1['low'] + df1['high'] + df1['open'] + df1['close'])/4
df1['oc_diff']    = df1['open'] - df1['close']

df1.tail(50)
df_test=df1[-30:]
df_test.head(30)
df1.dropna(how='any', inplace=True)


# In[ ]:


# Add PCA as a feature instead of for reducing the dimensionality. This improves the accuracy a bit.
from sklearn.decomposition import PCA

dataset = df.copy().values.astype('float32')
pca_features = df.columns.tolist()

pca = PCA(n_components=1)
df['pca'] = pca.fit_transform(dataset)


# In[ ]:


import matplotlib.colors as colors
import matplotlib.cm as cm
import pylab

plt.figure(figsize=(10,5))
norm = colors.Normalize(df['ohlc_price'].values.min(), df['ohlc_price'].values.max())
color = cm.viridis(norm(df['ohlc_price'].values))
plt.scatter(df['ohlc_price'].values, df['pca'].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)
plt.title('ohlc_price vs pca')
plt.show()

plt.figure(figsize=(10,5))
norm = colors.Normalize(df['volume'].values.min(), df['volume'].values.max())
color = cm.viridis(norm(df['volume'].values))
plt.scatter(df['volume'].values, df['pca'].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)
plt.title('volume vs pca')
plt.show()

plt.figure(figsize=(10,5))
norm = colors.Normalize(df['ohlc_price'].values.min(), df['ohlc_price'].values.max())
color = cm.viridis(norm(df['ohlc_price'].values))
plt.scatter(df['ohlc_price'].shift().values, df['pca'].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)
plt.title('ohlc_price - 15min future vs pca')
plt.show()

plt.figure(figsize=(10,5))
norm = colors.Normalize(df['volume'].values.min(), df['volume'].values.max())
color = cm.viridis(norm(df['volume'].values))
plt.scatter(df['volume'].shift().values, df['pca'].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)
plt.title('volume - 15min future vs pca')
plt.show()


# As observed above, using PCA shows data seperability that somehwat clusters the data into different price groups.

# In[ ]:


df=df1
df.head(40)
df.to_pickle("eurusd.pkl")
import os
print(os.getcwd())


# In[ ]:


os.listdir()


# In[ ]:





# In[ ]:


def create_dataset(dataset, look_back=60):
    dataX,dataY = [],[]
    
    for i in range(len(dataset)-look_back-1-7):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back+7])
    return np.array(dataX), np.array(dataY)


# In[ ]:





# # Doing a bit of features analysis

# In[ ]:





# In[ ]:


colormap = plt.cm.inferno
plt.figure(figsize=(15,15))
plt.title('Pearson correlation of features', y=1.05, size=15)
sns.heatmap(df.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()

plt.figure(figsize=(15,5))
corr = df.corr()
sns.heatmap(corr[corr.index == 'close'], linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True);


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# Scale and create datasets
target_index = df1.columns.tolist().index('close')
dataset = df1.values.astype('float32')


#   a = dataset[i:(i+look_back)
#         dataX.append(a)
#         dataY.append(dataset[i + look_back+8])
# # Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Set look_back to 20 which is 5 hours (15min*20)
X, y = create_dataset(dataset, look_back=1)
y = y[:,target_index]
X = np.reshape(X, (X.shape[0], X.shape[2]))


# In[ ]:





# In[ ]:


forest = RandomForestRegressor(n_estimators = 100)
forest = forest.fit(X, y)


# In[ ]:


importances = forest.feature_importances_
std = np.std([forest.feature_importances_ for forest in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

column_list = df.columns.tolist()
print("Feature ranking:")
for f in range(X.shape[1]-1):
    print("%d. %s %d (%f)" % (f, column_list[indices[f]], indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(20,10))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="salmon", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# # Exploration

# In[ ]:


# ax = df1.plot(x=df1.index, y='close', c='red', figsize=(40,10))
# index = [str(item) for item in df1.index]
# plt.fill_between(x=index, y1='low',y2='high', data=df1, alpha=0.4)
# plt.show()
# print(df1.index)



# p = df[:200].copy()
# ax = p.plot(x=p.index, y='close', c='red', figsize=(40,10))
# index = [str(item) for item in p.index]
# plt.fill_between(x=index, y1='low', y2='high', data=p, alpha=0.4)
# plt.title('zoomed, first 200')
# plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
df1.head()


# In[ ]:


df1['ema20'].plot(figsize=(100,8))


# In[ ]:


# ax = df.plot(x=df.index, y='close', c='red', figsize=(40,10))
# index = [str(item) for item in df.index]
# plt.fill_between(x=index, y1='low',y2='high', data=df, alpha=0.4)
# plt.show()

# p = df[:200].copy()
# ax = p.plot(x=p.index, y='close', c='red', figsize=(40,10))
# index = [str(item) for item in p.index]
# plt.fill_between(x=index, y1='low', y2='high', data=p, alpha=0.4)
# plt.title('zoomed, first 200')
# plt.show()


# In[ ]:


# Scale and create datasets
target_index = df.columns.tolist().index('close')
high_index = df.columns.tolist().index('high')
low_index = df.columns.tolist().index('low')
dataset = df.values.astype('float32')

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Create y_scaler to inverse it later
y_scaler = MinMaxScaler(feature_range=(0, 1))
t_y = df['close'].values.astype('float32')
t_y = np.reshape(t_y, (-1, 1))
y_scaler = y_scaler.fit(t_y)

# Set look_back to 20 which is 5 hours (15min*20)
X, y = create_dataset(dataset, look_back=20)
y = y[:,target_index]
X.shape[2]
X.shape[1]


# In[ ]:


# Set training data size
# We have a large enough dataset. So divid into 98% training / 1%  development / 1% test sets
train_size = int(len(X) * 0.95)
print(train_size)
print(len(X))
trainX = X[:train_size]
trainY = y[:train_size]

testX = X[train_size:]
testY = y[train_size:]
print(len(testX))
print(len(testY))
X.shape[2]




# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, LSTM, Dense


# create a small LSTM network
model = Sequential()
model.add(LSTM(290, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(49, return_sequences=True))
model.add(LSTM(49, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(20, return_sequences=False))
model.add(Dense(14, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
print(model.summary())


# In[ ]:


# Save the best weight during training.
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("EURUSD_7.best.hdf5", monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='max')

# Fit
callbacks_list = [checkpoint]
history = model.fit(trainX, trainY, epochs=20, batch_size=100, verbose=1, callbacks=callbacks_list, validation_split=0.1)


# In[ ]:


epoch = len(history.history['loss'])
for k in list(history.history.keys()):
    if 'val' not in k:
        plt.figure(figsize=(40,10))
        plt.plot(history.history[k])
        plt.plot(history.history['val_' + k])
        plt.title(k)
        plt.ylabel(k)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


# In[ ]:


min(history.history['val_mean_absolute_error'])


# As seen from the above, the model seems to have converged nicely, but the mean absolute error on the development data remains at ~0.003X which means the model is unusable in practice. Ideally, we want to get ~0.0005. Let's go back to the best weight, and decay the learning rate while retraining the model

# In[ ]:


# Baby the model a bit
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, LSTM, Dense


# create a small LSTM network
model = Sequential()
model.add(LSTM(290, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(49, return_sequences=True))
model.add(LSTM(49, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(20, return_sequences=False))
model.add(Dense(14, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
print(model.summary())
# Load the weight that worked the best
model.save("EURUSD_20.best.hdf5")
model.load_weights("EURUSD_20.best.hdf5")

# Train again with decaying learning rate
from keras.callbacks import LearningRateScheduler
import keras.backend as K

def scheduler(epoch):
    if epoch%10==0 and epoch!=0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*.9)
        print("lr changed to {}".format(lr*.9))
    return K.get_value(model.optimizer.lr)
lr_decay = LearningRateScheduler(scheduler)

callbacks_list = [checkpoint, lr_decay]
history = model.fit(trainX, trainY, epochs=int(1005), batch_size=32, verbose=1, callbacks=callbacks_list, validation_split=0.1)


# In[ ]:


min(history.history['val_mean_absolute_error'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


epoch = len(history.history['loss'])
for k in list(history.history.keys()):
    if 'val' not in k:
        plt.figure(figsize=(40,10))
        plt.plot(history.history[k])
        plt.plot(history.history['val_' + k])
        plt.title(k)
        plt.ylabel(k)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


# In[ ]:


min(history.history['val_mean_absolute_error'])


# The variance should have improved slightly. However, unless the mean absolute error is not small enough. The model is still not an usable model in practice. This is mainly due to only using the sample data for training and limiting epoch to a few hundreds.

# # Visually compare the delta between the prediction and actual (scaled values)

# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Benchmark
model.load_weights("EURUSD_20.best.hdf5")

pred = model.predict(testX)

predictions = pd.DataFrame()
predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))
predictions['actual'] = testY
predictions = predictions.astype(float)

predictions.plot(figsize=(20,10))
plt.show()

predictions['diff'] = predictions['predicted'] - predictions['actual']
plt.figure(figsize=(10,10))
sns.distplot(predictions['diff']);
plt.title('Distribution of differences between actual and prediction')
plt.show()

print("MSE : ", mean_squared_error(predictions['predicted'].values, predictions['actual'].values))
print("MAE : ", mean_absolute_error(predictions['predicted'].values, predictions['actual'].values))
predictions['diff'].describe()


# In[ ]:


predictions = pd.DataFrame()
predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))
predictions['actual'] = testY
predictions = predictions.astype(float)
import plotly.express as px

import pandas as pd

fig = px.line(predictions, )
fig.show()


# # Compare the unscaled values and see if the prediction falls within the Low and High

# In[ ]:


print (testX
      )


# In[ ]:


pred = model.predict(testX)
pred = y_scaler.inverse_transform(pred)
close = y_scaler.inverse_transform(np.reshape(testY, (testY.shape[0], 1)))

predictions = pd.DataFrame()
predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))
predictions['close'] = pd.Series(np.reshape(close, (close.shape[0])))
print (predictions['predicted'] )
predictions.tail()


p = df1[-pred.shape[0]:].copy()
predictions.index = p.index
predictions = predictions.astype(float)
predictions.tail()

# predictions = predictions.merge(p[['low', 'high']], right_index=True, left_index=True)
# predictions.tail()
# ax = predictions.plot(x=predictions.index, y='close', c='red', figsize=(40,10))
# ax = predictions.plot(x=predictions.index, y='predicted', c='blue', figsize=(40,10), ax=ax)
# index = [str(item) for item in predictions.index]
# plt.fill_between(x=index, y1='low', y2='high', data=p, alpha=0.4)
# plt.title('Prediction vs Actual (low and high as blue region)')
# plt.show()

# predictions['diff'] = predictions['predicted'] - predictions['close']
# plt.figure(figsize=(10,10))
# sns.distplot(predictions['diff']);
# plt.title('Distribution of differences between actual and prediction ')
# plt.show()

# g = sns.jointplot("diff", "predicted", data=predictions, kind="kde", space=0)
# plt.title('Distributtion of error and price')
# plt.show()

# predictions['correct'] = (predictions['predicted'] <= predictions['high']) & (predictions['predicted'] >= predictions['low'])
# sns.factorplot(data=predictions, x='correct', kind='count')

# print("MSE : ", mean_squared_error(predictions['predicted'].values, predictions['close'].values))
# print("MAE : ", mean_absolute_error(predictions['predicted'].values, predictions['close'].values))
# predictions['diff'].describe()


# In[ ]:


predictions.head()


# In[ ]:


df_test.info()


# In[ ]:


dataset[0:20]


# In[ ]:


len(df_test)


# testX = X[train_size:]
# testY = y[train_size:]

# In[ ]:



def create_dataset_test(dataset, look_back=20):
    dataX = []
    for i in range(11):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
    return np.array(dataX)

# Scale and create datasets
target_index = df_test.columns.tolist().index('close')
high_index = df_test.columns.tolist().index('high')
low_index = df_test.columns.tolist().index('low')
dataset = df_test.values.astype('float32')



scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Create y_scaler to inverse it later
# y_scaler = MinMaxScaler(feature_range=(0, 1))
# t_y = df_test['close'].values.astype('float32')
# t_y = np.reshape(t_y, (-1, 1))
# y_scaler = y_scaler.fit(t_y)

pred = y_scaler.inverse_transform(pred)

# Set look_back to 20 which is 5 hours (15min*20)
X_predict= create_dataset_test(dataset, look_back=20)

X.shape[0]


# In[ ]:


close = y_scaler.inverse_transform(np.reshape(testY, (testY.shape[0], 1)))


# In[ ]:


df_test.head(30)


# In[ ]:


# Scale and create datasets
target_index = df_test.columns.tolist().index('close')
high_index = df_test.columns.tolist().index('high')
low_index = df_test.columns.tolist().index('low')
dataset = df_test.values.astype('float32')

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Create y_scaler to inverse it later
y_scaler = MinMaxScaler(feature_range=(0, 1))
t_y = df_test['close'].values.astype('float32')
t_y = np.reshape(t_y, (-1, 1))
y_scaler = y_scaler.fit(t_y)

# Set look_back to 20 which is 5 hours (15min*20)
X= create_dataset_test(dataset, look_back=20)
X.shape


# In[ ]:


df_test.tail(30)


# In[ ]:


for i in range(30-20-1):
    print (i)
    


# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Benchmark
model.load_weights("EURUSD.best.hdf5")

pred = model.predict(X_predict)
print(pred)
pred = y_scaler.inverse_transform(pred)
print(pred)
a=pd.DataFrame(pred)
a.plot()


# predictions = pd.DataFrame()
# predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))
# predictions['actual'] = testY
# predictions = predictions.astype(float)

# predictions.plot(figsize=(20,10))
# plt.show()

# predictions['diff'] = predictions['predicted'] - predictions['actual']
# plt.figure(figsize=(10,10))
# sns.distplot(predictions['diff']);
# plt.title('Distribution of differences between actual and prediction')
# plt.show()

# print("MSE : ", mean_squared_error(predictions['predicted'].values, predictions['actual'].values))
# print("MAE : ", mean_absolute_error(predictions['predicted'].values, predictions['actual'].values))
# predictions['diff'].describe()


# In[ ]:


dtest=df_test['close']
dtest = dtest.values.astype('float32')
print(dtest)


# In[ ]:


X=[]
for i in range(30):
    X.append(dtest[i])


# In[ ]:


print(X)

