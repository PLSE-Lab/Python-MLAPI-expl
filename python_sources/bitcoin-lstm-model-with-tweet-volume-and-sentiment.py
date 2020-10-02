#!/usr/bin/env python
# coding: utf-8

# # Bitcoin LSTM Model with Tweet Volume and Sentiment

# In[ ]:


import pandas as pd
import re 
from matplotlib import pyplot
import seaborn as sns
import numpy as np
import os # accessing directory structure


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#set seed
np.random.seed(12345)


# # Data Pre-processing

# In[ ]:


notclean = pd.read_csv('../input/bitcoin-tweets-14m/cleanprep.csv', delimiter=',', error_bad_lines=False,engine = 'python',header = None)


# In[ ]:


notclean.head()


# In[ ]:


#-----------------Pre-processing -------------------#

notclean.columns =['dt', 'name','text','polarity','sensitivity']


# In[ ]:


notclean =notclean.drop(['name','text'], axis=1)


# In[ ]:


notclean.head()


# In[ ]:


notclean.info()


# In[ ]:


notclean['dt'] = pd.to_datetime(notclean['dt'])


# In[ ]:


notclean['DateTime'] = notclean['dt'].dt.floor('h')
notclean.head()


# In[ ]:


vdf = notclean.groupby(pd.Grouper(key='dt',freq='H')).size().reset_index(name='tweet_vol')


# In[ ]:


vdf.head()


# In[ ]:


vdf.info()


# In[ ]:


vdf.index = pd.to_datetime(vdf.index)
vdf=vdf.set_index('dt')


# In[ ]:


vdf.info()


# In[ ]:


vdf.head()


# In[ ]:


notclean.info()


# In[ ]:


notclean.index = pd.to_datetime(notclean.index)


# In[ ]:


notclean.info()


# In[ ]:


vdf['tweet_vol'] =vdf['tweet_vol'].astype(float)


# In[ ]:


vdf.info()


# In[ ]:


notclean.info()


# In[ ]:


notclean.head()


# In[ ]:


#ndf = pd.merge(notclean,vdf, how='inner',left_index=True, right_index=True)


# In[ ]:


notclean.head()


# In[ ]:


df = notclean.groupby('DateTime').agg(lambda x: x.mean())


# In[ ]:


df['Tweet_vol'] = vdf['tweet_vol']


# In[ ]:


df = df.drop(df.index[0])


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.info()


# In[ ]:


btcDF = pd.read_csv('../input/btc-price/btcSave2.csv', error_bad_lines=False,engine = 'python')


# In[ ]:


btcDF['Timestamp'] = pd.to_datetime(btcDF['Timestamp'])
btcDF = btcDF.set_index(pd.DatetimeIndex(btcDF['Timestamp']))


# In[ ]:


btcDF.head()


# In[ ]:


btcDF = btcDF.drop(['Timestamp'], axis=1)


# In[ ]:


btcDF.head()


# In[ ]:


Final_df = pd.merge(df,btcDF, how='inner',left_index=True, right_index=True)


# In[ ]:


Final_df.head()


# In[ ]:


Final_df.info()


# In[ ]:


Final_df=Final_df.drop(['Weighted Price'],axis=1 )


# In[ ]:


Final_df.head()


# In[ ]:


Final_df.columns = ['Polarity', 'Sensitivity','Tweet_vol','Open','High','Low', 'Close_Price', 'Volume_BTC', 'Volume_Dollar']


# In[ ]:


Final_df.head()


# In[ ]:


Final_df = Final_df[['Polarity', 'Sensitivity','Tweet_vol', 'Open','High','Low', 'Volume_BTC', 'Volume_Dollar', 'Close_Price']]


# In[ ]:


Final_df


# In[ ]:


#---------------Stage 1 Complete ------------------#
#Final_df.to_csv('Desktop/Sentiment.csv')


# # Exploratory Analysis

# In[ ]:


#--------------Analysis----------------------------#

values = Final_df.values
groups = [0,1,2,3,4,5,6,7]
i =1  
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1,i)
    pyplot.plot(values[:,group])
    pyplot.title(Final_df.columns[group], y=.5, loc='right')
    i += 1
pyplot.show()


# In[ ]:


Final_df['Volume_BTC'].max()


# In[ ]:


Final_df['Volume_Dollar'].max()


# In[ ]:


Final_df['Volume_BTC'].sum()


# In[ ]:


Final_df['Volume_Dollar'].sum()


# In[ ]:


Final_df['Tweet_vol'].max()


# In[ ]:


Final_df.describe()


# In[ ]:


cor = Final_df.corr()
cor


# In[ ]:


Top_Vol =Final_df['Volume_BTC'].nlargest(10)
Top_Vol


# In[ ]:


Top_Sen =Final_df['Sensitivity'].nlargest(10)
Top_Sen


# In[ ]:


Top_Pol =Final_df['Polarity'].nlargest(10)
Top_Pol


# In[ ]:


Top_Tweet =Final_df['Tweet_vol'].nlargest(10)
Top_Tweet


# In[ ]:


import matplotlib.pyplot as plt
sns.set(style="white")
f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax =sns.heatmap(cor, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .7})
plt.show()


# In[ ]:


plt.plot(Final_df.index, Final_df['Close_Price'], 'black')
plt.plot(Final_df.index, Final_df['Open'], 'yellow')
plt.plot(Final_df.index, Final_df['Low'], 'red')
plt.plot(Final_df.index, Final_df['High'], 'green')
plt.title('BTC Close Price(hr)')
plt.xticks(rotation='vertical')
plt.ylabel('Price ($)');
plt.show();

plt.plot(Final_df.index, Final_df['Volume_BTC'], 'g')
plt.title('Trading Vol BTC(hr)')
plt.xticks(rotation='vertical')
plt.ylabel('Vol BTC');
plt.show();

plt.plot(Final_df.index, Final_df['Polarity'], 'b')
plt.xticks(rotation='vertical')
plt.title('Twitter Sentiment(hr)')
plt.ylabel('Pol (0-1)');
plt.show();
plt.legend()

plt.plot(Final_df.index, Final_df['Tweet_vol'], 'b')
plt.xticks(rotation='vertical')
plt.title('Tweet Vol(hr)')
plt.ylabel('No. of Tweets');
plt.show();
plt.legend()


# In[ ]:


#sns Heatmap for Hour x volume 
#Final_df['time']=Final_df.index.time()
Final_df['time']=Final_df.index.to_series().apply(lambda x: x.strftime("%X"))


# In[ ]:


Final_df.head()


# In[ ]:


hour_df=Final_df


# In[ ]:


hour_df=hour_df.groupby('time').agg(lambda x: x.mean())


# In[ ]:


hour_df


# In[ ]:


hour_df.head()


# In[ ]:


#sns Hourly Heatmap
hour_df['hour'] = hour_df.index
result = hour_df.pivot(index='hour', columns='Polarity', values='Volume_BTC')
sns.heatmap(result, annot=True, fmt="g", cmap='viridis')
plt.title('Polarity x BTC Volume avg(Hr)')
plt.show()

#sns daily heatmap?


# In[ ]:


hour_df['hour'] = hour_df.index
result = hour_df.pivot(index='Volume_BTC', columns='hour', values='Tweet_vol')
sns.heatmap(result, annot=True, fmt="g", cmap='viridis')
plt.title('BTC Vol x Tweet Vol avg(Hr)')
plt.show()


# In[ ]:


cor = Final_df.corr()
cor


# In[ ]:


#----------------End Analysis------------------------#


# In[ ]:


#---------------- LSTM Prep ------------------------#


# In[ ]:


df = Final_df


# In[ ]:


df.info()


# In[ ]:


df = df.drop(['Open','High', 'Low', 'Volume_Dollar'], axis=1)
df.head()


# In[ ]:


df = df[['Close_Price', 'Polarity', 'Sensitivity','Tweet_vol','Volume_BTC']]
df.head()


# In[ ]:


cor = df.corr()
import matplotlib.pyplot as plt
sns.set(style="white")
f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax =sns.heatmap(cor, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .7})
plt.show()


# # LSTM Model

# In[ ]:


from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
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
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# In[ ]:


values = df.values
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df = df[['Close_Price', 'Polarity', 'Sensitivity','Tweet_vol','Volume_BTC']]
df.head()


# In[ ]:


scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df.values)


# In[ ]:


n_hours = 3 #adding 3 hours lags creating number of observations 
n_features = 5 #Features in the dataset.
n_obs = n_hours*n_features


# In[ ]:


reframed = series_to_supervised(scaled, n_hours, 1)
reframed.head()


# In[ ]:


reframed.drop(reframed.columns[-4], axis=1)
reframed.head()


# In[ ]:


print(reframed.head())


# In[ ]:


values = reframed.values
n_train_hours = 200
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
train.shape


# In[ ]:


# split into input and outputs
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]


# In[ ]:


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[ ]:


# design network
model = Sequential()
model.add(LSTM(5, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=6, validation_data=(test_X, test_y), verbose=2, shuffle=False,validation_split=0.2)
# plot history


# In[ ]:


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[ ]:


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours* n_features,))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -4:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -4:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
mse = (mean_squared_error(inv_y, inv_yhat))
print('Test MSE: %.3f' % mse)
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# In[ ]:


plt.plot(inv_y, label='Real')
plt.plot(inv_yhat, label='Predicted')


# In[ ]:


plt.title('Real v Predicted Close_Price')
plt.ylabel('Price ($)')
plt.xlabel('epochs (Hr)')
plt.show()


# In[ ]:





# In[ ]:




