#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/all_currencies.csv', parse_dates=['Date'])


# In[ ]:


df.sample(10)


# In[ ]:


df.head(10)


# In[ ]:


df.dtypes


# In[ ]:


df.index= df['Date']


# In[ ]:


symbol_list = df['Symbol'].value_counts().index.tolist()


# In[ ]:


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10


# In[ ]:


for symbol in symbol_list[2:5]:
    plt.plot(df.loc[df['Symbol'] == symbol,['Close']])


# In[ ]:


plt.plot(df.loc[df['Symbol'] == 'BTC',['Close']])


# In[ ]:


df_BTC = df.loc[df['Symbol'] == 'BTC',['Close']]
df_BTC.shape


# In[ ]:


train = df_BTC[:'2018'].values
valid = df_BTC['2018':].values


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_BTC)


# In[ ]:


x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


# In[ ]:


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))


# In[ ]:


model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=2, batch_size=1, verbose=1)


# In[ ]:


inputs = df_BTC[len(df_BTC) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)


# In[ ]:


X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)


# In[ ]:


X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)


# In[ ]:


rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms


# In[ ]:


train = df_BTC[:'2018']
valid = df_BTC['2018':]
valid['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])

