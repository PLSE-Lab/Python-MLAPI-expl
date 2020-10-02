#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM, Embedding, Bidirectional, Flatten, Activation, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Conv2D, Reshape, MaxPool2D, Concatenate, Dropout, CuDNNLSTM
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import plot_model
from keras import losses
from IPython.display import SVG


# In[5]:


# approach: tackle this as a classification problem 
df_plot = pd.read_csv("../input/3409dji/DJI.csv")
df_plot['Date'] = pd.to_datetime(df_plot['Date'])
df = pd.read_csv("../input/3409dji/DJI.csv", usecols = [4])
# Index(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set(style="darkgrid")
sns.distplot(df_plot['Close'])  
sns.relplot(x = 'Date', y = 'Close', kind = 'line', data = df_plot)
sns.jointplot(x = "Close", y = "Volume", data = df_plot, kind="kde", space=0)


# In[ ]:


# dropping of anomalous values done here 
'''
df.sort_values(by = 'Close', ascending = True).head(10)
to_drop = df.sort_values(by = 'Close', ascending = False)['Open'].head(5).index
df.drop(index = to_drop, inplace = True)
to_drop = df.sort_values(by = 'Close', ascending = True)['Open'].head(5).index
df.drop(index = to_drop, inplace = True)
'''


# In[8]:


# first approach: take diff between past days then predict
df_test = df.iloc[1:].reset_index(drop = True)
df = df[:-1]
df_new = df_test - df


# In[9]:


# scale the min/max to feed into the nn
scaler = MinMaxScaler(feature_range=(0, 1))
df_new['Close'] = scaler.fit_transform(df_new)


# In[10]:


train_size = 0.7
test_size = 0.3
i = int(len(df_new['Close']) * train_size)
df_train = df_new[:i]
df_test = df_new[i:]


# In[11]:


def build_model(input_shape, output_shape = 1, type = 'lstm', activation = 'softmax'):
    if type == 'lstm': 
        inputs = Input(shape=input_shape)
        layer = CuDNNLSTM(64)(inputs)
        # layer = Flatten()(layer)
        layer = Dense(256,name='FC1')(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.2)(layer)
        layer = Dense(output_shape, name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs,outputs=layer)
        return model 
    elif type == 'cnn':
        inputs = Input(shape=input_shape) 
        layer = Conv1D(64, 3, strides=1, padding='same', activation='relu')(inputs)
        # Cuts the size of the output in half, maxing over every 2 inputs
        layer = MaxPooling1D(pool_size=2)(layer)
        # layer = Conv1D(64, 3, strides=1, padding='same', activation='relu')(inputs)
        # layer = MaxPooling1D(pool_size=2)(layer)
        # layer = CuDNNLSTM(64)(layer)
        layer = Dropout(0.2)(layer)
        layer = Flatten()(layer)
        outputs = Dense(output_shape, activation='sigmoid')(layer)
        model = Model(inputs=inputs, outputs=outputs, name='CNN')
        return model 


# In[12]:


# create train/test set
seq = df_train.values.copy()
seq.resize(seq.shape[0], 1)
seq_test = df_test.values.copy()
seq_test.shape


# In[13]:


x = build_model((365, 1), type = 'lstm') 
train_data_gen = TimeseriesGenerator(seq, seq,
                               length=365, sampling_rate=1,stride=1,
                               batch_size=128)
test_data_gen = TimeseriesGenerator(seq_test, seq_test,
                               length=365, sampling_rate=1,stride=1,
                               batch_size=128)
x.compile(optimizer = 'rmsprop', loss = losses.mean_squared_error, metrics = ['mse'])
x.fit_generator(train_data_gen, epochs = 20, shuffle = False)


# In[22]:


plot_model(x, to_file = "model1.png")


# In[14]:


# obtain predicted differences
predicted_values = x.predict_generator(test_data_gen)
predicted_values = scaler.inverse_transform(predicted_values)
plot = df[i+365:] + predicted_values


# In[15]:


# join actual/predicted results and plot 
df_plot = pd.DataFrame(columns = ["actual", "predicted"])
df["actual"] = df[i+365:]
df["predicted"] = plot
plt.plot(df)


# In[ ]:


df_raw = df[['Close']] 
scaler = MinMaxScaler(feature_range=(0, 1))
df_raw['Close'] = scaler.fit_transform(df_raw)


# In[ ]:


seq = np.asarray(df_raw['Close'][:i])
seq = seq.reshape(seq.shape[0], 1)
seq_test = np.asarray(df_raw['Close'][i:])
seq_test = seq_test.reshape(seq_test.shape[0], 1)


# In[ ]:


# second approach: take raw values
x = build_model((365, 1), type = 'lstm') 
train_data_gen = TimeseriesGenerator(seq, seq,
                               length=365, sampling_rate=1,stride=1,
                               batch_size=64)
test_data_gen = TimeseriesGenerator(seq_test, seq_test,
                               length=365, sampling_rate=1,stride=1,
                               batch_size=64)
x.compile(optimizer = 'adam', loss = losses.mean_squared_error, metrics = ['mse'])
x.fit_generator(train_data_gen, epochs = 20, shuffle = False)


# In[ ]:


plot_model(x, to_file = "model2.png")


# In[ ]:


to_plot = pd.DataFrame(columns = ["predicted", "actual"])
predicted_values = x.predict_generator(test_data_gen)
predicted_values = scaler.inverse_transform(predicted_values)
to_plot["predicted"] = np.asarray(predicted_values).ravel()
to_plot["actual"] = scaler.inverse_transform(df_raw[i+365:])
plt.plot(to_plot)


# In[ ]:


# approach utilizing 3 other sets of influential data - google, amazon, apple
# ls = AAPL, AMZN, GOOG in this order now.
df = pd.read_csv("../input/3409dji/DJI.csv", usecols = [4])
ls = []
temp = sorted(os.listdir("../input/3409folder_big3"))
for i in range(len(temp)):
    ls.append(pd.read_csv("../input/3409folder_big3/" + temp[i], usecols = [4]))
    ls[i] = ls[i].rename(columns = {'Close': temp[i][:4]})


# In[ ]:


# join the dataframes together to create 1 big dataframe 
indexed = df.shape[0] - ls[0].shape[0]
df_big = pd.concat([ls[0], ls[1], ls[2], df.iloc[indexed:].reset_index(drop = True)], axis=1, join='inner')


# In[ ]:


scaler_aapl = MinMaxScaler()
scaler_amzn = MinMaxScaler()
scaler_goog = MinMaxScaler()
scaler = MinMaxScaler()


# In[ ]:


df_big['AAPL'] = scaler_aapl.fit_transform(df_big[['AAPL']])
df_big['AMZN'] = scaler_aapl.fit_transform(df_big[['AMZN']])
df_big['GOOG'] = scaler_aapl.fit_transform(df_big[['GOOG']])
df_big['Close'] = scaler.fit_transform(df_big[['Close']])


# In[ ]:


i = int(len(df_big['Close']) * train_size)
df_train = df_big[:i]
df_test = df_big[i:]


# In[ ]:


seq = df_train.values.copy()
seq.resize(seq.shape[0], 4)
seq_y = df_train['Close'].values.copy()
seq_y.resize((seq_y.shape[0], 1))
seq_test = df_test.values.copy()
seq_test.resize(seq_test.shape[0], 4)
seq_test_y = df_test['Close'].values.copy()
seq_test_y.resize((seq_test_y.shape[0], 1))


# In[ ]:


x = build_model((365, 4), type = 'cnn') 
train_data_gen = TimeseriesGenerator(seq, seq_y,
                               length=365, sampling_rate=1,stride=1,
                               batch_size=128)
test_data_gen = TimeseriesGenerator(seq_test, seq_test_y,
                               length=365, sampling_rate=1,stride=1,
                               batch_size=128)
x.compile(optimizer = 'adam', loss = losses.mean_squared_error, metrics = ['mse'])
x.fit_generator(train_data_gen, epochs = 20, shuffle = False)


# In[ ]:


plot_model(x, to_file = "model3.png")


# In[ ]:


to_plot = pd.DataFrame(columns = ["predicted", "actual"])
predicted_values = x.predict_generator(test_data_gen)
predicted_values = scaler.inverse_transform(predicted_values)
to_plot["predicted"] = np.asarray(predicted_values).ravel()
to_plot["actual"] = scaler.inverse_transform(df_big[['Close']][i+365:])
plt.plot(to_plot)

