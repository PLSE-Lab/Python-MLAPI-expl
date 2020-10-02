#!/usr/bin/env python
# coding: utf-8

# # Sequence2Sequence 2019-nCoV 86% Accuracy - Prediction of corona prevalence in the future
# 
# - > You can contact me directly if you are eager to write an immediately published scientific research
# - > If you can't see Visualzations, fork it you will can see it or download it
# 
# Let's not pay much attention to the introductions. The title is very clear. That algorithm may save millions of people. We can apply it to more than one region and more than one domain and add a lot of features to the data. I am fully prepared to work next to any team that has a data provider directly from the World Health Organization. Let's Begin
# 
# ### Sequence2Sequence Model:
# ![sequence2sequence](https://indico.io/wp-content/uploads/2016/04/figure1-1.jpeg)
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy import array

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df['Last Update'] = pd.to_datetime(df['Last Update'])


# In[ ]:


df['Day'] = df['Last Update'].apply(lambda x:x.day)
df['Hour'] = df['Last Update'].apply(lambda x:x.hour)


# In[ ]:


df.head()


# In[ ]:


import matplotlib.pyplot as plt 
plt.figure(figsize=(16,6))
df.groupby('Day').sum()['Confirmed'].plot()


# In[ ]:


Confirmed = df.groupby('Day').sum()['Confirmed']
Deaths = df.groupby('Day').sum()['Deaths']
Recovered = df.groupby('Day').sum()['Recovered']


# In[ ]:


df = pd.DataFrame(data=[Confirmed, Deaths,Recovered])
df = df.T
df


# In[ ]:


from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense


# In[ ]:


def load_data(data, time_step=2, after_day=1, validate_percent=0.67):
    seq_length = time_step + after_day
    result = []
    for index in range(len(data) - seq_length + 1):
        result.append(data[index: index + seq_length])
    result = np.array(result)
    print('total data: ', result.shape)

    train_size = int(len(result) * validate_percent)
    train = result[:train_size, :]
    validate = result[train_size:, :]

    x_train = train[:, :time_step]
    y_train = train[:, time_step:]
    x_validate = validate[:, :time_step]
    y_validate = validate[:, time_step:]
    
     

    return [x_train, y_train, x_validate, y_validate]


# In[ ]:


def base_model(feature_len=3, after_day=3, input_shape=(8, 1)):
    model = Sequential()

    model.add(LSTM(units=100, return_sequences=False, input_shape=input_shape))
    #model.add(LSTM(units=100, return_sequences=False, input_shape=input_shape))

    # one to many
    model.add(RepeatVector(after_day))
    model.add(LSTM(200, return_sequences=True))
    #model.add(LSTM(50, return_sequences=True))

    model.add(TimeDistributed(Dense(units=feature_len, activation='linear')))

    return model

def seq2seq(feature_len=1, after_day=1, input_shape=(8, 1)):
    '''
    Encoder:
    X = Input sequence
    C = LSTM(X); The context vector

    Decoder:
    y(t) = LSTM(s(t-1), y(t-1)); where s is the hidden state of the LSTM(h and c)
    y(0) = LSTM(s0, C); C is the context vector from the encoder.
    '''

    # Encoder
    encoder_inputs = Input(shape=input_shape) # (timesteps, feature)
    encoder = LSTM(units=100, return_state=True,  name='encoder')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    states = [state_h, state_c]

    # Decoder
    reshapor = Reshape((1, 100), name='reshapor')
    decoder = LSTM(units=100, return_sequences=True, return_state=True, name='decoder')

    # Densor
    #tdensor = TimeDistributed(Dense(units=200, activation='linear', name='time_densor'))
    densor_output = Dense(units=feature_len, activation='linear', name='output')

    inputs = reshapor(encoder_outputs)
    #inputs = tdensor(inputs)
    all_outputs = []



    for _ in range(after_day):
        outputs, h, c = decoder(inputs, initial_state=states)

        #inputs = tdensor(outputs)
        inputs = outputs
        states = [state_h, state_c]

        outputs = densor_output(outputs)
        all_outputs.append(outputs)

    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    model = Model(inputs=encoder_inputs, outputs=decoder_outputs)

    return model


# In[ ]:


def normalize_data(data, scaler, feature_len):
    minmaxscaler = scaler.fit(data)
    normalize_data = minmaxscaler.transform(data)
    return normalize_data


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
data = normalize_data(df, scaler,df.shape[1])


# In[ ]:


x_train, y_train, x_validate, y_validate = load_data(data,time_step=3, after_day=4, validate_percent=0.5)


# In[ ]:


print('train data: ', x_train.shape, y_train.shape)
print('validate data: ', x_validate.shape, y_validate.shape)
print('validate data: ', x_validate.shape, y_validate.shape)


# In[ ]:


#x_test = data[:20]
#x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))


# In[ ]:


from keras import backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Activation, TimeDistributed, Dropout, Lambda, RepeatVector, Input, Reshape
from keras.callbacks import ModelCheckpoint


# In[ ]:


# model complie
input_shape = (3, data.shape[1])
model = seq2seq(data.shape[1], 4, input_shape)
model.compile(loss='mse', optimizer='adam',metrics=['acc'])
model.summary()


# In[ ]:


#history = model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_validate, y_validate))
history = model.fit(x_train, y_train, batch_size=3, epochs=50)


# In[ ]:


import math 
print('-' * 100)
train_score = model.evaluate(x=x_train, y=y_train, batch_size=3, verbose=0)
print('Train Score: %.8f MSE (%.8f RMSE ) , %.8f  ACC' % (train_score[0], math.sqrt(train_score[0]),train_score[1]*100)  )


validate_score = model.evaluate(x=x_validate, y=y_validate, batch_size=3, verbose=0)
print('Validation Score: %.8f MSE (%.8f RMSE ) , %.8f  ACC' % (validate_score[0], math.sqrt(validate_score[0]),validate_score[1]*100))


# In[ ]:


train_predict = model.predict(x_train)
validate_predict = model.predict(x_validate)
#test_predict = model.predict(x_test)


# In[ ]:


def inverse_normalize_data(data, scaler):
    for i in range(len(data)):
        data[i] = scaler.inverse_transform(data[i])

    return data


# In[ ]:


train_predict = inverse_normalize_data(train_predict, scaler)
y_train = inverse_normalize_data(y_train, scaler)
validate_predict = inverse_normalize_data(validate_predict, scaler)
y_validate = inverse_normalize_data(y_validate, scaler)
#test_predict = inverse_normalize_data(test_predict, scaler)


# In[ ]:


#train data:  (3, 3, 3) (3, 4, 3)
#validate data:  (1, 3, 3) (1, 4, 3)
#validate data:  (1, 3, 3) (1, 4, 3)


# In[ ]:


day = ['First Day','Second Day','Third Day','Fourth Day']
#dfx = pd.DataFrame(data=[y_validate[:], validate_predict])
#dfx = dfx.T
#dfx
fig = plt.figure(figsize=(20, 15))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)

ax1.plot(day,y_validate[:,:,0][0],color='red',label='Confirmed Actual')
ax1.plot(day,validate_predict[:,:,0][0],color='maroon',label='Confirmed Prediction')
ax1.title.set_text("Confirmed per/day")
ax1.legend()


ax2.bar(day,y_validate[:,:,1][0],color='red',label='Deaths Actual')
ax2.bar(day,validate_predict[:,:,1][0],color='maroon',label='Deaths Prediction')
ax2.title.set_text("Deaths per/day")
ax2.legend()


ax3.bar(day,y_validate[:,:,2][0],color='red',label='Recoverd Actual')
ax3.bar(day,validate_predict[:,:,2][0],color='maroon',label='Recoverd Prediction')
ax3.title.set_text("Recoverd per/day")
ax3.legend()

plt.show()


# In[ ]:


x_train, y_train, x_validate, y_validate = load_data(data,time_step=3, after_day=4, validate_percent=0.)


# In[ ]:


train_predict = inverse_normalize_data(train_predict, scaler)
y_train = inverse_normalize_data(y_train, scaler)
validate_predict = inverse_normalize_data(validate_predict, scaler)
y_validate = inverse_normalize_data(y_validate, scaler)
#test_predict = inverse_normalize_data(test_predict, scaler)


# In[ ]:


x_test = data[7:]
x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))


# In[ ]:


x_test.shape


# In[ ]:


next_predict = model.predict(x_test)
next_predict_res = inverse_normalize_data(next_predict, scaler)
next_predict_res


# In[ ]:


next4_0 = np.pad(next_predict_res[:,:,0][0], [(4, 0)], mode='constant')
next4_0[:4]=y_validate[:,:,0][0]

next4_1 = np.pad(next_predict_res[:,:,1][0], [(4, 0)], mode='constant')
next4_1[:4]=y_validate[:,:,1][0]

next4_2 = np.pad(next_predict_res[:,:,2][0], [(4, 0)], mode='constant')
next4_2[:4]=y_validate[:,:,2][0]



BACK4_0 = np.pad(y_validate[:,:,0][0], [(0, 4)], mode='constant')
BACK4_0[4:]=np.NAN

BACK4_1 = np.pad(y_validate[:,:,1][0], [(0, 4)], mode='constant')
BACK4_1[4:]=np.NAN

BACK4_2 = np.pad(y_validate[:,:,2][0], [(0, 4)], mode='constant')
BACK4_2[4:]=np.NAN





# In[ ]:


day = ['28-1-2020','29-1-2020','30-1-2020','31-1-2020','1-2-2020','2-2-2020','3-2-2020','4-2-2020']
#dfx = pd.DataFrame(data=[y_validate[:], validate_predict])
#dfx = dfx.T
#dfx
fig = plt.figure(figsize=(35, 20))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)                                            

ax1.plot(day,next4_0,color='maroon',ls='--',clip_on=True,label='Predicted Confirmed')
ax1.plot(BACK4_0,color='red',clip_on=True,label='Actual Confirmed')
#ax1.plot(day,np.pad(validate_predict[:,:,0][0], [(0, 4)], mode='constant'),color='blue',clip_on=False)

ax1.title.set_text("Confirmed per/day")
ax1.legend()



#ax2.plot(day,np.pad(validate_predict[:,:,1][0], [(0, 4)], mode='constant'),color='blue')
ax2.bar(day,next4_1,color='maroon',ls='--',clip_on=True,label='Predicted Deaths')
ax2.bar(day,BACK4_1,color='red',clip_on=True,label='Actual Deaths')
ax2.legend()

ax2.title.set_text("Deaths per/day")
ax2.legend()

ax3.bar(day,next4_2,color='maroon',ls='--',clip_on=True,label='Predicted Recoverd')
ax3.bar(day,BACK4_2,color='red',clip_on=True,label='Actual Recoverd')
#ax3.plot(day,np.pad(validate_predict[:,:,2][0], [(0, 4)], mode='constant'),color='blue')

ax3.title.set_text("Recoverd per/day")
ax3.legend()

plt.show()


# In[ ]:



df = pd.DataFrame(data=[day,next4_0, next4_1,next4_2])
df = df.T
df.columns=['Date','Confirmed','Deaths','Recoverd']
df


# In[ ]:


df.Confirmed = df.Confirmed.apply(lambda x : np.round(x,0))
df.Deaths = df.Deaths.apply(lambda x : np.round(x,0))
df.Recoverd = df.Recoverd.apply(lambda x : np.round(x,0))

df


# In[ ]:




