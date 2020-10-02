#!/usr/bin/env python
# coding: utf-8

# This notebook aims to produce predictions about the number of new cases and the death tolls of coronavirus (per each country) using a sequence to sequence model. The first step is to create the time-series for each country in order to have the input sequences for the model, once we have the sequences I decided to normalize the data with an adaptive normalization layer in order to take in consideration the magnitude of each sequence therefore the predictions will have higher fluctuations if the number of cases is big and viceversa. The model is trained to produce only the next number of the newcases time-series and the deathtoll time-series, when the model is trained it will produce the next value of an input sequence at the time t, with that value is possible to create a new timeseries at the time t+1 just concatenating the new data at the end of the sequence and therefore it's possible to ask to the model for a new prediction and continuing until the desired number of predictions is reached. To be noticed that the model is trained with the sequences of all the countries, and not one model per country. I made this choice becuase I believe the model can detect similitaries among the different time-series and use them to create more accurate results.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense, BatchNormalization, Lambda, Flatten, Reshape
from sklearn.preprocessing import MinMaxScaler
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import Input, Dense, Flatten, Dropout, Lambda,     TimeDistributed, Permute, RepeatVector, LSTM, GRU, Add, Concatenate, Reshape, Multiply, merge, Dot, Activation,     concatenate, dot, Subtract
from keras.initializers import Identity
from keras.activations import sigmoid

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop
# from database import get_datasets
from sklearn.neighbors import KernelDensity
from scipy.stats import ks_2samp, trim_mean, shapiro, normaltest, anderson
from keras.losses import mse, binary_crossentropy, sparse_categorical_crossentropy
from keras import backend as K
import matplotlib.pyplot as plt


# In[ ]:


def read_data():
    data = pd.read_csv("../input/datatrain4/train4.csv")
    data = data.values
    return data


# In[ ]:


data = read_data()


# In[ ]:


def handle_country_text(data):
    stats = list(np.unique(data[:, 2]))
    for idx, d in enumerate(data):
        country = d[2]
        id = stats.index(country)
        d[2] = id

    return stats, data


# In[ ]:


stats, data = handle_country_text(data)


# In[ ]:


def create_sequences(data, stats):
    sequences = []
    to_compute = []
    for idx, s in enumerate(stats):
        seq = data[data[:, 2] == idx]
        if pd.isnull(seq[0, 1]):
            seq = np.delete(seq, [1], 1)
        else:
            to_compute.append(seq)
            stats_p = list(np.unique(seq[:, 1]))
            for idx2, s2 in enumerate(stats_p):
                seqs2 = seq[seq[:, 1] == s2]
                seqs2 = np.delete(seqs2, [0, 1, 3], 1)
                for idx, value in enumerate(reversed(seqs2[:, 1:])):
                    if idx + 1 < len(seqs2):
                        cases = value[0] - seqs2[-(idx + 2), 1]
                        deaths = value[1] - seqs2[-(idx + 2), 2]
                        seqs2[-(idx + 1), 1] = cases
                        seqs2[-(idx + 1), 2] = deaths
                # seqs2[:, 3] = seqs2[:, 3] / 100
                # seqs2[:, 4] = seqs2[:, 4] / 100
                offset = float(idx2) / 10
                seqs2[:, 0] = seqs2[:, 0] + offset
                sequences.append(seqs2)
            continue

        seq = np.delete(seq, [0, 2], 1)

        for idx,value in enumerate(reversed(seq[:,1:])):
            if idx + 1 < len(seq):
                cases = value[0] - seq[-(idx + 2), 1]
                deaths = value[1] - seq[-(idx + 2), 2]
                seq[-(idx + 1), 1] = cases
                seq[-(idx + 1), 2] = deaths
            # seq[:, 3] = seq[:, 3] / 100
            # seq[:, 4] = seq[:, 4] / 100
        sequences.append(seq)

    return np.array(sequences)


# In[ ]:


sequences = create_sequences(data, stats)
sequences = np.array(sequences)
sequences_train = np.delete(sequences, [0], 2)
sequences_train = np.array(sequences_train)


# In[ ]:


def dain(input):

    n_features = 2

    #mean
    mean = Lambda(lambda x: K.mean(input, axis=1))(input)
    adaptive_avg = Dense(n_features,
                         kernel_initializer=Identity(gain=1.0),
                         bias=False)(mean)
    adaptive_avg = Reshape((1, n_features))(adaptive_avg)
    X = Lambda(lambda inputs: inputs[0] - inputs[1])([input, adaptive_avg])

    #std
    std = Lambda(lambda x: K.mean(x**2, axis=1))(X)
    std = Lambda(lambda x: K.sqrt(x+1e-8))(std)
    adaptive_std = Dense(n_features,
                         #kernel_initializer=Identity(gain=1.0),
                         bias=False)(std)
    adaptive_std = Reshape((1, n_features))(adaptive_std)
    # eps = 1e-8
    #adaptive_avg[adaptive_avg <= eps] = 1
    X = Lambda(lambda inputs: inputs[0] / inputs[1])([X, adaptive_std])

    # # #gating
    avg = Lambda(lambda x: K.mean(x, axis=1))(X)
    gate = Dense(n_features,
                 activation="sigmoid",
                 kernel_initializer=Identity(gain=1.0),
                 bias=False)(avg)
    gate = Reshape((1, n_features))(gate)
    X = Lambda(lambda inputs: inputs[0] * inputs[1])([X, gate])

    return X, adaptive_avg, adaptive_std


# In[ ]:


def build_generator(encoder_input_shape, missing_len, verbose=True):
    learning_rate = 0.0002
    optimizer = Adam(lr=learning_rate)
    generator_decoder_type ='seq2seq'

    encoder_inputs = Input(shape=encoder_input_shape)

    hidden, avg, std = dain(encoder_inputs)
    decoder_outputs = []
    # encoder

    encoder = LSTM(128, return_sequences=True, return_state=True)
    lstm_outputs, state_h, state_c = encoder(hidden)
    if generator_decoder_type == 'seq2seq':
        states = [state_h, state_c]
        decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
        decoder_cases = Dense(1, activation='relu')
        decoder_deaths = Dense(1, activation='relu')
        all_outputs_c = []
        all_outputs_d = []
        inputs = lstm_outputs
        for idx in range(missing_len):
            outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)
            inputs = outputs
            outputs = BatchNormalization()(outputs)
            outputs = Flatten()(outputs)
            outputs_cases = decoder_cases(outputs)
            outputs_deaths = decoder_deaths(outputs)

            states = [state_h, state_c]
            std_c = Lambda(lambda inputs: inputs[:, 0, 0])(std)
            avg_c = Lambda(lambda inputs: inputs[:, 0, 0])(avg)

            outputs_cases = Multiply()([outputs_cases, std_c])
            outputs_cases = Add()([outputs_cases, avg_c])

            std_d = Lambda(lambda inputs: inputs[:, 0, 1])(std)
            avg_d = Lambda(lambda inputs: inputs[:, 0, 1])(avg)

            outputs_deaths = Multiply()([outputs_deaths, std_d])
            outputs_deaths = Add()([outputs_deaths, avg_d])
            all_outputs_c.append(outputs_cases)
            all_outputs_d.append(outputs_deaths)

        decoder_outputs_c = Lambda(lambda x: x)(outputs_cases)
        decoder_outputs_d = Lambda(lambda x: x)(outputs_deaths)
   
    model = Model(inputs=encoder_inputs,
                  outputs=[decoder_outputs_c, decoder_outputs_d])
    if verbose:
        print('\nGenerator summary: ')
        print(model.summary())

    model.compile(loss='mean_squared_logarithmic_error', optimizer=optimizer)
    return model


# In[ ]:


given = 80
missing = 1
total_missing = 33

model = build_generator(sequences_train[:, :given, :].shape[1:], missing)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
history = model.fit(x=sequences_train[:, :given, :],
              y=[sequences_train[:, given:, 0], sequences_train[:, given:, 1]],
              epochs=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[es])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('plots/losses.png')
plt.close()


# ## Trained in local
# ![loss](https://i.ibb.co/JHMSZR3/losses.png)
# losses

# In[ ]:


def backtest2(sequences, model, given, missing):
    sequences_test = sequences[:, -given:]

    pred_d = []
    pred_c = []
    for i in range(0, missing):
        predictions = model.predict(sequences_test[:, :, ])
                                       
        predictions[0][predictions[0] < 0] = 0 #* std[0] + m[0]
        predictions[1][predictions[1] < 0] = 0 #* std[1] + m[1]
        predictions[1] = np.around(predictions[1].astype(np.double))
        predictions[0] = np.around(predictions[0].astype(np.double))
        pred = np.concatenate(
            [np.expand_dims(predictions[0], axis=2),
             np.expand_dims(predictions[1], axis=2)],
            axis=2)
        pred_c.append(pred)
        pred_d.append(predictions[1])
        sequences_test = np.concatenate([sequences_test[:, 1:],
                                         pred],
                                        axis=1)
    predictions = np.array(pred_c[0])
    for i in range(1, len(pred_c)):
        predictions = np.concatenate([predictions,
                                     pred_c[i]],
                                     axis=1)


    # #real cases/death denorm
    seq_cases = sequences[:, :, 0] #* std[0] + m[0]
    seq_death = sequences[:, :, 1]# * std[1] + m[1]

    #reverse real variations
    death = np.cumsum(seq_death, axis=1)
    # death = seq_death
    # cases = seq_cases
    cases = np.cumsum(seq_cases, axis=1)

    cases = np.around(cases.astype(np.double))
    cases[cases < 0] = 0
    cases_csv = np.expand_dims(cases[:, -1], axis=1)
    predictions[0] = np.around(predictions[0].astype(np.double))
    cases_csv = np.concatenate((cases_csv, predictions[:, :, 0]), axis=1)

    death = np.around(death.astype(np.double))
    death[death < 0] = 0
    death_csv = np.expand_dims(death[:, -1], axis=1)
    predictions[1] = np.around(predictions[1].astype(np.double))
    death_csv = np.concatenate((death_csv, predictions[:, :, 1]), axis=1)

    #reverse variations predictions
    cases_csv = np.cumsum(cases_csv, axis=1)
    death_csv = np.cumsum(death_csv, axis=1)
    death_csv = death_csv[:, 1:]
    cases_csv = cases_csv[:, 1:]

    #align with testset
    death_csv = np.concatenate((death[:, -11:], death_csv), axis=1)
    cases_csv = np.concatenate((cases[:, -11:], cases_csv), axis=1)

    #flatten and save
    csv = []
    cases_csv = np.reshape(cases_csv[:, 1:], (-1, 1))
    death_csv = np.reshape(death_csv[:, 1:], (-1, 1))

    j = 1
    for idx, (c, d) in enumerate(zip(cases_csv, death_csv)):
        csv.append([j, c, d])
        j += 1
#     df = pd.DataFrame(csv, columns =['ForecastId','ConfirmedCases','Fatalities'])
#     df.ConfirmedCases.astype(np.double)
#     df.Fatalities.astype(np.double)
#     df.ForecastId = df.ForecastId.astype(np.int)
#     df.to_csv("submission.csv", index=False)
    print('done')


# In[ ]:


backtest2(sequences_train, model, given, total_missing)


# ## Trained in local, next cell it's for the submission file creation

# In[ ]:


sub = pd.read_csv('../input/submission2/submission.csv', header=None,dtype=np.float32) 
sub = pd.DataFrame(sub.values, columns =['ForecastId','ConfirmedCases','Fatalities'])
sub.ConfirmedCases.astype(np.double)
sub.Fatalities.astype(np.double)
sub.ForecastId = sub.ForecastId.astype(np.int)
sub.to_csv("submission.csv", index=False)
print('done')

