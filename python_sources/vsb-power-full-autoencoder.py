#!/usr/bin/env python
# coding: utf-8

# let's build an [autoencoder](http://https://blog.keras.io/building-autoencoders-in-keras.html) on test data, shall we?

# In[ ]:


import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os

N_MEASUREMENTS = 800_000
N_FISR_SIGNAL_INDEX = 8712
N_LAST_SIGNAL_INDEX = 29048
N_TOTAL_SIGNALS = N_LAST_SIGNAL_INDEX - N_FISR_SIGNAL_INDEX


# In[ ]:


idx = N_FISR_SIGNAL_INDEX + np.arange(N_TOTAL_SIGNALS)
idx


# In[ ]:


np.random.shuffle(idx) # more fun on each run!


# In[ ]:


# the whole set is just too big to be read at once, will use data generator instead
def generator(indexes, batch_size):
    # From the docs: The generator is expected to loop over its data indefinitely.
    # An epoch finishes when steps_per_epoch (see fit_generator()) batches have been seen by the model.
    while True:
        # 3 phases per signal
        for start_col in range(0, N_TOTAL_SIGNALS, batch_size * 3):
            # uncomment to see what signals are being loaded right now
            # print(f'\nloading signals from {start_col} to {start_col + batch_size * 3}')
            
            cols = [str(c) for c in indexes[start_col:start_col + batch_size * 3]]
            n_signals = len(cols) // 3   # could be less than batch_size!
            signals = pq.read_pandas('../input/test.parquet', columns=cols).to_pandas().values
            # transform data from column-wise to row-wise
            signals = np.vstack(np.split(signals, n_signals, axis=1))
            signals = signals.reshape(-1, N_MEASUREMENTS, 3).astype(np.float16) / 130. 
            
            # X == y, cool
            yield signals, signals # shape=(?, 800_000, 3)


# In[ ]:


from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv1D, ELU, BatchNormalization, GRU, Conv2DTranspose, Reshape, Activation


# In[ ]:


# our toy model
def make_model(ts=N_MEASUREMENTS,):
    #
    encoder = Sequential(
        [
            # 400 samples per filter, @ 4MHz sampling rate that should detect 10KHz and below
            Conv1D(8, 400, strides=80, padding='same', input_shape=(ts, 3)), # (10_000, 8)
            ELU(),
            Conv1D(16, 20, strides=2, padding='same'), # (5_000, 16)
            ELU(),
            Conv1D(32, 4, strides=2, padding='same'), # (2_500, 32)
            BatchNormalization(),
            ELU(),

            # LSTM will hopefully detect some periods
            GRU(128,  return_sequences=False, recurrent_dropout=0.3),
            Dense(128),
            BatchNormalization(),
            ELU(),
            
            # finally, compress our signal to 9 dimentions (3 per phase? who knows..)
            Dense(9, activation='tanh'),
        ]
    )
    encoder.summary()

    decoder = Sequential(
        [
            Dense(128, input_shape=(9,)),
            ELU(),
            Dense(ts // 2000),  # [?, 400]
            BatchNormalization(),
            ELU(),
            Reshape((ts // 2000, 1, 1)),           # [?, rows, cols, ch]
            Conv2DTranspose(16, kernel_size= 4, strides=(2,  5), padding='same'),
            # [?,  800,   5, 16]
            Conv2DTranspose( 8, kernel_size=16, strides=(2,  2), padding='same'),    
            # [?, 1600,  10,  8]
            Conv2DTranspose( 3, kernel_size=16, strides=(5, 10), padding='same'),    
            # [?, 8000, 100,  3]
            Reshape((ts, 3)),  # [?, ts, 3]
            Activation('tanh')
        ]
    )
    decoder.summary()

    inp = Input(shape=(ts, 3))
    output = decoder(encoder(inp))
    ae = Model(inputs=inp, outputs=output)
    ae.summary()

    return encoder, decoder, ae


# In[ ]:


encoder, decoder, ae = make_model()

metric = 'mae'
loss = 'mse'
optimizer = 'adam' # RMSprop()
ae.compile(optimizer=optimizer, loss=loss, metrics=[metric])


# In[ ]:


# keep this low, as each signal is 800K * 3 samples ( or batch_size * 800_000 * 3 * 2 in bytes )
batch_size = 12

# keep this low to get nice history graph
steps_per_epoch = 10

# may wary, depending on how logn you are willing to train
n_epochs = 10
print(f'will train on {batch_size * steps_per_epoch * n_epochs} signals')


# In[ ]:


history = ae.fit_generator(
            generator(idx, batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=n_epochs,
            validation_data=None, # need another generator for this
            shuffle=False,        # already shuffled
        )


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


def plot_history(history):
   loss = history.history['loss']
   mae = history.history['mean_absolute_error']
   #
   epochs = range(len(loss))
   #
   plt.plot(epochs, loss, 'bo', label='loss')
   plt.plot(epochs, mae, 'b', label='mae')
   plt.legend()


# In[ ]:


# training is hard
plot_history(history)


# the fun part, let's plot our predicted signals!

# In[ ]:


def plot_signals(x, y_pred):
    n_samples = N_MEASUREMENTS
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(40, 20), dpi=80, facecolor='w', edgecolor='k', sharey=False)
    for ph in range(3):
        ax[0][0].plot(x[:, ph])
        ax[0][1].plot(x[:, ph][n_samples//2:n_samples//2 + n_samples//16])
        ax[0][2].plot(x[:, ph][n_samples//2:n_samples//2 + n_samples//64])
    for ph in range(3):
        ax[1][0].plot(y_pred[0, :, ph])
        ax[1][1].plot(y_pred[0, :, ph][n_samples//2:n_samples//2 + n_samples//16])
        ax[1][2].plot(y_pred[0, :, ph][n_samples//2:n_samples//2 + n_samples//64])        


# In[ ]:


signals, _ = next(generator(idx, 2))
y_pred = ae.predict(signals[0].reshape(1, -1, 3))
plot_signals(signals[0], y_pred)


# nice, our model was able to encode the source signal into 9-dimentional vector (or just 18 bytes!), and train the decoder to restore a pretty complex shape from it
# 
# how to use this vector for something usefull is another kernel..

# In[ ]:




