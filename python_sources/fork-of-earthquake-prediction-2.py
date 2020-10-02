#!/usr/bin/env python
# coding: utf-8

# Los Alamos National Laboratory - Earthquake analysis
# ------------------------------------------------------------------------------

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sb
import os
import matplotlib.pyplot as plt


# In[ ]:


def extract_features(df):
    # container
    data = []
    
    # features from the acoustic data set only, since the segments contain nothing else
    max_acoustic = df['acoustic_data'].max()
    mean_acoustic = df['acoustic_data'].mean()
    
    data.append([mean_acoustic])
    data.append([df['acoustic_data'].std()])
    data.append([max_acoustic])
    data.append([df['acoustic_data'].min()])
    
    #number of `peaks` -> above mean + (max-mean)/2 -> any value above (max + mean)/2
    signal_values = df['acoustic_data'].loc[df['acoustic_data'] > (max_acoustic + mean_acoustic) / 2.]
    
    signal_values = np.array(signal_values)
    
    data.append([signal_values.shape[0]]) # number of peaks
    
    data.append(np.correlate(df['acoustic_data'].values[::1000],
                       df['acoustic_data'].values[::1000], mode='same')) # auto-correlate 0.01 % of the data
                                                                         # to see how self-similair it is
                                             
    acoustic_histo = np.histogram(df['acoustic_data'], bins=75)
    data.append(acoustic_histo[0]) # bins
    data.append(acoustic_histo[0]) # values
    
    data.append(np.abs(np.fft.fft(df['acoustic_data'].values[::1000], n=100)))
    
    # we must flatten out the features
    return [item for sublist in data for item in sublist]


# In[ ]:


TextFileReader = pd.read_csv('../input/train.csv', chunksize=150000) # the segment files contain 150000 lines each!

reduced_data = dict()
counter = 0

for df in TextFileReader:
    reduced_data[counter] = dict()
    last_time_to_failure = df['time_to_failure'].values[::-1][0]
    reduced_data[counter][last_time_to_failure] = extract_features(df)
    counter += 1
    if counter % 250 == 0: print('%d segments - done.' % counter)


# In[ ]:


TextFileReader = pd.read_csv('../input/train.csv', chunksize=150000, skiprows=25000)

for df in TextFileReader:
    df.columns = ['acoustic_data', 'time_to_failure']
    reduced_data[counter] = dict()
    last_time_to_failure = df['time_to_failure'].values[::-1][0]
    reduced_data[counter][last_time_to_failure] = extract_features(df)
    counter += 1
    if counter % 250 == 0: print('%d segments - done.' % counter)


# In[ ]:


TextFileReader = pd.read_csv('../input/train.csv', chunksize=150000, skiprows=75000)

for df in TextFileReader:
    df.columns = ['acoustic_data', 'time_to_failure']
    reduced_data[counter] = dict()
    last_time_to_failure = df['time_to_failure'].values[::-1][0]
    reduced_data[counter][last_time_to_failure] = extract_features(df)
    counter += 1
    if counter % 250 == 0: print('%d segments - done.' % counter)


# In[ ]:


len(reduced_data) # number of segments achieved that we could predict on!


# In[ ]:


dataframes = []

for index in range(len(reduced_data)):
    df = pd.DataFrame.from_dict(reduced_data[index], orient='index')
    df['_id'] = index
    df['ttf'] = df.index
    df.set_index('_id', inplace=True)
    dataframes.append(df)
    
del reduced_data


# In[ ]:


for df in dataframes:
    df.to_csv('df_all.csv', mode='a', header=False, index=False)

del dataframes


# In[ ]:


train = pd.read_csv('df_all.csv', header=None)
os.remove('df_all.csv')

train.head()


# In[ ]:


from sklearn.preprocessing import normalize


# In[ ]:


train = train.dropna()
train.shape


# In[ ]:


X = normalize(train.values[:, :405])
y = train.values[:, 405]


# In[ ]:


del train


# In[ ]:


import tensorflow as tf
tf.enable_eager_execution()


# In[ ]:


X_ = np.reshape(X, (X.shape[0], 1, X.shape[1]))

dataset = tf.data.Dataset.from_tensor_slices((X_, y))
sequences = dataset.batch(1, drop_remainder=True)


# In[ ]:


for seq, target in sequences.take(1):
    print(seq.shape, target)


# In[ ]:


BATCH_SIZE = 6

BUFFER_SIZE = 20000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset


# In[ ]:


if tf.test.is_gpu_available():
    rnn = tf.keras.layers.CuDNNGRU
else:
    import functools
    rnn = functools.partial(
        tf.keras.layers.GRU, recurrent_activation='sigmoid')


# In[ ]:


def build_model(rnn_units, batch_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1, 405), batch_size=batch_size),
        tf.keras.layers.Dense(1024, activation='relu'),
        rnn(rnn_units),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    return model


# In[ ]:


model = build_model(1024, BATCH_SIZE)


# In[ ]:


model.summary()


# In[ ]:


def loss(labels, logits):
    return tf.keras.losses.MSE(labels, logits)


# In[ ]:


model.compile(
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001),
    loss = loss)


# In[ ]:


EPOCHS = 35

samples_per_epoch = X.shape[0]
steps_per_epoch = samples_per_epoch // BATCH_SIZE

history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch)


# In[ ]:


model_ = build_model(rnn_units=1024, batch_size=1)

weights = model.get_weights()

model_.set_weights(weights)


# In[ ]:


test_files = os.listdir('../input/test/')


# In[ ]:


result = dict()
for file in test_files:
    
    df = pd.read_csv('../input/test/' + file)
    
    data = np.array(extract_features(df))

    X_test = normalize(data.reshape(1, -1))
    X_test = X_test.reshape(1, 1, 405)
    prediction = model_.predict(X_test)[0]
    result[file[::-1][4:][::-1]] = prediction


# In[ ]:


result_df = pd.DataFrame.from_dict(result, orient='index', columns=['time_to_failure'])
result_df.head(n=2)


# In[ ]:


result_df.to_csv('./submission.csv', columns=['time_to_failure'], index_label='seg_id')

