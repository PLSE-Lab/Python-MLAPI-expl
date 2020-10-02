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
DATADIR = "../input/LANL-Earthquake-Prediction"

print(os.listdir(DATADIR))

# Any results you write to the current directory are saved as output.


# In[ ]:


# second cell
df = pd.read_csv(os.path.join(DATADIR, 'train.csv'),
                 dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})


# In[ ]:


class SignalFeatures(object):
    SEQUENCE_LENGHT = 150_000

    S_MEAN = 4
    S_STD = 10

    def __init__(self, normalize=True):
        self.normalize = normalize

    def shape(self):
        return (self.SEQUENCE_LENGHT, 1)

    def generate(self, df: pd.DataFrame, predict=False):
        """ The performance of this function when vectorized is 10x of
        an iterative loop.
        """
        X = df['acoustic_data'].values[:, np.newaxis]
        if self.normalize:
            X = (X - self.S_MEAN) / self.S_STD
        if predict:
            return X
        y = df['time_to_failure'].iloc[df.shape[0] - 1]
        return X, np.array([y])


# In[ ]:


import lanl_generator_py as lanl_generator

def get_generators(df, folder: lanl_generator.FoldGenerator, index: int):
    """ Use 3x the train data with a random offset on the start of the segment.
        This should maintain a similar distribution between train and test datasets.
    """
    train_indices, eval_indices = folder[index]
    gen_train = lanl_generator.SegmentGenerator(
        df, SignalFeatures(), train_indices, rand_offset=50_000)
    gen_eval = lanl_generator.SegmentGenerator(
        df, SignalFeatures(), eval_indices)
    return gen_train, gen_eval


# In[ ]:


import lanl_generator_py as lanl_generator

classes = lanl_generator.get_lanl_classes()
df_segments = lanl_generator.classify_segments(df, classes)
folder = lanl_generator.FoldGenerator(df_segments)


# In[41]:


from tqdm.autonotebook import tqdm

def generate_data(gen):
    X_data = []
    Y_data = []
    for i in tqdm(range(len(gen))):
        x, y = gen[i]
        X_data.append(x)
        Y_data.append(y)

    return np.vstack(X_data), np.concatenate(Y_data)

def prediction_error(model, classes, gen_eval):
    X_eval, Y_eval = generate_data(gen_eval)
    y_pred = model.predict(X_eval)
    serr = (y_pred - Y_eval).reshape(-1)
    
    err = np.abs(serr)
    bins = np.digitize(Y_eval, classes)
    hist = np.bincount(bins.reshape(-1), minlength=classes.size)
    errhist = np.zeros((classes.size))
    for e, b in zip(err, bins):
        errhist[b] += e
    errhist /= (hist + 1.0e-9)

    hdensity = hist / np.sum(hist)
    return serr, errhist, hdensity


# In[46]:


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K

def make_layer(inp, index, filters, kernel_size, strides=1):
    W_REG = 0.02
    c = Conv1D(filters, kernel_size, strides=strides,
               name='layer_{0}_conv'.format(index),
               activation='relu',
               kernel_regularizer=keras.regularizers.l2(W_REG))(inp)
    p = MaxPooling1D(name='layer_{0}_pool'.format(index))(c)
    d = Dropout(0.1)(p)
    return d

class range_initializer(keras.initializers.Initializer):
    def __init__(self, vmax, vmean):
        self.vmax = vmax
        self.vmean = vmean

    def get_config(self):
        return {'vmax': self.vmax, 'vmean': self.vmean}

    def __call__(self, shape, dtype=None, partition_info=None):
        if len(shape) != 2 or shape[1] != 1:
            raise ValueError('Expected shape (N, 1), got ', shape)
        return np.arange(shape[0])[:, np.newaxis] / shape[0] * self.vmax - self.vmean

def make_model():
    W_REG = 0.05
    inp = Input(shape=SignalFeatures().shape())
    
    params = [
        (16, 16, 4),
        (32, 16, 4),
        (48, 16, 4),
        (64, 8, 2),
        (32, 8, 2),
    ]

    layer_in = inp
    for i, param in enumerate(params):
        output = make_layer(layer_in, i, *param)
        layer_in = output

    s = Dense(16,
              kernel_regularizer=keras.regularizers.l2(W_REG))(output)
    summary = Flatten()(s)

    last = Dense(32, name='conv-to-ttf', activation='softmax')(summary)
    out = Dense(1, name='ttf',
                kernel_initializer=range_initializer(16.0, 5.6),
                bias_initializer=keras.initializers.Constant(5.6))(last)
    model = Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mae', metrics=['mae'])
    return model


# In[47]:


K.clear_session()
model = make_model()
model.summary()
model.save('signal-conv.h5')


# In[ ]:


ckpt_filepath="signal-conv.{0}.ckpt.hdf5"

cv_history = []
error_measurements = []

for fold in range(len(folder)):
    model = make_model()
    gen_train, gen_eval = get_generators(df, folder, fold)
    cb_checkpoint = keras.callbacks.ModelCheckpoint(
        ckpt_filepath.format(fold), monitor='val_loss', verbose=False,
        save_best_only=True, mode='min')

    cb_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit_generator(
        gen_train,
        validation_data=gen_eval,
        callbacks=[cb_checkpoint, cb_stop],
        epochs=50)

    cv_history.append(history)

    measurements = prediction_error(model, classes, gen_eval)
    error_measurements.append(measurements)


# In[ ]:


from tqdm.autonotebook import tqdm

# average the prediction from the multiple folds

submission = pd.read_csv(os.path.join(DATADIR, 'sample_submission.csv'), index_col='seg_id', dtype={'time_to_failure': np.float32})

for fold in range(len(folder)):
    model.load_weights('signal-conv.{0}.ckpt.hdf5'.format(fold))
    for seg_id in tqdm(submission.index):
        seg = pd.read_csv(os.path.join(DATADIR, 'test/' + seg_id + '.csv'))
        X = SignalFeatures().generate(seg, predict=True)
        y = model.predict(X[np.newaxis, :])
        submission.loc[seg_id]['time_to_failure'] += y

submission['time_to_failure'] /= len(folder)
submission.to_csv('submission.csv')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

def show_history(hist, fold):
    fig, ax1 = plt.subplots()
    ax1.plot(hist.history['loss'], label='loss')
    ax1.plot(hist.history['val_loss'], c='g', label='val_loss')
    ax1.set_title('Model loss - {0}'.format(fold))
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    if 'mean_absolute_error' in hist.history:
        ax2 = ax1.twinx()
        ax2.plot(hist.history['mean_absolute_error'], c='y', label='mae')
        ax2.plot(hist.history['val_mean_absolute_error'], c='r', label='val_mae')
        ax2.set_ylabel('MAE')
    fig.legend()
    plt.show()

for i, hist in enumerate(cv_history):
    show_history(hist, i)


# In[ ]:


for fold, measurements in enumerate(error_measurements):
    serr, errhist, hdensity = measurements
    print(fold)
    print(' ', '[0, 1.0)', np.mean(errhist[:2]),
          '[1.0, 8.0)', np.mean(errhist[2:16]),
          '[8.0, inf)', np.mean(errhist[16:]))
    print(' ', '[2.0, 4.0)', np.mean(errhist[4:8]))


# In[ ]:


fig, axes = plt.subplots(len(error_measurements), 1, figsize=(8, 16),
                        sharex=True, sharey=True)

def show_mae_distribution(ax1, classes, measurements, fold):
    _, errhist, hdensity = measurements
    ax1.set_title('MAE per ttf class - {0}'.format(fold))
    ax1.plot(classes, errhist, label='mae', c='r')
    ax2 = ax1.twinx()
    ax2.plot(classes, hdensity, label='density', c='b')

for i, measurements in enumerate(error_measurements):
    show_mae_distribution(axes[i], classes, measurements, fold)

fig.legend()
plt.show()

