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


import tensorflow as tf

tf.set_random_seed(42)
np.random.seed(42)

print('GPU', tf.test.is_gpu_available())


# In[ ]:


class DataTrainTestSplit(object):
    """ Warning: The validation set should be statistically significant.
    
    There are 16 failures in the dataset.
    """
    def __init__(self, n_blocks=16, split=0.2):
        self.n_blocks = n_blocks
        perm = np.random.permutation(n_blocks)
        test_samples = int(np.floor(n_blocks * split))
        self.train_slice = perm[:-test_samples]
        self.test_slice = perm[-test_samples:]
    def train(self):
        return self.train_slice
    def test(self):
        return self.test_slice


# In[ ]:


import linear_signal_py as linear_signal

class SignalFeatures(linear_signal.SignalFeatureGenerator):
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


kFolds = [ DataTrainTestSplit(n_blocks=256, split=0.20) for _ in range(5) ]


# In[ ]:


SEGMENT_SIZE = 150_000
STRIDES = 15_000

def get_generators(spliter):
    ds_train = linear_signal.LinearDatasetAccessor(df, spliter.n_blocks, spliter.train())
    ds_eval = linear_signal.LinearDatasetAccessor(df, spliter.n_blocks, spliter.test())

    gen_train = linear_signal.LinearSignalGenerator(
        ds_train, SEGMENT_SIZE, SignalFeatures(), strides=STRIDES, batch_size=64)
    gen_eval = linear_signal.LinearSignalGenerator(
        ds_eval, SEGMENT_SIZE, SignalFeatures(), strides=25_000)

    return gen_train, gen_eval


# In[ ]:


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K

def make_layer(inp, index, filters, kernel_size, strides=1):
    c = Conv1D(filters, kernel_size, strides=strides, name='layer_{0}_conv'.format(index), activation='relu')(inp)
    p = MaxPooling1D(name='layer_{0}_pool'.format(index))(c)
    d = Dropout(0.1)(p)
    return d

class range_initializer(keras.initializers.Initializer):
    def __init__(self, vmax, vmean):
        self.vmax = vmax
        self.vmean = vmean
    def __call__(self, shape, dtype=None, partition_info=None):
        if len(shape) != 2 or shape[1] != 1:
            raise ValueError('Expected shape (N, 1), got ', shape)
        return np.arange(shape[0])[:, np.newaxis] / shape[0] * self.vmax - self.vmean

def make_model():
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

    p = Permute((2, 1))(output)
    s = Dense(8)(p)
    summary = Flatten()(s)

    last = Dense(24, activation='softmax')(summary)
    out = Dense(1, kernel_initializer=range_initializer(16.0, 5.6),
                bias_initializer=keras.initializers.Constant(5.6))(last)
    model = Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0005), loss='mae')
    return model

K.clear_session()
model = make_model()
model.summary()


# In[ ]:


model.save('signal-conv.h5')


# In[ ]:


filepath="signal-conv.{0}.ckpt.hdf5"

cv_history = []

for i, spliter in enumerate(kFolds):
    model = make_model()
    gen_train, gen_val = get_generators(spliter)
    cb_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath.format(i), monitor='val_loss', verbose=True,
        save_best_only=True, mode='min')

    history = model.fit_generator(
        gen_train,
        validation_data=gen_val,
        callbacks=[cb_checkpoint],
        epochs=30)

    cv_history.append(history)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

for hist in cv_history:
    plt.figure()
    plt.plot(hist.history['loss'], label='loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot()
    plt.show()


# In[ ]:


from tqdm import tqdm_notebook

# average the prediction from the multiple folds

submission = pd.read_csv(os.path.join(DATADIR, 'sample_submission.csv'), index_col='seg_id', dtype={'time_to_failure': np.float32})

for fold in range(len(kFolds)):
    model.load_weights('signal-conv.{0}.ckpt.hdf5'.format(fold))
    for seg_id in tqdm_notebook(submission.index):
        seg = pd.read_csv(os.path.join(DATADIR, 'test/' + seg_id + '.csv'))
        X = SignalFeatures().generate(seg, predict=True)
        y = model.predict(X[np.newaxis, :])
        submission.loc[seg_id]['time_to_failure'] += y

submission['time_to_failure'] /= len(kFolds)
submission.to_csv('submission.csv')

