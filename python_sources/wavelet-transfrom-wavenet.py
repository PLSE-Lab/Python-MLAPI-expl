#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow-addons')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import random
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.utils.fixes import loguniform
from sklearn import model_selection, metrics
from scipy.stats import uniform
from tqdm.notebook import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, activations, layers, Model, Input, utils, callbacks
sns.set()
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('/kaggle/input/data-without-drift/train_clean.csv')
test = pd.read_csv('/kaggle/input/data-without-drift/test_clean.csv')
NUM_CLASSES = train.open_channels.nunique()
SEQ_LENGTH = 1000
WIDTHS = np.power(2, np.arange(-4, 9), dtype=np.float32)
train.shape, test.shape


# In[ ]:


def wavelet_transform(sig, wd):
    widths = wd
    wt = signal.cwt(sig.values, signal.ricker, widths)
    wt = wt.T
    
    eps = np.max(wt) * 1e-2
    s1 = np.log(np.abs(wt) + eps) - np.log(eps)
    wt = s1 / np.max(s1)
    return wt


# Preprocessing function inspired from https://www.kaggle.com/jazivxt/physically-possible

# In[ ]:


def features(df):
    df = df.sort_values(by=['time']).reset_index(drop=True)
    df['batch'] = np.floor(df.index / 50_000).astype(int)
    df['batch_index'] = df.index  - (df.batch * 50_000)
    df['batch_slices'] = df['batch_index']  // 5_000
    df['batch_slices2'] = df.batch.astype(str).str.zfill(3).astype(str).str.cat(df.batch_slices.astype(str).str.zfill(3).astype(str), sep='_')
    df.reset_index(drop=True, inplace=True)
    
    transforms = df.groupby('batch').signal.apply(wavelet_transform, wd=WIDTHS)
    transforms = np.concatenate(transforms)
    assert transforms.shape[0] == df.shape[0]
    wt_dict = {}
    for i in range(len(WIDTHS)):
        wt_dict['wt_'+str(i)] = transforms[:, i]
    del transforms
    wt_dict = pd.DataFrame(wt_dict)
    df = pd.concat([df, wt_dict], axis=1, sort=False)
    
    for c in ['batch_slices2']:
        d = {}
        d['mean_'+c] = df.groupby([c])['signal'].mean()
        d['median_'+c] = df.groupby([c])['signal'].median()
        d['max_'+c] = df.groupby([c])['signal'].max()
        d['min_'+c] = df.groupby([c])['signal'].min()
        d['std_'+c] = df.groupby([c])['signal'].std()
        d['mean_abs_chg_'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))
        d['abs_max_'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))
        d['abs_min_'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))
        for v in d:
            df[v] = df[c].map(d[v].to_dict())        
        df['range_'+c] = df['max_'+c] - df['min_'+c]
        df['maxtomin_'+c] = df['max_'+c] / df['min_'+c]
        df['abs_avg_'+c] = (df['abs_min_'+c] + df['abs_max_'+c]) / 2

    df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])
    df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]
    df['signal_shift_+2'] = [0, 0,] + list(df['signal'].values[:-2])
    df['signal_shift_-2'] = list(df['signal'].values[2:]) + [0, 0]
    
    for i in df[df['batch_index']==0].index:
        df.loc[i, 'signal_shift_+1'] = -99
    for i in df[df['batch_index']==49999].index:
        df.loc[i, 'signal_shift_-1'] = -99
    for i in df[(df['batch_index']==0) | (df['batch_index']==1)].index:
        df.loc[i, 'signal_shift_+2'] = -99
    for i in df[(df['batch_index']==49999) | (df['batch_index']==49998)].index:
        df.loc[i, 'signal_shift_-2'] = -99
    return df

train = features(train)
test = features(test)


# In[ ]:


cols = [
    'signal',
    'mean_batch_slices2',
    'median_batch_slices2',
    'max_batch_slices2',
    'min_batch_slices2',
    'std_batch_slices2',
    'mean_abs_chg_batch_slices2',
    'abs_max_batch_slices2',
    'abs_min_batch_slices2',
    'range_batch_slices2',
    'maxtomin_batch_slices2',
    'abs_avg_batch_slices2',
    'signal_shift_+1',
    'signal_shift_-1',
    'signal_shift_+2',
    'signal_shift_-2',
]

cols.extend(['wt_'+str(i) for i in range(len(WIDTHS))])
print (len(cols))


# In[ ]:


def macro_f1(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES])
    
    threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
    y_pred = tf.logical_and(y_pred >= threshold,
                                    tf.abs(y_pred) > 1e-12)
    
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)
    
    tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
    tp = tf.cast(tp, tf.float32)
    tp = tf.reduce_sum(tp, axis=0)
    
    fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
    fp = tf.cast(fp, tf.float32)
    fp = tf.reduce_sum(fp, axis=0)
    
    fn = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
    fn = tf.cast(fn, tf.float32)
    fn = tf.reduce_sum(fn, axis=0)
    
    precision = tf.math.divide_no_nan(tp, tf.math.add(tp, fp))
    recall = tf.math.divide_no_nan(tp, tf.math.add(tp, fn))
    f1 = tf.math.divide_no_nan(tf.math.multiply_no_nan(2.0, tf.math.multiply_no_nan(precision, recall)), tf.math.add(precision, recall))
    f1 = tf.reduce_mean(f1)
    
    return f1


# In[ ]:


def gated_residual_block(nb_filters, use_bias, i, x):
    res_x = layers.Conv1D(nb_filters, 1, strides=1, padding='same', use_bias=use_bias)(x)
    tanh_out = layers.Conv1D(nb_filters, 2, dilation_rate=2**i, padding='same',
                             use_bias=use_bias, activation='tanh')(x)
    
    sigm_out = layers.Conv1D(nb_filters, 2, dilation_rate=2**i, padding='same',
                             use_bias=use_bias, activation='sigmoid')(x)
    
    gated_out = layers.Multiply()([tanh_out, sigm_out])
    gated_out = layers.Conv1D(nb_filters, 1, strides=1, padding='same', use_bias=use_bias)(gated_out)
    res_x = layers.Add()([res_x, gated_out])
    return res_x


# In[ ]:


def wavenet(NB_STACKS, DEPTH, NB_FILTERS, DROP_RATE):
    
    def scheduler(epoch):
        if epoch < 10:
            return 0.001
        elif epoch < 20:
            return 0.0008
        elif epoch < 40:
            return 0.0001
        elif epoch < 60:
            return 0.00008
        elif epoch < 80:
            return 0.00006
        elif epoch < 120:
            return 0.00004
        else:
            return 0.00002

    ep = np.arange(0, 100)
    plt.plot(ep, [scheduler(i) for i in ep])
    plt.autoscale()
    plt.title('Learning rate Scheduler')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rate')
    plt.show()
    
    class_names = np.arange(0, 11)
    class_weights = np.arange(1, 12) * 1e-1 + 0.5
    class_weights = dict(zip(class_names, class_weights))

    w = (train.shape[0] / (len(np.bincount(train.open_channels.values))*np.bincount(train.open_channels.values)))
    weights = dict(zip(np.arange(11), w))
    
    scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        
    early_stop = callbacks.EarlyStopping(monitor='val_macro_f1', patience=50, mode='max', restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_macro_f1', factor=.8, mode='max', patience=10, min_lr=1e-5, verbose=1)
    
    train_weights = train.open_channels.map(weights).values
    train_input = train[cols].values.reshape(-1,SEQ_LENGTH,len(cols))
    train_target = pd.get_dummies(train["open_channels"]).values.reshape(-1,SEQ_LENGTH,NUM_CLASSES)
    idx = np.arange(train_input.shape[0])
    train_idx, val_idx = model_selection.train_test_split(idx, random_state = 0, test_size = 0.2)
    
    tf.keras.backend.clear_session()
    seq_length = SEQ_LENGTH
    dilation_depth = DEPTH
    nb_filters = NB_FILTERS
    use_bias=False
    batch_size=40
    nb_stacks = NB_STACKS
    
    model_in = Input(shape=(seq_length, len(cols)))
    out = layers.Conv1D(64, 7, strides=1, padding='same', dilation_rate=1,)(model_in)
    for stack in range(nb_stacks):
        for depth in range(dilation_depth):
            out = gated_residual_block(nb_filters, use_bias, depth, out)
    out = layers.Conv1D(nb_filters, 2, strides=1, padding='same', activation='relu')(out)
    out = layers.Dropout(.2)(out)
    model_out = layers.TimeDistributed(layers.Dense(NUM_CLASSES, activation='softmax'))(out)
    
    model = Model(model_in, model_out)
    opt = optimizers.Adam(learning_rate=1e-3)
    opt = tfa.optimizers.SWA(opt)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=[macro_f1],
                  sample_weight_mode=None,)
    results = model.fit(train_input[train_idx,:,:], train_target[train_idx,:, :],
                        sample_weight=train_weights[train_idx],
                        epochs=1000, verbose=0, callbacks=[early_stop, scheduler],
                        validation_data=(train_input[val_idx,:,:], train_target[val_idx,:,:]),
                        batch_size=batch_size,
                        shuffle=True)
    
    preds = model.predict(train_input[val_idx,:,:])
    preds = preds.reshape(-1, NUM_CLASSES)
    preds = np.argmax(preds, axis=1)
    f1_score = metrics.f1_score(np.argmax(train_target[val_idx].reshape(-1,NUM_CLASSES), axis=1), preds, average='macro')
    return model, results, f1_score


# In[ ]:


def compute_receptive_field(dilation_depth, nb_stacks):
    receptive_field = nb_stacks * (2**dilation_depth * 2) - (nb_stacks - 1)
    return receptive_field

compute_receptive_field(9, 1)


# In[ ]:


best_model, best_r, best_score = wavenet(9, 1, 128, 0)
print(f'Macro F1: {best_score}')


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(16,6))
ax[0].plot(best_r.history['loss'])
ax[0].plot(best_r.history['val_loss'])
ax[0].set_title('Model loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_ylim([0.02, 0.125])
ax[0].legend(['Train', 'Test'], loc='upper left')

ax[1].plot(best_r.history['macro_f1'])
ax[1].plot(best_r.history['val_macro_f1'])
ax[1].set_title('Model Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].set_ylim([0.9, 0.95])
ax[1].legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:


preds = best_model.predict(test[cols].values.reshape(-1,SEQ_LENGTH,len(cols)))
preds = preds.reshape(-1, 11)
preds = np.argmax(preds, axis=1)
test['open_channels'] = preds
test[['time', 'open_channels']].to_csv('submission.csv', index=False, float_format='%.4f')


# In[ ]:




