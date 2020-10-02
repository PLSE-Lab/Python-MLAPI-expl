#!/usr/bin/env python
# coding: utf-8

# ## Acknowledgements
# * https://www.kaggle.com/ragnar123/wavenet-with-1-more-feature
# * https://www.kaggle.com/vbmokin/ion-switching-advanced-fe-lgb-xgb-confmatrix
# * https://www.kaggle.com/nxrprime/more-low-pass-filtering
# * https://www.kaggle.com/ragnar123/wavenet-with-1-more-feature
# * https://www.kaggle.com/cdeotte/one-feature-model-0-930#Test-Data

# ## Commit Summary
# 
# ### Commit 1: Public Score Timed Out
# 
# 
# ### Commit 2: Public Score 0.933
# * EPOCHS = 110
# * NNBATCHSIZE = 16
# * GROUP_BATCH_SIZE = 4000
# * SEED = 123
# * LR = 0.005
# * SPLITS = 5
# 
# ### Commit 3: Public Score 0.940
# * EPOCHS = 110
# * NNBATCHSIZE = 16
# * GROUP_BATCH_SIZE = 4000
# * SEED = 123
# * LR = 0.001
# * SPLITS = 5
# 
# ### Commit 4: Public Score 0.940
# * EPOCHS = 100
# * NNBATCHSIZE = 16
# * GROUP_BATCH_SIZE = 4000
# * SEED = 123
# * LR = 0.0005
# * SPLITS = 4
# 
# 
# ### Commit 5: Public Score 0.940
# * EPOCHS = 100
# * NNBATCHSIZE = 32
# * GROUP_BATCH_SIZE = 4000
# * SEED = 123
# * LR = 0.001
# * SPLITS = 4
# 
# 
# ### Commit 6: Public Score Timed Out on GPU
# * Shifts Added
# * EPOCHS = 110
# * NNBATCHSIZE = 16
# * GROUP_BATCH_SIZE = 4000
# * SEED = 123
# * LR = 0.005
# * SPLITS = 4
# 
# ### Commit 7: Public Score Timed Out on GPU
# * Shifts Added
# * EPOCHS = 110
# * NNBATCHSIZE = 16
# * GROUP_BATCH_SIZE = 4000
# * SEED = 123
# * LR = 0.005
# * SPLITS = 4
# 
# ### Commit 8: Public Score Timed Out on GPU
# * 1 Shift Added
# * EPOCHS = 110
# * NNBATCHSIZE = 16
# * GROUP_BATCH_SIZE = 4000
# * SEED = 123
# * LR = 0.005
# * SPLITS = 4
# 
# ### Commit 9: Public Score Time Out on GPU
# * 1 Shift Added
# * EPOCHS = 110
# * NNBATCHSIZE = 16
# * GROUP_BATCH_SIZE = 4000
# * SEED = 123
# * LR = 0.005
# * SPLITS = 4
# 
# ### Commit 10: Public Score 
# * EPOCHS = 100
# * NNBATCHSIZE = 32
# * GROUP_BATCH_SIZE = 4000
# * SEED = 123
# * LR = 0.001
# * SPLITS = 4

# <a class="anchor" id="0.1"></a>
# ## Table of Contents
# 
# 1. [Libraries, Reading Files](#1)
# 1. [Reviewing Drift](#2)
# 1. [Removing Drifts](#3)
#     -  [Linear](#3.1)
#     -  [Parabolic](#3.2)    
# 1. [Adding Shifts](#4)
# 1. [Model](#5)

# ## 1. Libraries, Reading Files <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


get_ipython().system('pip install tensorflow_addons')
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Input, Dense, Add, Multiply
import pandas as pd
import numpy as np
import random
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses, models, optimizers
import tensorflow_addons as tfa
import gc

import time
import datetime
import seaborn as sns
import math
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy as sp
from scipy.signal import butter,filtfilt,freqz

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/liverpool-ion-switching/train.csv')
test = pd.read_csv('../input/liverpool-ion-switching/test.csv')
ss = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')


# In[ ]:


def make_batches(df, dataset="train"):
    batches = []
    batch_size = [500000, 100000]
    if dataset == "train":
        for idx in range(10):
            batches.append(df[idx * batch_size[0]: (idx + 1) * batch_size[0]])
    else:
        for idx in range(10):
            batches.append(df[idx * batch_size[1]: (idx + 1) * batch_size[1]])
        for idx in range(2):
            base = 10 * batch_size[1]
            batches.append(df[base + idx * batch_size[0]: base + (idx + 1) * batch_size[0]])
    return batches

train_batches = make_batches(train, "train")
test_batches = make_batches(test, "test")


# ## 2. Reviewing Drifts <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


def plot_all(train, test, suffix=""):
    plt.figure(figsize=(25, 5))
    plt.subplot("211")
    plt.title("Train " + suffix)
    plt.ylabel("Signal")
    plt.xticks(np.arange(0, 501, 50))
    for x in train:
        plt.plot(x['time'], x['signal'], linewidth=.1)
    plt.grid()
    plt.subplot("212")
    plt.title("Test " + suffix)
    plt.ylabel("Signal")
    plt.xticks(np.arange(500, 701, 10))
    for x in test:
        plt.plot(x['time'], x['signal'], linewidth=.1)
    plt.grid()

plot_all(train_batches, test_batches, "Original")


# ## 3. Removing Drifts <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# * ## 3.1. Linear <a class="anchor" id="3.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


linear_train_idx = [1]
linear_test_idx = [0, 1, 4, 6, 7, 8]

def linear_drift(x, x0):
    return 0.3 * (x - x0)


def remove_linear_drift(data, dataset="train"):
    if dataset == "train":
        data[1].loc[data[1].index[0:100000], 'signal'] = data[1].signal[0:100000].values - linear_drift(data[1].time[0:100000].values, data[1].time[0:1].values)
    else:
        for idx in linear_test_idx:
            data[idx].loc[data[idx].index[0:100000], 'signal'] = data[idx].signal[0:100000].values - linear_drift(data[idx].time[0:100000].values, data[idx].time[0:1].values)
            
    return data

train_drift_removed = remove_linear_drift(train_batches, "train")
test_drift_removed = remove_linear_drift(test_batches, "test")


# In[ ]:


plot_all(train_drift_removed, test_drift_removed, "- Linear Drift Removed")


# In[ ]:


df_train = train_drift_removed.copy()
df_test = test_drift_removed.copy()


# * ## 3.2. Parabolic <a class="anchor" id="3.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


parabola_train_idx = [6, 7, 8, 9]
parabola_test_idx = [10]

plt.figure(figsize=(30, 4))
for n, idx in enumerate(parabola_train_idx):
    plt.subplot("15" + str(n + 1))
    plt.title("Train " + str(idx))
    plt.ylabel("Signal", fontsize=8)
    plt.plot(df_train[idx]['time'], df_train[idx]['signal'], linewidth=.1)
    plt.grid()
    plt.ylim([np.min(df_train[idx]['signal']), np.min(df_train[idx]['signal']) + 18])
plt.subplot("155")
plt.title("Test 10")
plt.ylabel("Signal", fontsize=8)
plt.ylim([np.min(df_test[10]['signal']), np.min(df_test[10]['signal']) + 18])
plt.plot(df_test[10]['time'], df_test[10]['signal'], linewidth=.1)
plt.grid()


# In[ ]:


def my_sin(x, A, ph, d):
    frequency = 0.01
    omega = 2 * np.pi * frequency
    return A * np.sin(omega * x + ph) + d


def parabolic_drift_fit(data):
    x = data['time']
    y = data['signal']

    frequency = 0.01
    omega = 2 * np.pi * frequency
    M = np.array([[np.sin(omega * t), np.cos(omega * t), 1] for t in x])
    y = np.array(y).reshape(len(y), 1)

    (theta, _, _, _) = np.linalg.lstsq(M, y)
    
    A = np.sqrt(theta[0,0]**2 + theta[1,0]**2)
    ph = math.atan2(theta[1,0], theta[0,0])
    d = theta[2,0]

    popt = [A, ph, d]
    print(popt)
    return popt


parabola_params = []
for idx in parabola_train_idx:
    parabola_params.append(parabolic_drift_fit(df_train[idx]))
parabola_params.append(parabolic_drift_fit(df_test[parabola_test_idx[0]]))    
    
plt.figure(figsize=(30, 4))
for n, idx in enumerate(parabola_train_idx):
    plt.subplot("15" + str(n + 1))
    plt.title("Train " + str(idx))
    plt.ylabel("Signal", fontsize=8)
    plt.plot(df_train[idx]['time'], df_train[idx]['signal'], linewidth=.1)
    plt.plot(df_train[idx]['time'], my_sin(df_train[idx]['time'], *parabola_params[n]), 'y')
    plt.grid()
    plt.ylim([np.min(df_train[idx]['signal']), np.min(df_train[idx]['signal']) + 18])
plt.subplot("155")
plt.title("Test 10")
plt.ylabel("Signal", fontsize=8)
plt.ylim([np.min(df_test[10]['signal']), np.min(df_test[10]['signal']) + 18])
plt.plot(df_test[10]['time'], df_test[10]['signal'], linewidth=.1)
plt.plot(df_test[10]['time'], my_sin(df_test[10]['time'], *parabola_params[-1]), 'y')
plt.grid()


# In[ ]:


def my_sin(x, A, ph, d):
    frequency = 0.01
    omega = 2 * np.pi * frequency
    return A * np.sin(omega * x + ph) + d


def parabolic_drift_fit(data):
    x = data['time']
    y = data['signal']

    frequency = 0.01
    omega = 2 * np.pi * frequency
    M = np.array([[np.sin(omega * t), np.cos(omega * t), 1] for t in x])
    y = np.array(y).reshape(len(y), 1)

    (theta, _, _, _) = np.linalg.lstsq(M, y)
    
    A = np.sqrt(theta[0,0]**2 + theta[1,0]**2)
    ph = math.atan2(theta[1,0], theta[0,0])
    d = theta[2,0]

    popt = [A, ph, d]
    print(popt)
    return popt


parabola_params = []
for idx in parabola_train_idx:
    parabola_params.append(parabolic_drift_fit(df_train[idx]))
parabola_params.append(parabolic_drift_fit(df_test[parabola_test_idx[0]]))    
    
plt.figure(figsize=(30, 4))
for n, idx in enumerate(parabola_train_idx):
    plt.subplot("15" + str(n + 1))
    plt.title("Train " + str(idx))
    plt.ylabel("Signal", fontsize=8)
    plt.plot(df_train[idx]['time'], df_train[idx]['signal'], linewidth=.1)
    plt.plot(df_train[idx]['time'], my_sin(df_train[idx]['time'], *parabola_params[n]), 'y')
    plt.grid()
    plt.ylim([np.min(df_train[idx]['signal']), np.min(df_train[idx]['signal']) + 18])
plt.subplot("155")
plt.title("Test 10")
plt.ylabel("Signal", fontsize=8)
plt.ylim([np.min(df_test[10]['signal']), np.min(df_test[10]['signal']) + 18])
plt.plot(df_test[10]['time'], df_test[10]['signal'], linewidth=.1)
plt.plot(df_test[10]['time'], my_sin(df_test[10]['time'], *parabola_params[-1]), 'y')
plt.grid()


# In[ ]:


parabola_train_idx = [6, 7, 8, 9]
parabola_test_idx = [10]

def parabolic_drift(x, t=0):
    f = 0.01
    omega = 2 * np.pi * f
    return 5 * np.sin(omega * x + t * np.pi)


def remove_parabolic_drift(data, dataset="train"):
    if dataset == "train":
        for idx in parabola_train_idx:
            data[idx].loc[data[idx].index[0:500000], 'signal'] = data[idx].signal[0:500000].values - parabolic_drift(data[idx].time[0:500000].values, (idx % 2))
    else:
        data[10].loc[data[10].index[0:500000], 'signal'] = data[10].signal[0:500000].values - parabolic_drift(data[10].time[0:500000].values)
            
    return data

df_train = remove_parabolic_drift(train_drift_removed, "train")
df_test = remove_parabolic_drift(test_drift_removed, "test")


# In[ ]:


plot_all(df_train, df_test, "- Without Drift")


# In[ ]:


df_train_clean = df_train[0]
df_test_clean = df_test[0]
for df in df_train[1:]:
    df_train_clean = pd.concat([df_train_clean, df], ignore_index=True)
for df in df_test[1:]:
    df_test_clean = pd.concat([df_test_clean, df], ignore_index=True)

df_train_clean.to_csv("train_wo_drift.csv", index=False, float_format="%.4f")
df_test_clean.to_csv("test_wo_drift.csv", index=False, float_format="%.4f")


# ## 4. Adding Shifts <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


def features(df):
    df = df.sort_values(by=['time']).reset_index(drop=True)
    df.index = ((df.time * 10_000) - 1).values
    df['batch'] = df.index // 25_000
    df['batch_index'] = df.index  - (df.batch * 25_000)
    df['batch_slices'] = df['batch_index']  // 2500
    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)
    
    for c in ['batch','batch_slices2']:
        d = {}
        d['mean'+c] = df.groupby([c])['signal'].mean()
        d['median'+c] = df.groupby([c])['signal'].median()
        d['max'+c] = df.groupby([c])['signal'].max()
        d['min'+c] = df.groupby([c])['signal'].min()
        d['std'+c] = df.groupby([c])['signal'].std()
        d['mean_abs_chg'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))
        d['abs_max'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))
        d['abs_min'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))
        d['range'+c] = d['max'+c] - d['min'+c]
        d['maxtomin'+c] = d['max'+c] / d['min'+c]
        d['abs_avg'+c] = (d['abs_min'+c] + d['abs_max'+c]) / 2
        for v in d:
            df[v] = df[c].map(d[v].to_dict())

    
    # add shifts_1
    #df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])
    #df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]
    #for i in df[df['batch_index']==0].index:
     #   df['signal_shift_+1'][i] = np.nan
    #for i in df[df['batch_index']==49999].index:
     #   df['signal_shift_-1'][i] = np.nan
    
 
    df = df.replace([np.inf, -np.inf], np.nan)    
    df.fillna(0, inplace=True)
    gc.collect()
    return df


# In[ ]:


df_train = features(df_train_clean)
df_test = features(df_test_clean)


# In[ ]:


df_train.to_csv("train_wo_drift_with_shifts.csv", index=False, float_format="%.4f")
df_test.to_csv("test_wo_drift_with_shifts.csv", index=False, float_format="%.4f")


# In[ ]:


def read_data():
    train = pd.read_csv('/kaggle/working/train_wo_drift_with_shifts.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv('/kaggle/working/test_wo_drift_with_shifts.csv', dtype={'time': np.float32, 'signal': np.float32})
    sub  = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})
    return train, test, sub


# In[ ]:


read_data()


# ## 5. Model <a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# configurations and main hyperparammeters
EPOCHS = 100
NNBATCHSIZE = 32
GROUP_BATCH_SIZE = 4000
SEED = 123
LR = 0.001
SPLITS = 4


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


# In[ ]:


def batching(df, batch_size):
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df

# normalize the data (standard scaler). We can also try other scalers for a better score!
def normalize(train, test):
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    train['signal'] = (train.signal - train_input_mean) / train_input_sigma
    test['signal'] = (test.signal - train_input_mean) / train_input_sigma
    return train, test

# get lead and lags features
def lag_with_pct_change(df, windows):
    for window in windows:    
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
    return df

# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).
def run_feat_engineering(df, batch_size):
    # create batches
    df = batching(df, batch_size = batch_size)
    # create leads and lags (1, 2, 3 making them 6 features)
    df = lag_with_pct_change(df, [1, 2, 3])
    # create signal ** 2 (this is the new feature)
    df['signal_2'] = df['signal'] ** 2
    return df

# fillna with the mean and select features for training
def feature_selection(train, test):
    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()
        train[feature] = train[feature].fillna(feature_mean)
        test[feature] = test[feature].fillna(feature_mean)
    return train, test, features

# model function (very important, you can try different arquitectures to get a better score. I believe that top public leaderboard is a 1D Conv + RNN style)
def Classifier(shape_):
    
    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = Conv1D(filters = filters,
                   kernel_size = 1,
                   padding = 'same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same', 
                              activation = 'tanh', 
                              dilation_rate = dilation_rate)(x)
            sigm_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same',
                              activation = 'sigmoid', 
                              dilation_rate = dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters = filters,
                       kernel_size = 1,
                       padding = 'same')(x)
            res_x = Add()([res_x, x])
        return res_x
    
    inp = Input(shape = (shape_))
    
    x = wave_block(inp, 32, 3, 12)
    x = wave_block(x, 64, 3, 8)
    x = wave_block(x, 64, 3, 4)
    x = wave_block(x, 128, 3, 1)
    
    out = Dense(11, activation = 'softmax', name = 'out')(x)
    
    model = models.Model(inputs = inp, outputs = out)
    
    opt = Adam(lr = LR)
    opt = tfa.optimizers.SWA(opt)
    model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])
    return model

# function that decrease the learning as epochs increase (i also change this part of the code)
def lr_schedule(epoch):
    if epoch < 30:
        lr = LR
    elif epoch < 40:
        lr = LR / 3
    elif epoch < 50:
        lr = LR / 5
    elif epoch < 60:
        lr = LR / 7
    elif epoch < 70:
        lr = LR / 9
    elif epoch < 80:
        lr = LR / 11
    elif epoch < 90:
        lr = LR / 13
    else:
        lr = LR / 100
    return lr

# class to get macro f1 score. This is not entirely necessary but it's fun to check f1 score of each epoch (be carefull, if you use this function early stopping callback will not work)
class MacroF1(Callback):
    def __init__(self, model, inputs, targets):
        self.model = model
        self.inputs = inputs
        self.targets = np.argmax(targets, axis = 2).reshape(-1)
        
    def on_epoch_end(self, epoch, logs):
        pred = np.argmax(self.model.predict(self.inputs), axis = 2).reshape(-1)
        score = f1_score(self.targets, pred, average = 'macro')
        print(f'F1 Macro Score: {score:.5f}')

# main function to perfrom groupkfold cross validation (we have 1000 vectores of 4000 rows and 8 features (columns)). Going to make 5 groups with this subgroups.
def run_cv_model_by_batch(train, test, splits, batch_col, feats, sample_submission, nn_epochs, nn_batch_size):
    
    seed_everything(SEED)
    K.clear_session()
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    oof_ = np.zeros((len(train), 11)) # build out of folds matrix with 11 columns, they represent our target variables classes (from 0 to 10)
    preds_ = np.zeros((len(test), 11))
    target = ['open_channels']
    group = train['group']
    kf = GroupKFold(n_splits=4)
    splits = [x for x in kf.split(train, train[target], group)]

    new_splits = []
    for sp in splits:
        new_split = []
        new_split.append(np.unique(group[sp[0]]))
        new_split.append(np.unique(group[sp[1]]))
        new_split.append(sp[1])    
        new_splits.append(new_split)
    # pivot target columns to transform the net to a multiclass classification estructure (you can also leave it in 1 vector with sparsecategoricalcrossentropy loss function)
    tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)

    tr.columns = ['target_'+str(i) for i in range(11)] + ['group']
    target_cols = ['target_'+str(i) for i in range(11)]
    train_tr = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)
    train = np.array(list(train.groupby('group').apply(lambda x: x[feats].values)))
    test = np.array(list(test.groupby('group').apply(lambda x: x[feats].values)))

    for n_fold, (tr_idx, val_idx, val_orig_idx) in enumerate(new_splits[0:], start=0):
        train_x, train_y = train[tr_idx], train_tr[tr_idx]
        valid_x, valid_y = train[val_idx], train_tr[val_idx]
        print(f'Our training dataset shape is {train_x.shape}')
        print(f'Our validation dataset shape is {valid_x.shape}')

        gc.collect()
        shape_ = (None, train_x.shape[2]) # input is going to be the number of feature we are using (dimension 2 of 0, 1, 2)
        model = Classifier(shape_)
        # using our lr_schedule function
        cb_lr_schedule = LearningRateScheduler(lr_schedule)
        model.fit(train_x,train_y,
                  epochs = nn_epochs,
                  callbacks = [cb_lr_schedule, MacroF1(model, valid_x, valid_y)], # adding custom evaluation metric for each epoch
                  batch_size = nn_batch_size,verbose = 2,
                  validation_data = (valid_x,valid_y))
        preds_f = model.predict(valid_x)
        f1_score_ = f1_score(np.argmax(valid_y, axis=2).reshape(-1),  np.argmax(preds_f, axis=2).reshape(-1), average = 'macro') # need to get the class with the biggest probability
        print(f'Training fold {n_fold + 1} completed. macro f1 score : {f1_score_ :1.5f}')
        preds_f = preds_f.reshape(-1, preds_f.shape[-1])
        oof_[val_orig_idx,:] += preds_f
        te_preds = model.predict(test)
        te_preds = te_preds.reshape(-1, te_preds.shape[-1])           
        preds_ += te_preds / SPLITS
    # calculate the oof macro f1_score
    f1_score_ = f1_score(np.argmax(train_tr, axis = 2).reshape(-1),  np.argmax(oof_, axis = 1), average = 'macro') # axis 2 for the 3 Dimension array and axis 1 for the 2 Domension Array (extracting the best class)
    print(f'Training completed. oof macro f1 score : {f1_score_:1.5f}')
    sample_submission['open_channels'] = np.argmax(preds_, axis = 1).astype(int)
    sample_submission.to_csv('submission_wavenet.csv', index=False, float_format='%.4f')
    
# this function run our entire program
def run_everything():
    
    print('Reading Data Started...')
    train, test, sample_submission = read_data()
    train, test = normalize(train, test)
    print('Reading and Normalizing Data Completed')
        
    print('Creating Features')
    print('Feature Engineering Started...')
    train = run_feat_engineering(train, batch_size = GROUP_BATCH_SIZE)
    test = run_feat_engineering(test, batch_size = GROUP_BATCH_SIZE)
    train, test, features = feature_selection(train, test)
    print('Feature Engineering Completed...')
        
   
    print(f'Training Wavenet model with {SPLITS} folds of GroupKFold Started...')
    run_cv_model_by_batch(train, test, SPLITS, 'group', features, sample_submission, EPOCHS, NNBATCHSIZE)
    print('Training completed...')
        
run_everything()


# In[ ]:




