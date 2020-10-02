#!/usr/bin/env python
# coding: utf-8

# # reffrence  
# https://www.kaggle.com/martxelo/fe-and-ensemble-mlp-and-lgbm

# In[ ]:


import numpy as np 
import pandas as pd

# https://www.kaggle.com/friedchips/clean-removal-of-data-drift/output
import os
import random as rn
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#!pip install optuna
#import optuna


# In[ ]:


# imports
get_ipython().run_line_magic('matplotlib', 'inline')

import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, plot_confusion_matrix
from keras.models import Model
from keras.optimizers import Adagrad
import keras.layers as L
import lightgbm as lgb
import xgboost as xgb
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.pooling import MaxPool1D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.pooling import MaxPooling1D
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(7)
rn.seed(7)


# In[ ]:


MLP_EPOCH_NUM = 1#30
LGBM_BOOST_NUM = 1#3000
FOLD_NUM = 4


# # Load data

# In[ ]:


# read data
data = pd.read_csv('../input/data-without-drift/train_clean.csv')
data.iloc[478587:478588, [1]] = -2
data.iloc[478609:478610, [1]] = -2
data_ = data[3500000:3642922].append(data[3822754:4000000])
data = data[:3500000].append(data[4000000:]).reset_index().append(data_, ignore_index=True)
data.head()
data[["signal", "open_channels"]].plot(figsize=(19,5), alpha=0.7)


# # Feature engineering
# Add to signal several other signals: gradients, rolling mean, std, low/high pass filters...
# 
# FE is the same as this notebook https://www.kaggle.com/martxelo/fe-and-simple-mlp with corrections in filters.

# In[ ]:


def calc_gradients(s, n_grads=3):
    '''
    Calculate gradients for a pandas series. Returns the same number of samples
    '''
    grads = pd.DataFrame()
    
    g = s.values
    for i in range(n_grads):
        g = np.gradient(g)
        grads['grad_' + str(i+1)] = g
        
    return grads


# In[ ]:


def calc_low_pass(s, n_filts=10):
    '''
    Applies low pass filters to the signal. Left delayed and no delayed
    '''
    wns = np.logspace(-2, -0.9, n_filts)
    
    low_pass = pd.DataFrame()
    x = s.values
    for wn in wns:
        b, a = signal.butter(1, Wn=wn, btype='low')
        zi = signal.lfilter_zi(b, a)
        low_pass['lowpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]
        low_pass['lowpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)
        
    return low_pass


# In[ ]:


def calc_high_pass(s, n_filts=10):
    '''
    Applies high pass filters to the signal. Left delayed and no delayed
    '''
    wns = np.logspace(-2, -0.9, n_filts)
    
    high_pass = pd.DataFrame()
    x = s.values
    for wn in wns:
        b, a = signal.butter(1, Wn=wn, btype='high')
        zi = signal.lfilter_zi(b, a)
        high_pass['highpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]
        high_pass['highpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)
        
    return high_pass


# In[ ]:


def calc_roll_stats(s, windows=[3, 10, 50, 100, 500]):
    '''
    Calculates rolling stats like mean, std, min, max...
    '''
    roll_stats = pd.DataFrame()
    for w in windows:
        #roll_stats['roll_mean_2_' + str(w)] = s.rolling(window=2*w, min_periods=1).mean().shift(-w)
        #roll_stats['roll_std_2_' + str(w)] = s.rolling(window=2*w, min_periods=1).std().shift(-w)
        #roll_stats['roll_min_2_' + str(w)] = s.rolling(window=2*w, min_periods=1).min().shift(-w)
        #roll_stats['roll_max_2_' + str(w)] = s.rolling(window=2*w, min_periods=1).max().shift(-w)
        #roll_stats['roll_range_2_' + str(w)] = roll_stats['roll_max_2_' + str(w)] - roll_stats['roll_min_2_' + str(w)].shift(-w)
        roll_stats['roll_mean_' + str(w)] = s.rolling(window=w, min_periods=1).mean()
        roll_stats['roll_std_' + str(w)] = s.rolling(window=w, min_periods=1).std()
        roll_stats['roll_min_' + str(w)] = s.rolling(window=w, min_periods=1).min()
        roll_stats['roll_max_' + str(w)] = s.rolling(window=w, min_periods=1).max()
        roll_stats['roll_range_' + str(w)] = roll_stats['roll_max_' + str(w)] - roll_stats['roll_min_' + str(w)]
        roll_stats['roll_mean_s_' + str(w)] = s.rolling(window=w, min_periods=1).mean().shift(-w)
        roll_stats['roll_std_s_' + str(w)] = s.rolling(window=w, min_periods=1).std().shift(-w)
        roll_stats['roll_min_s_' + str(w)] = s.rolling(window=w, min_periods=1).min().shift(-w)
        roll_stats['roll_max_s_' + str(w)] = s.rolling(window=w, min_periods=1).max().shift(-w)
        roll_stats['roll_range_s_' + str(w)] = roll_stats['roll_max_s_' + str(w)] - roll_stats['roll_min_s_' + str(w)]
        roll_stats['roll_min_abs_' + str(w)] = s.rolling(window=2*w, min_periods=1).min().abs().shift(-w)
        roll_stats['roll_range_sbs_' + str(w)] = roll_stats['roll_max_' + str(w)] - roll_stats['roll_min_abs_' + str(w)].shift(-w)
        roll_stats['roll_q10_' + str(w)] = s.rolling(window=2*w, min_periods=1).quantile(0.10).shift(-w)
        roll_stats['roll_q25_' + str(w)] = s.rolling(window=2*w, min_periods=1).quantile(0.25).shift(-w)
        roll_stats['roll_q50_' + str(w)] = s.rolling(window=2*w, min_periods=1).quantile(0.50).shift(-w)
        roll_stats['roll_q75_' + str(w)] = s.rolling(window=2*w, min_periods=1).quantile(0.75).shift(-w)
        roll_stats['roll_q90_' + str(w)] = s.rolling(window=2*w, min_periods=1).quantile(0.90).shift(-w)
        roll_stats['mean_abs_chg' + str(w)] = roll_stats.apply(lambda x: np.mean(np.abs(np.diff(x))))
    
    # add zeros when na values (std)
    roll_stats = roll_stats.fillna(value=0)
             
    return roll_stats


# In[ ]:


def calc_ewm(s, windows=[10, 50, 100, 1000]):
    '''
    Calculates exponential weighted functions
    '''
    ewm = pd.DataFrame()
    for w in windows:
        ewm['ewm_mean_' + str(w)] = s.ewm(span=w, min_periods=1).mean()
        ewm['ewm_std_' + str(w)] = s.ewm(span=w, min_periods=1).std()
        
    # add zeros when na values (std)
    ewm = ewm.fillna(value=0)
        
    return ewm


# In[ ]:


def add_features(s):
    '''
    All calculations together
    '''
    gradients = calc_gradients(s)
    low_pass = calc_low_pass(s)
    high_pass = calc_high_pass(s)
    roll_stats = calc_roll_stats(s)
    ewm = calc_ewm(s)
    
    return pd.concat([s, gradients, low_pass, high_pass, roll_stats, ewm], axis=1)


def divide_and_add_features(s, signal_size=500000):
    '''
    Divide the signal in bags of "signal_size".
    Normalize the data dividing it by 15.0
    '''
    # normalize
    s = s/15.0
    
    ls = []
    for i in tqdm(range(int(s.shape[0]/signal_size))):
        sig = s[i*signal_size:(i+1)*signal_size].copy().reset_index(drop=True)
        sig_featured = add_features(sig)
        ls.append(sig_featured)
    if len(s) > 4000000:
        sig = s[(i+1)*signal_size:4820168].copy().reset_index(drop=True)
        sig_featured = add_features(sig)
        ls.append(sig_featured)
    
    df = pd.concat(ls, axis=0)
    df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])
    df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]
    df['signal_shift_+2'] = [0,] + [1,] + list(df['signal'].values[:-2])
    df['signal_shift_-2'] = list(df['signal'].values[2:]) + [0] + [1]
    df['signal_shift_+3'] = [0,] + [1,] + [1,] + list(df['signal'].values[:-3])
    df['signal_shift_-3'] = list(df['signal'].values[3:]) + [0] + [1] + [2]
    return df


# In[ ]:


# apply every feature to data
df = divide_and_add_features(data['signal'])
df.head()


# Let's plot the signals to see how they look like.

# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
df = reduce_mem_usage(df)


# In[ ]:


# read m_data
m_data = pd.read_csv('../input/viterbi/Viterbi_train.csv', names=["gb", "signal"])
m_data_ = m_data[3500000:3642922].append(m_data[3822754:4000000])
m_data = m_data[:3500000].append(m_data[4000000:]).reset_index().append(m_data_, ignore_index=True)
m_data.head()
m_data[["signal"]].plot(figsize=(19,5), alpha=0.7)
df["m_signal"] = m_data["signal"][:-1].values/15
df = reduce_mem_usage(df)
df_columns = df.columns


# # Classes weights

# In[ ]:


def get_class_weight(classes, exp=1):
    '''
    Weight of the class is inversely proportional to the population of the class.
    There is an exponent for adding more weight.
    '''
    hist, _ = np.histogram(classes, bins=np.arange(12)-0.5)
    class_weight = hist.sum()/np.power(hist, exp)
    
    return class_weight


# # Build a MLP model

# In[ ]:


def create_mpl(shape):
    '''
    Returns a keras model
    '''
    
    model = Sequential()
    model.add(Conv1D(128,3,input_shape=shape))
    model.add(Activation('relu'))
    model.add(Conv1D(128,3))
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))

    model.add(Conv1D(252,3))
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(1.0))

    model.add(L.Dense(11, activation='softmax'))
    
    return model


# In[ ]:


def lgb_Metric(preds, dtrain):
    labels = dtrain.get_label()
    num_labels = 11
    preds = preds.reshape(num_labels, len(preds)//num_labels)
    preds = np.argmax(preds, axis=0)
    score = f1_score(labels, preds, average="macro")
    return ('KaggleMetric', score, True)
lgb_r_params = {
    'objective': 'multiclass',
    'num_class': 11,
    'metric': 'multi_logloss',
    'learning_rate': 0.00987173774816051,
    'lambda_l1': 0.00031963798315506463,
    'lambda_l2': 0.18977456778807847,
    'num_leaves': 171, 
    'feature_fraction': 0.58733782457345, 
    'bagging_fraction': 0.7057826081907392, 
    'bagging_freq': 4}


# In[ ]:


import pandas as pd
test_data = pd.read_csv('../input/data-without-drift/test_clean.csv')

test_df = divide_and_add_features(test_data['signal'])
test_df = reduce_mem_usage(test_df)
# read m_data
m_data = pd.read_csv('../input/viterbi/Viterbi_test.csv').rename(columns={"open_channels": "signal"})
m_data_ = m_data[3500000:3642922].append(m_data[3822754:4000000])
m_data = m_data[:3500000].append(m_data[4000000:]).reset_index().append(m_data_, ignore_index=True)
m_data.head()
m_data[["signal"]].plot(figsize=(19,5), alpha=0.7)
test_df["m_signal"] = m_data["signal"].values/15
test_df = reduce_mem_usage(test_df)
test_df.shape


# In[ ]:


kf = KFold(n_splits=FOLD_NUM, shuffle=True, random_state=42)

preds = np.zeros(2000000*11).reshape((2000000, 11))

X = df#.values
y = data['open_channels']
y_values = data['open_channels'].values

for i, (tdx, vdx) in enumerate(kf.split(X, y)):
    print(f'Fold : {i}')
    X_train, X_valid, y_train, y_valid = X.iloc[tdx], X.iloc[vdx], y_values[tdx], y_values[vdx]
    #X_train, X_valid, y_train, y_valid = X[tdx], X[vdx], y_values[tdx], y_values[vdx]
    print(f"sep: {X_train.shape}, {X_valid.shape}, {y_train.shape}, {y_valid.shape}")
    
    #MLP
    mlp = create_mpl((X_train.values.shape[1], 1))
    #mlp = create_mpl(X_train[0].shape)
    mlp.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    class_weight = get_class_weight(y_train)

    mlp.fit(x=np.reshape(X_train.values, (-1, X_train.shape[1], 1)), y=y_train, epochs=MLP_EPOCH_NUM, batch_size=1024, 
            class_weight=class_weight,
            validation_data=(np.reshape(X_valid.values, (-1, X_valid.shape[1], 1)), y_valid))
    #mlp.fit(x=X_train, y=y_train, epochs=MLP_EPOCH_NUM, batch_size=1024, class_weight=class_weight,
    #       validation_data=(X_valid, y_valid), verbose=0)
    mlp_pred = mlp.predict(np.reshape(X_valid.values, (-1, X_valid.shape[1], 1)))
    #mlp_pred = mlp.predict(X_valid)
    f1_mlp = f1_score(y_valid, np.argmax(mlp_pred, axis=-1), average='macro')
    print(f"f1 score is :{f1_mlp}")
    plt.figure(1)
    plt.plot(mlp.history.history['loss'], 'b', label='loss')
    plt.plot(mlp.history.history['val_loss'], 'r', label='loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.figure(2)
    plt.plot(mlp.history.history['sparse_categorical_accuracy'], 'g', label='sparse_categorical_accuracy')
    plt.plot(mlp.history.history['val_sparse_categorical_accuracy'], 'r', label='val_sparse_categorical_accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    
    # GBC
    #lgb_dataset = lgb.Dataset(X_train, label=y_train, weight=class_weight[y_train])
    lgb_dataset = lgb.Dataset(X_train.values, label=y_train, weight=class_weight[y_train])
    #lgb_valid_dataset = lgb.Dataset(X_valid, label=y_valid, weight=class_weight[y_valid])
    lgb_valid_dataset = lgb.Dataset(X_valid.values, label=y_valid, weight=class_weight[y_valid])
    print('Training LGBM...')
    gbc = lgb.train(lgb_r_params, lgb_dataset, LGBM_BOOST_NUM, valid_names=["train", "valid"], 
                    valid_sets=[lgb_dataset, lgb_valid_dataset], verbose_eval=-1, 
                    feval=lgb_Metric, early_stopping_rounds=10)
    print('LGBM trained!')
    # predict on test
    gbc_pred = gbc.predict(X_valid.values, num_iteration=gbc.best_iteration)
    attr2 = {k: v for k, v in zip(df_columns, gbc.feature_importance()) if 200 > v and v>0}
    print("weak fe##############")
    print(attr2)
    print("##############")
    #gbc_pred = gbc.predict(X_valid, num_iteration=gbc.best_iteration)
    print(f1_score(y_valid, np.argmax(gbc_pred, axis=1), average='macro'))
    
    # lists for keep results
    f1s = []
    alphas = []

    # loop for every alpha
    for alpha in tqdm(np.linspace(0,1,101)):
        #y_pred = alpha*mlp_pred + (1 - alpha)*np.round(np.clip(gbc_pred, 0, 10)).astype(int)
        y_pred = alpha*mlp_pred + (1 - alpha)*gbc_pred
        f1 = f1_score(y_valid, np.argmax(y_pred, axis=1), average='macro')
        f1s.append(f1)
        alphas.append(alpha)

    # convert to numpy arrays
    f1s = np.array(f1s)
    alphas = np.array(alphas)

    # get best_alpha
    best_alpha = alphas[np.argmax(f1s)]

    print('best_f1=', f1s.max())
    print('best_alpha=', best_alpha)
    plt.plot(alphas, f1s)
    plt.title('f1_score for ensemble')
    plt.xlabel('alpha')
    plt.ylabel('f1_score')
    plt.show()

    mlp_pred = mlp.predict(np.reshape(test_df.values, (-1, test_df.shape[1], 1)))
    #mlp_pred = mlp.predict(test_df.values)
    gbc_pred = gbc.predict(test_df, num_iteration=gbc.best_iteration)
    #gbc_pred = gbc.predict(test_df.values, num_iteration=gbc.best_iteration)
    pred = best_alpha*mlp_pred + (1 - best_alpha)*gbc_pred
    preds += pred
    
    
    print(f"f1_mlp is {f1_mlp}")
    gc.collect()


# # Submit result

# In[ ]:


pred = np.argmax(preds, axis=1)

print('Writing submission...')
submission = pd.DataFrame()
submission['time'] = test_data['time']

submission['open_channels'] = pred
submission.to_csv('mlp_gbc_submission.csv', index=False, float_format='%.4f')

print('Submission finished!')


# In[ ]:




