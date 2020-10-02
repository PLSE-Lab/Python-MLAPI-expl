#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import Counter
from contextlib import contextmanager
import gc
import os
import psutil
import time
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
#from sklearn.preprocessing import StandardScaler
#from tsfresh.feature_extraction import feature_calculators
from tqdm import tqdm_notebook as tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import random as rn
import scipy as sp
import itertools
import warnings
# import librosa
import pywt
import math
import warnings
warnings.filterwarnings("ignore")
from scipy import signal
from keras.models import Model
import keras.layers as L
from sklearn import preprocessing
def normalize(X_train, X_test, normalize_opt, feats):
    if normalize_opt != None:
        if normalize_opt == 'min_max':
            scaler = preprocessing.MinMaxScaler()
        elif normalize_opt == 'robust':
            scaler = preprocessing.RobustScaler()
        elif normalize_opt == 'standard':
            scaler = preprocessing.StandardScaler()
        elif normalize_opt == 'max_abs':
            scaler = preprocessing.MaxAbsScaler()
        scaler = scaler.fit(X_train[feats])
        X_train[feats] = scaler.transform(X_train[feats])
        X_test[feats] = scaler.transform(X_test[feats])
    return X_train, X_test


# In[ ]:


def reduce_mem_usage(df,verbose=True):
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if (c_min > np.iinfo(np.int8).min
                        and c_max < np.iinfo(np.int8).max):
                    df[col] = df[col].astype(np.int8)
                elif (c_min > np.iinfo(np.int16).min
                      and c_max < np.iinfo(np.int16).max):
                    df[col] = df[col].astype(np.int16)
                elif (c_min > np.iinfo(np.int32).min
                      and c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min
                      and c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min
                      and c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem

    msg = f'Mem. usage decreased to {end_mem:5.2f} MB ({reduction * 100:.1f} % reduction)'
    if verbose:
        print(msg)

    return df



@contextmanager
def timer(title, new_line=True):
    """
    USAGE:
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how="left", on="SK_ID_CURR")
        del bureau
        gc.collect()
    """
    t0 = time.time()
    yield
    print(f"{title} - done in {time.time() - t0:.0f}s")
    if new_line:
        print()
        
        
def report_process_mem_usage():
    """
    Print memory usage (in GB) of main process
    """
    print(f"Main process memory usage: "
          f"{psutil.Process(os.getpid()).memory_info().rss/1024**3:.2f} GB")
 


   
def evaluate_macroF1(data_vali, preds):  
    labels = data_vali.astype(int)
    preds = np.array(preds)
    preds = np.argmax(preds,axis=1)
    score_vali = f1_score(y_true=labels,y_pred=preds,average='macro')
    return  score_vali

def get_class_weight(classes, exp=1):
    '''
    Weight of the class is inversely proportional to the population of the class.
    There is an exponent for adding more weight.
    '''
    hist, _ = np.histogram(classes, bins=np.arange(12)-0.5)
    class_weight = hist.sum()/np.power(hist, exp)
    
    return class_weight
    

    
def create_mpl(shape):
    '''
    Returns a keras model
    '''
    
    X_input = L.Input(shape)
    
    X = L.Dense(150, activation='relu')(X_input)
    X = L.Dense(150, activation='relu')(X)
    X = L.Dense(125, activation='relu')(X)
    X = L.Dense(75, activation='relu')(X)
    X = L.Dense(50, activation='relu')(X)
    X = L.Dense(25, activation='relu')(X)
    X = L.Dense(1)(X)
    
    model = Model(inputs=X_input, outputs=X)
    
    return model





def calc_gradients(s, n_grads=4):
    '''
    Calculate gradients for a pandas series. Returns the same number of samples
    '''
    grads = pd.DataFrame()
    
    g = s.values
    for i in range(n_grads):
        g = np.gradient(g)
        grads['grad_' + str(i+1)] = g
        
    return grads


def calc_low_pass(s, n_filts=10):
    '''
    Applies low pass filters to the signal. Left delayed and no delayed
    '''
    wns = np.logspace(-2, -0.3, n_filts)
    
    low_pass = pd.DataFrame()
    x = s.values
    for wn in wns:
        b, a = signal.butter(1, Wn=wn, btype='low')
        zi = signal.lfilter_zi(b, a)
        low_pass['lowpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]
        low_pass['lowpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)
        
    return low_pass


def calc_high_pass(s, n_filts=10):
    '''
    Applies high pass filters to the signal. Left delayed and no delayed
    '''
    wns = np.logspace(-2, -0.1, n_filts)
    
    high_pass = pd.DataFrame()
    x = s.values
    for wn in wns:
        b, a = signal.butter(1, Wn=wn, btype='high')
        zi = signal.lfilter_zi(b, a)
        high_pass['highpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]
        high_pass['highpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)
        
    return high_pass


def calc_roll_stats(s, windows=[10, 50, 100, 500, 1000]):
    '''
    Calculates rolling stats like mean, std, min, max...
    '''
    roll_stats = pd.DataFrame()
    for w in windows:
        roll_stats['roll_mean_' + str(w)] = s.rolling(window=w, min_periods=1).mean()
        roll_stats['roll_std_' + str(w)] = s.rolling(window=w, min_periods=1).std()
        roll_stats['roll_min_' + str(w)] = s.rolling(window=w, min_periods=1).min()
        roll_stats['roll_max_' + str(w)] = s.rolling(window=w, min_periods=1).max()
        roll_stats['roll_range_' + str(w)] = roll_stats['roll_max_' + str(w)] - roll_stats['roll_min_' + str(w)]
        roll_stats['roll_q10_' + str(w)] = s.rolling(window=w, min_periods=1).quantile(0.10)
        roll_stats['roll_q25_' + str(w)] = s.rolling(window=w, min_periods=1).quantile(0.25)
        roll_stats['roll_q50_' + str(w)] = s.rolling(window=w, min_periods=1).quantile(0.50)
        roll_stats['roll_q75_' + str(w)] = s.rolling(window=w, min_periods=1).quantile(0.75)
        roll_stats['roll_q90_' + str(w)] = s.rolling(window=w, min_periods=1).quantile(0.90)
    
    # add zeros when na values (std)
    roll_stats = roll_stats.fillna(value=0)
             
    return roll_stats

def calc_ewm(s, windows=[10, 50, 100, 500, 1000]):
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


def add_features(s):
    '''
    All calculations together
    '''
    
    # gradients = calc_gradients(s)
    # low_pass = calc_low_pass(s)
    high_pass = calc_high_pass(s)
    roll_stats = calc_roll_stats(s)
    ewm = calc_ewm(s)
    temp=pd.concat([s, high_pass, roll_stats, ewm], axis=1)
    return temp


def divide_and_add_features(s, signal_size=100000):
    '''
    Divide the signal in bags of "signal_size".
    Normalize the data dividing it by 15.0
    '''
    # normalize
    # s = s/15.0
    
    ls = []
    for i in tqdm(range(int(s.shape[0]/signal_size))):
        sig = s[i*signal_size:(i+1)*signal_size].copy().reset_index(drop=True)
        sig_featured = add_features(sig)
        ls.append(sig_featured)

    return pd.concat(ls, axis=0)


# In[ ]:


import random
import tensorflow as tf
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    
SEED = 321
seed_everything(SEED)


# In[ ]:


with timer("Load saved features from disk"):
    submission  = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv') 
    train = pd.read_csv("/kaggle/input/remove-trends-giba/train_clean_giba.csv", usecols=["signal","open_channels"], dtype={'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv("/kaggle/input/remove-trends-giba/test_clean_giba.csv", usecols=["signal"], dtype={'signal': np.float32})

    # 5+8 Augmentation to Create new batch with 10 channels
    train['group'] = np.arange(train.shape[0])//500_000
    aug_df = train[train["group"] == 5].copy()
    aug_df["group"] = 10

    for col in ["signal", "open_channels"]:
        aug_df[col] += train[train["group"] == 8][col].values

    train = train.append(aug_df, sort=False).reset_index(drop=True)
    del aug_df

    y=train['open_channels']
    del train['open_channels']
    gc.collect()

    report_process_mem_usage()


# In[ ]:


for item in ['signal']:
    if item in train.columns:
        print(item)
        train_input_mean = train[item].mean()
        train_input_sigma = train[item].std()
        train[item]= (train[item] - train_input_mean) / train_input_sigma
        test[item] = (test[item] - train_input_mean) / train_input_sigma


# In[ ]:


with timer("public features"): 
    train= divide_and_add_features(train['signal'],signal_size=100_000)
    test= divide_and_add_features(test['signal'],signal_size=100_000)


# In[ ]:


n_splits=5    
remove_fea=['time','batch','batch_index','batch_slices','batch_slices2','group']
features=[i for i in train.columns if i not in remove_fea]

with timer("train lgb model:"):
    cv_result = []
    cv_pred = []
    oof_preds = np.zeros(train.shape[0])
    y_preds = np.zeros(test.shape[0]) 

    target = "open_channels"
    train['group'] = np.arange(train.shape[0])//4000
    group = train['group']
    kf = GroupKFold(n_splits=5)
    splits = [x for x in kf.split(train, y, group)]

    for fold, (tr_ind, val_ind) in enumerate(splits):
        x_train, x_val = train[features].iloc[tr_ind].values, train[features].iloc[val_ind].values
        y_train, y_val = y[tr_ind].values, y[val_ind].values
        print(f'Fold {fold + 1}, {x_train.shape}, {x_val.shape}')
        class_weight = get_class_weight(y_train)
        print(x_train[0].shape)
        #weight=class_weight[y_train]
        mlp = create_mpl(x_train[0].shape)
        mlp.compile(optimizer='adam', loss='mean_squared_error')
        mlp.fit(x=x_train, y=y_train, epochs=50, batch_size=256, class_weight=class_weight, verbose = 1)  
        del x_train,y_train
        gc.collect()
        oof_preds[val_ind] = mlp.predict(x_val).reshape(x_val.shape[0],)
        del x_val
        gc.collect()
        result = f1_score(y_val,np.round(np.clip(oof_preds[val_ind], 0, 10)).astype(int),average='macro')
        print('f1 score : ',result)
        cv_result.append(round(result,5))
        y_preds += mlp.predict(test[features]).reshape(test.shape[0],)/n_splits


# In[ ]:


f1_score(y,np.round(np.clip(oof_preds, 0, 10)).astype(int),average='macro')


# In[ ]:


f1_score(y[:5000_000],np.round(np.clip(oof_preds[:5000_000], 0, 10)).astype(int),average='macro')


# In[ ]:


np.savez_compressed('mlp_reg.npz',valid=oof_preds, test=y_preds)


# In[ ]:


# report OOF RMSE and QWK
print(cv_result)
f1_mean,f1_std = np.mean(cv_result),np.std(cv_result)
print(f"[CV] F1 Mean: {f1_mean}")
print(f"[CV] F1 Std: {f1_std}")


# In[ ]:


# make test predictions with optimized coefficients
sub_preds = np.round(np.clip(y_preds, 0, 10)).astype(int)
submission['open_channels'] = sub_preds
print(submission['open_channels'].value_counts()) 
submission.to_csv("submission.csv",index=False)


# In[ ]:





# In[ ]:




