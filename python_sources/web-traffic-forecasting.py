#!/usr/bin/env python
# coding: utf-8

# # Web Traffic Time Series Forecasting
# 
# This competition focuses on the problem of forecasting the future values of multiple time series, as it has always been one of the most challenging problems in the field. More specifically, we aim the competition at testing state-of-the-art methods designed by the participants, on the problem of forecasting future web traffic for approximately 145,000 Wikipedia articles.

# In[ ]:


import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.max_rows = 10
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 600
from tqdm import tqdm
import gc
from tqdm import tqdm_notebook
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.decomposition import PCA
import pandas_profiling
from keras.layers.normalization import BatchNormalization

from keras.models import Sequential, Model

from keras.layers import Input, Embedding, Dense, Activation, Dropout, Flatten

from keras import regularizers 

import keras
from collections import Counter
import re
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GroupKFold


# # Data Analysis and Basi EDA, Feature Engineering

# In[ ]:


df_train = pd.read_csv('../input/train_1.csv')
df_train.shape
df_train.head()


# In[ ]:


df_train.info()
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in tqdm_notebook(df.columns):
        col_type = df[col].dtype

        if col_type != object:
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

df_train = reduce_mem_usage(df_train)


# In[ ]:


profile = pandas_profiling.ProfileReport(df_train.sample(50000))


# # Language in Pages
# One thing that might be interesting to look it is how the different languages used in Wikipedia might affect the dataset. I'll use a simple regular expression to search for the language code in the wikipedia URL. There are also a number of non-wikipedia URLs that will fail the regex search. These are wikimedia pages, so I'll give them the code 'na' since I haven't determined their language. Many of these will be things like images that do not really have a language.

# In[ ]:


def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res[0][0:2]
    return 'na'

df_train['lang'] = df_train.Page.map(get_language)


print(Counter(df_train.lang))


# In[ ]:


lang_sets = {}
lang_sets['en'] = df_train[df_train.lang=='en'].iloc[:,0:-1]
lang_sets['ja'] = df_train[df_train.lang=='ja'].iloc[:,0:-1]
lang_sets['de'] = df_train[df_train.lang=='de'].iloc[:,0:-1]
lang_sets['na'] = df_train[df_train.lang=='na'].iloc[:,0:-1]
lang_sets['fr'] = df_train[df_train.lang=='fr'].iloc[:,0:-1]
lang_sets['zh'] = df_train[df_train.lang=='zh'].iloc[:,0:-1]
lang_sets['ru'] = df_train[df_train.lang=='ru'].iloc[:,0:-1]
lang_sets['es'] = df_train[df_train.lang=='es'].iloc[:,0:-1]

sums = {}
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]
days = [r for r in range(sums['en'].shape[0])]

fig = plt.figure(1,figsize=[10,10])
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Pages in Different Languages')
labels={'en':'English','ja':'Japanese','de':'German',
        'na':'Media','fr':'French','zh':'Chinese',
        'ru':'Russian','es':'Spanish'
       }

for key in sums:
    plt.plot(days,sums[key],label = labels[key] )
    
plt.legend()
plt.show()


# English shows a much higher number of views per page, as might be expected since Wikipedia is a US-based site. There is a lot more structure here than I would have expected. The English and Russian plots show very large spikes around day 400 (around August 2016), with several more spikes in the English data later in 2016. My guess is that this is the effect of both the Summer Olympics in August and the election in the US.
# 
# There's also a strange feature in the English data around day 200.
# 
# The Spanish data is very interesting too. There is a clear periodic structure there, with a ~1 week fast period and what looks like a significant dip around every 6 months or so.

# # Periodic Structure and FFTs
# Since it looks like there is some periodic structure here, I will plot each of these separately so that the scale is more visible. Along with the individual plots, I will also look at the magnitude of the Fast Fourier Transform (FFT). Peaks in the FFT show us the strongest frequencies in the periodic signal.

# In[ ]:


from scipy.fftpack import fft
def plot_with_fft(key):

    fig = plt.figure(1,figsize=[15,5])
    plt.ylabel('Views per Page')
    plt.xlabel('Day')
    plt.title(labels[key])
    plt.plot(days,sums[key],label = labels[key] )
    
    fig = plt.figure(2,figsize=[15,5])
    fft_complex = fft(sums[key])
    fft_mag = [np.sqrt(np.real(x)*np.real(x)+np.imag(x)*np.imag(x)) for x in fft_complex]
    fft_xvals = [day / days[-1] for day in days]
    npts = len(fft_xvals) // 2 + 1
    fft_mag = fft_mag[:npts]
    fft_xvals = fft_xvals[:npts]
        
    plt.ylabel('FFT Magnitude')
    plt.xlabel(r"Frequency [days]$^{-1}$")
    plt.title('Fourier Transform')
    plt.plot(fft_xvals[1:],fft_mag[1:],label = labels[key] )
    # Draw lines at 1, 1/2, and 1/3 week periods
    plt.axvline(x=1./7,color='red',alpha=0.3)
    plt.axvline(x=2./7,color='red',alpha=0.3)
    plt.axvline(x=3./7,color='red',alpha=0.3)

    plt.show()

for key in sums:
    plot_with_fft(key)


# # Some individual Entries
# 

# In[ ]:


def plot_entry(key,idx):
    data = lang_sets[key].iloc[idx,1:]
    fig = plt.figure(1,figsize=(10,5))
    plt.plot(days,data)
    plt.xlabel('day')
    plt.ylabel('views')
    plt.title(df_train.iloc[lang_sets[key].index[idx],0])
    
    plt.show()

idx = [10, 50, 100, 250,500,750,1000,1500,2000,3000,4000,5000]
for i in idx:
    plot_entry('en',i)


# # Lets look at the top pages
# 

# In[ ]:


npages = 5
top_pages = {}
for key in lang_sets:
    sum_set = pd.DataFrame(lang_sets[key][['Page']])
    sum_set['total'] = lang_sets[key].sum(axis=1)
    sum_set = sum_set.sort_values('total',ascending=False)
    top_pages[key] = sum_set.index[0]


# # Making Predictions
# The baseline given so far is for guessing 0 views for everything. There are a few other easy benchmarks we can try out:
# 
# Average number of views for that page (constant value)
# Linear regression
# More complicated regression curves
# But we've also seen that the data for related topics seems to be correlated and that topics that are in the news get a lot of traffic, so this maybe points to some ways to improve things. Unfortunately, with the different languages, training a model to identify related topics may be quite difficult. However, trying to cluster similar topics using just the data rather than the page name might help us out a bit. We could potentially smooth out some of the unpredictable spikes and could maybe use high-view pages to reduce effects of statistical fluctuations on low-view pages.
# 
# I also wonder if something like a recurrent neural net might help out here if we want to try out more complicated methods.
# 
# As an example, I look at ARIMA models for a small set of pages.
# 
# # ARIMA Models
# Statsmodels also includes things like ARMA and ARIMA models that can be used to make predictions from time series. This data is not necessarily very stationary and often has strong periodic effects, so these may not necessarily work very well. I'll look at ARIMA predictions for the same set of very high viewcount pages.

# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
import warnings

cols = df_train.columns[1:-1]
for key in top_pages:
    data = np.array(df_train.loc[top_pages[key],cols],'f')
    result = None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            arima = ARIMA(data,[2,1,4])
            result = arima.fit(disp=False)
        except:
            try:
                arima = ARIMA(data,[2,1,2])
                result = arima.fit(disp=False)
            except:
                print(df_train.loc[top_pages[key],'Page'])
                print('\tARIMA failed')
    #print(result.params)
    pred = result.predict(2,599,typ='levels')
    x = [i for i in range(600)]
    i=0

    plt.plot(x[2:len(data)],data[2:] ,label='Data')
    plt.plot(x[2:],pred,label='ARIMA Model')
    plt.title(df_train.loc[top_pages[key],'Page'])
    plt.xlabel('Days')
    plt.ylabel('Views')
    plt.legend()
    plt.show()


# # The Final Model
# 
# Based on the Solutions provided by Arthur Suilin,CPMP and thousandvoices, Although they are 1st,2nd,3rd place, my model recreation would have got me 9th place. although its just a simple implementation!
# 
# Based on the following:
# The NN is a feedforward network with 200, 200, 100, 200 cells in each layer. Input is concatenated again with the output of the first layer. I don't know why but this boosted accuracy. Activation is relu except for last one which is linear. I used dropout of 0.5 for almost all layers (0.5 was selected by CV). I also used a batch normalization for the middle layer. Model is compiled with adam optimizer and the loss function defined above.I used 5 fold Cv for NN Batch size was huge: 4096. I found that increasing batch size was both decreasing training time, and improving accuracy. Training went for 200 epochs, computing actual training SMAPE every 10 epochs across all folds, and using median of all predictions of best epoch for submission. For each fold I trained 1 model, and took the median of their predictions

# In[ ]:


def init():
    np.random.seed = 0
    
init()


# In[ ]:


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

def smape2D(y_true, y_pred):
    return smape(np.ravel(y_true), np.ravel(y_pred))
    
def smape_mask(y_true, y_pred, threshold):
    denominator = (np.abs(y_true) + np.abs(y_pred)) 
    diff = np.abs(y_true - y_pred) 
    diff[denominator == 0] = 0.0
    
    return diff <= (threshold / 2.0) * denominator


# In[ ]:


max_size = 181 # number of days in 2015 with 3 days before end

offset = 1/2

train_all = pd.read_csv("../input/train_2.csv")
train_all.head()


# In[ ]:


all_page = train_all.Page.copy()
train_key = train_all[['Page']].copy()
train_all = train_all.iloc[:,1:] * offset 
train_all.head()


# In[ ]:


def get_date_index(date, train_all=train_all):
    for idx, c in enumerate(train_all.columns):
        if date == c:
            break
    if idx == len(train_all.columns):
        return None
    return idx


# In[ ]:


get_date_index('2016-09-13')


# In[ ]:


get_date_index('2016-09-10')


# In[ ]:


train_all.shape[1] - get_date_index('2016-09-10')


# In[ ]:


get_date_index('2017-09-10') - get_date_index('2016-09-10')


# In[ ]:


train_end = get_date_index('2016-09-10') + 1
test_start = get_date_index('2016-09-13')

train = train_all.iloc[ : , (train_end - max_size) : train_end].copy().astype('float32')
test = train_all.iloc[:, test_start : (63 + test_start)].copy().astype('float32')
train = train.iloc[:,::-1].copy().astype('float32')

train_all = train_all.iloc[:,-(max_size):].astype('float32')
train_all = train_all.iloc[:,::-1].copy().astype('float32')

test_3_date = test.columns


# In[ ]:


train_all.head()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


data = [page.split('_') for page in tqdm(train_key.Page)]

access = ['_'.join(page[-2:]) for page in data]

site = [page[-3] for page in data]

page = ['_'.join(page[:-3]) for page in data]
page[:2]

train_key['PageTitle'] = page
train_key['Site'] = site
train_key['AccessAgent'] = access
train_key.head()


# In[ ]:


train_norm = np.log1p(train).astype('float32')
train_norm.head()


# In[ ]:


train_all_norm = np.log1p(train_all).astype('float32')
train_all_norm.head()


# In[ ]:


first_day = 1 # 2016-09-13 is a Tuesday
test_columns_date = list(test.columns)
test_columns_code = ['w%d_d%d' % (i // 7, (first_day + i) % 7) for i in range(63)]
test.columns = test_columns_code

test.head()


# In[ ]:


test.fillna(0, inplace=True)

test['Page'] = all_page
test.sort_values(by='Page', inplace=True)
test.reset_index(drop=True, inplace=True)


# In[ ]:


test = test.merge(train_key, how='left', on='Page', copy=False)

test.head()


# In[ ]:


test_all_id = pd.read_csv('../input/key_2.csv')

test_all_id['Date'] = [page[-10:] for page in tqdm(test_all_id.Page)]
test_all_id['Page'] = [page[:-11] for page in tqdm(test_all_id.Page)]
test_all_id.head()


# In[ ]:


test_all = test_all_id.drop('Id', axis=1)
test_all['Visits_true'] = np.NaN

test_all.Visits_true = test_all.Visits_true * offset
test_all = test_all.pivot(index='Page', columns='Date', values='Visits_true').astype('float32').reset_index()

test_all['2017-11-14'] = np.NaN
test_all.sort_values(by='Page', inplace=True)
test_all.reset_index(drop=True, inplace=True)

test_all.head()


# In[ ]:


test_all.shape


# In[ ]:


test_all_columns_date = list(test_all.columns[1:])
first_day = 2 # 2017-13-09 is a Wednesday
test_all_columns_code = ['w%d_d%d' % (i // 7, (first_day + i) % 7) for i in range(63)]
cols = ['Page']
cols.extend(test_all_columns_code)
test_all.columns = cols
test_all.head()


# In[ ]:


test_all = test_all.merge(train_key, how='left', on='Page')
test_all.head()


# In[ ]:


y_cols = test.columns[:63]
y_cols


# In[ ]:


test = test.reset_index()
test_all = test_all.reset_index()


# In[ ]:


test_all.shape


# In[ ]:


test.shape


# In[ ]:


test.head()


# In[ ]:


test_all = test_all[test.columns].copy()
test_all.head()


# In[ ]:


train_cols = ['d_%d' % i for i in range(train_norm.shape[1])]
len(train_cols)


# In[ ]:


train_norm.columns = train_cols
train_all_norm.columns = train_cols


# In[ ]:


train_norm.head()


# In[ ]:


all(test[:test_all.shape[0]].Page == test_all.Page)


# In[ ]:


sites = train_key.Site.unique()
sites


# In[ ]:


test_site = pd.factorize(test.Site)[0]
test['Site_label'] = test_site
test_all['Site_label'] = test_site[:test_all.shape[0]]


# In[ ]:


accesses = train_key.AccessAgent.unique()
accesses


# In[ ]:


test_access = pd.factorize(test.AccessAgent)[0]
test['Access_label'] = test_access
test_all['Access_label'] = test_access[:test_all.shape[0]]


# In[ ]:


test.shape


# In[ ]:


test_all.shape


# In[ ]:


test0 = test.copy()
test_all0 = test_all.copy()


# In[ ]:


y_norm_cols = [c+'_norm' for c in y_cols]
y_pred_cols = [c+'_pred' for c in y_cols]


# In[ ]:


# all visits is median
def add_median(test, train,
               train_key, periods, max_periods, first_train_weekday):
    train =  train.iloc[:,:7*max_periods]
    
    df = train_key[['Page']].copy()
    df['AllVisits'] = train.median(axis=1).fillna(0)
    test = test.merge(df, how='left', on='Page', copy=False)
    test.AllVisits = test.AllVisits.fillna(0).astype('float32')
    
    for site in sites:
        test[site] = (1 * (test.Site == site)).astype('float32')
    
    for access in accesses:
        test[access] = (1 * (test.AccessAgent == access)).astype('float32')

    for (w1, w2) in periods:
        
        df = train_key[['Page']].copy()
        c = 'median_%d_%d' % (w1, w2)
        df[c] = train.iloc[:,7*w1:7*w2].median(axis=1, skipna=True) 
        test = test.merge(df, how='left', on='Page', copy=False)
        test[c] = (test[c] - test.AllVisits).fillna(0).astype('float32')

    for c_norm, c in zip(y_norm_cols, y_cols):
        test[c_norm] = (np.log1p(test[c]) - test.AllVisits).astype('float32')

    gc.collect()

    return test

max_periods = 16
periods = [(0,1), (1,2), (2,3), (3,4), 
           (4,5), (5,6), (6,7), (7,8),
           ]


site_cols = list(sites)
access_cols = list(accesses)

test, test_all = test0.copy(), test_all0.copy()

for c in y_pred_cols:
    test[c] = np.NaN
    test_all[c] = np.NaN

test1 = add_median(test, train_norm, 
                   train_key, periods, max_periods, 3)

test_all1 = add_median(test_all, train_all_norm, 
                       train_key, periods, max_periods, 5)


# In[ ]:


num_cols = (['median_%d_%d' % (w1,w2) for (w1,w2) in periods])

import keras.backend as K

def smape_error(y_true, y_pred):
    return K.mean(K.clip(K.abs(y_pred - y_true),  0.0, 1.0), axis=-1)


def get_model(input_dim, num_sites, num_accesses, output_dim):
    
    dropout = 0.5
    regularizer = 0.00004
    main_input = Input(shape=(input_dim,), dtype='float32', name='main_input')
    site_input = Input(shape=(num_sites,), dtype='float32', name='site_input')
    access_input = Input(shape=(num_accesses,), dtype='float32', name='access_input')
    
    
    x0 = keras.layers.concatenate([main_input, site_input, access_input])
    x = Dense(200, activation='relu', 
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x0)
    x = Dropout(dropout)(x)
    x = keras.layers.concatenate([x0, x])
    x = Dense(200, activation='relu', 
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    x = BatchNormalization(beta_regularizer=regularizers.l2(regularizer),
                           gamma_regularizer=regularizers.l2(regularizer)
                          )(x)
    x = Dropout(dropout)(x)
    x = Dense(100, activation='relu', 
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    x = Dropout(dropout)(x)

    x = Dense(200, activation='relu', 
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    x = Dropout(dropout)(x)
    x = Dense(output_dim, activation='linear', 
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)

    model =  Model(inputs=[main_input, site_input, access_input], outputs=[x])
    model.compile(loss=smape_error, optimizer='adam')
    return model

group = pd.factorize(test1.Page)[0]

n_bag = 20
kf = GroupKFold(n_bag)
batch_size=4096

#print('week:', week)
test2 = test1
test_all2 = test_all1
X, Xs, Xa, y = test2[num_cols].values, test2[site_cols].values, test2[access_cols].values, test2[y_norm_cols].values
X_all, Xs_all, Xa_all, y_all = test_all2[num_cols].values, test_all2[site_cols].values, test_all2[access_cols].values, test_all2[y_norm_cols].fillna(0).values

y_true = test2[y_cols]
y_all_true = test_all2[y_cols]

models = [get_model(len(num_cols), len(site_cols), len(access_cols), len(y_cols)) for bag in range(n_bag)]

print('offset:', offset)
print('batch size:', batch_size)


best_score = 100
best_all_score = 100

save_pred = 0
saved_pred_all = 0

for n_epoch in range(10, 201, 10):
    print('************** start %d epochs **************************' % n_epoch)

    y_pred0 = np.zeros((y.shape[0], y.shape[1]))
    y_all_pred0 = np.zeros((n_bag, y_all.shape[0], y_all.shape[1]))
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y, group)):
        print('train fold', fold, end=' ')    
        model = models[fold]
        X_train, Xs_train, Xa_train, y_train = X[train_idx,:], Xs[train_idx,:], Xa[train_idx,:], y[train_idx,:]
        X_test, Xs_test, Xa_test, y_test = X[test_idx,:], Xs[test_idx,:], Xa[test_idx,:], y[test_idx,:]

        model.fit([ X_train, Xs_train, Xa_train],  y_train, 
                  epochs=10, batch_size=batch_size, verbose=0, shuffle=True, 
                  #validation_data=([X_test, Xs_test, Xa_test],  y_test)
                 )
        y_pred = model.predict([ X_test, Xs_test, Xa_test], batch_size=batch_size)
        y_all_pred = model.predict([X_all, Xs_all, Xa_all], batch_size=batch_size)

        y_pred0[test_idx,:] = y_pred
        y_all_pred0[fold,:,:]  = y_all_pred

        y_pred += test2.AllVisits.values[test_idx].reshape((-1,1))
        y_pred = np.expm1(y_pred)
        y_pred[y_pred < 0.5 * offset] = 0
        res = smape2D(test2[y_cols].values[test_idx, :], y_pred)
        y_pred = offset*((y_pred / offset).round())
        res_round = smape2D(test2[y_cols].values[test_idx, :], y_pred)

        y_all_pred += test_all2.AllVisits.values.reshape((-1,1))
        y_all_pred = np.expm1(y_all_pred)
        y_all_pred[y_all_pred < 0.5 * offset] = 0
        res_all = smape2D(test_all2[y_cols], y_all_pred)
        y_all_pred = offset*((y_all_pred / offset).round())
        res_all_round = smape2D(test_all2[y_cols], y_all_pred)
        print('smape train: %0.5f' % res, 'round: %0.5f' % res_round,
              '     smape LB: %0.5f' % res_all, 'round: %0.5f' % res_all_round)

    #y_pred0  = np.nanmedian(y_pred0, axis=0)
    y_all_pred0  = np.nanmedian(y_all_pred0, axis=0)

    y_pred0  += test2.AllVisits.values.reshape((-1,1))
    y_pred0 = np.expm1(y_pred0)
    y_pred0[y_pred0 < 0.5 * offset] = 0
    res = smape2D(y_true, y_pred0)
    print('smape train: %0.5f' % res, end=' ')
    y_pred0 = offset*((y_pred0 / offset).round())
    res_round = smape2D(y_true, y_pred0)
    print('round: %0.5f' % res_round)

    y_all_pred0 += test_all2.AllVisits.values.reshape((-1,1))
    y_all_pred0 = np.expm1(y_all_pred0)
    y_all_pred0[y_all_pred0 < 0.5 * offset] = 0
    #y_all_pred0 = y_all_pred0.round()
    res_all = smape2D(y_all_true, y_all_pred0)
    print('     smape LB: %0.5f' % res_all, end=' ')
    y_all_pred0 = offset*((y_all_pred0 / offset).round())
    res_all_round = smape2D(y_all_true, y_all_pred0)
    print('round: %0.5f' % res_all_round, end=' ')
    if res_round < best_score:
        print('saving')
        best_score = res_round
        best_all_score = res_all_round
        test.loc[:, y_pred_cols] = y_pred0
        test_all.loc[:, y_pred_cols] = y_all_pred0
    else:
        print()
    print('*************** end %d epochs **************************' % n_epoch)
print('best saved LB score:', best_all_score)


# # Conclusion:
# 
# The Three approaches all involved ensembling techniques, and i was fascinated as all three were quite different from each other.
# Recieved a score of 37.58649 which would place me in 9th position, which iam pretty happy about.
# 
# 
# I did try a ensemble technique but it took too long to run(2 days) there-fore stopped it.Above is the simple implementation of the Keras model of the 2nd place solution.His discussion was really helpful.
# 
# This Course was an informative journey, and was very pleased to be a part of.
