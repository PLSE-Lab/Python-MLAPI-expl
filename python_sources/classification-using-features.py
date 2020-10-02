#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
import gc
import numpy as np
import os
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import skew, kurtosis
import lightgbm as lgb

import Levenshtein
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from tqdm import tqdm


# In[ ]:


id_col = 'Id'
target_col = 'target'

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


def extract_features(df):
    df['nunique'] = df['ciphertext'].apply(lambda x: len(np.unique(x)))
    df['len'] = df['ciphertext'].apply(lambda x: len(x))

    def count_chars(x):
        n_l = 0 # count letters
        n_n = 0 # count numbers
        n_s = 0 # count symbols
        n_ul = 0 # count upper letters
        n_ll = 0 # count lower letters
        for i in range(0, len(x)):
            if x[i].isalpha():
                n_l += 1
                if x[i].isupper():
                    n_ul += 1
                elif x[i].islower():
                    n_ll += 1
            elif x[i].isdigit():
                n_n += 1
            else:
                n_s += 1

        return pd.Series([n_l, n_n, n_s, n_ul, n_ll])

    cols = ['n_l', 'n_n', 'n_s', 'n_ul', 'n_ll']
    for c in cols:
        df[c] = 0
    tqdm.pandas(desc='count_chars')
    df[cols] = df['ciphertext'].progress_apply(lambda x: count_chars(x))
    for c in cols:
        df[c] /= df['len']

    tqdm.pandas(desc='distances')
    df['Levenshtein_distance'] = df['ciphertext'].progress_apply(lambda x: Levenshtein.distance(x, x[::-1]))
    df['Levenshtein_ratio'] = df['ciphertext'].progress_apply(lambda x: Levenshtein.ratio(x, x[::-1]))
    df['Levenshtein_jaro'] = df['ciphertext'].progress_apply(lambda x: Levenshtein.jaro(x, x[::-1]))
    df['Levenshtein_hamming'] = df['ciphertext'].progress_apply(lambda x: Levenshtein.hamming(x, x[::-1]))

    for m in range(1, 5):
        df['Levenshtein_distance_m{}'.format(m)] = df['ciphertext'].progress_apply(lambda x: Levenshtein.distance(x[:-m], x[m:]))
        df['Levenshtein_ratio_m{}'.format(m)] = df['ciphertext'].progress_apply(lambda x: Levenshtein.ratio(x[:-m], x[m:]))
        df['Levenshtein_jaro_m{}'.format(m)] = df['ciphertext'].progress_apply(lambda x: Levenshtein.jaro(x[:-m], x[m:]))
        df['Levenshtein_hamming_m{}'.format(m)] = df['ciphertext'].progress_apply(lambda x: Levenshtein.hamming(x[:-m], x[m:]))
    
    df['Levenshtein_distance_h'] = df['ciphertext'].progress_apply(lambda x: Levenshtein.distance(x[:len(x)//2], x[len(x)//2:]))
    df['Levenshtein_ratio_h'] = df['ciphertext'].progress_apply(lambda x: Levenshtein.ratio(x[:len(x)//2], x[len(x)//2:]))
    df['Levenshtein_jaro_h'] = df['ciphertext'].progress_apply(lambda x: Levenshtein.jaro(x[:len(x)//2], x[len(x)//2:]))
    
    # All symbols stats
    def strstat(x):
        r = np.array([ord(c) for c in x])
        return pd.Series([
            np.sum(r), 
            np.mean(r), 
            np.std(r), 
            np.min(r), 
            np.max(r),
            skew(r), 
            kurtosis(r),
            ])
    cols = ['str_sum', 'str_mean', 'str_std', 'str_min', 'str_max', 'str_skew', 'str_kurtosis']
    for c in cols:
        df[c] = 0
    tqdm.pandas(desc='strstat')
    df[cols] = df['ciphertext'].progress_apply(lambda x: strstat(x))
    
    # Digit stats
    def str_digit_stat(x):
        r = np.array([ord(c) for c in x if c.isdigit()])
        if len(r) == 0:
            r = np.array([0])
        return pd.Series([
            np.sum(r), 
            np.mean(r), 
            np.std(r), 
            np.min(r), 
            np.max(r),
            skew(r), 
            kurtosis(r),
            ])
    cols = ['str_digit_sum', 'str_digit_mean', 'str_digit_std', 'str_digit_min', 
        'str_digit_max', 'str_digit_skew', 'str_digit_kurtosis']
    for c in cols:
        df[c] = 0
    tqdm.pandas(desc='str_digit_stat')
    df[cols] = df['ciphertext'].progress_apply(lambda x: str_digit_stat(x))


# In[ ]:


print('Extracting features for train:')
extract_features(train)
print('Extracting features for test:')
extract_features(test)


# In[ ]:


train.head()


# In[ ]:




