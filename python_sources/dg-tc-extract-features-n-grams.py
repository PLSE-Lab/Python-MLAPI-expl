# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#DG_TC_extract_features_n_grams.py

import gc
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def read_csv(fname, chunksize=10000):
    chunks = pd.read_csv(fname, chunksize=chunksize)
    df_chunks = []
    for idx, chunk in enumerate(chunks):
        df_chunks.append(chunk)
    return pd.concat(df_chunks).reset_index(drop=True)

print('Loading dataset...')
train = read_csv('../input/new_data/train_set.csv')
test = read_csv('../input/new_data/test_set.csv')
y_train = (train["class"]-1).astype(int)

X_train = train['word_seg']
X_test = test['word_seg']
X = pd.concat((X_train, X_test))

print('Extracting word n-grams features...')

print('Fit word_seg data...')
word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
word_vectorizer.fit(X)

print('Transform train data...')
train_features_word_n_grams = word_vectorizer.transform(X_train)
train_features_word = {'features': train_features_word_n_grams, 'labels': y_train}
with open('train_features_word_n_grams.pickle', 'wb', protocol=2) as f:
    pickle.dump(train_features_word, f)

del train_features_word_n_grams, test_features_word
gc.collect()

print('Transform test data...')
test_features_word_n_grams = word_vectorizer.transform(X_test)
test_features_word = {'features': test_features_word_n_grams}
with open('test_features_word_n_grams.pickle', 'wb', protocol=2) as f:
    pickle.dump(test_features_word, f)


del  test_features_word, test_features_word_n_grams
gc.collect()

print('Extracting char n-grams features...')

X_train = train['article']
X_test = test['article']
X = pd.concat((X_train, X_test), axis=0)

print('Fit article data...')
char_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
char_vectorizer.fit(X)

print('Transform train data...')
train_features_char_n_grams = char_vectorizer.transform(X_train)
train_features_char = {'features': train_features_char_n_grams, 'labels': y_train}
with open('train_features_char_n_grams.pickle', 'wb', protocol=2) as f:
    pickle.dump(train_features_char, f)

del train_features_char_n_grams, train_features_char
gc.collect()

print('Transform test data...')
test_features_char_n_grams = char_vectorizer.transform(X_test)
test_features_char = {'features': test_features_char_n_grams, 'labels': y}
with open('test_features_char_n_grams.pickle', 'wb', protocol=2) as f:
    pickle.dump(test_features_char, f)

del test_features_char_n_grams, test_features_char
gc.collect()