#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import multiprocessing as mp
import pandas as pd
import re
import time
from scipy.sparse import csr_matrix
import os
# from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import gc
from sklearn.base import BaseEstimator, TransformerMixin
import re
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from sklearn.model_selection import train_test_split
from datetime import datetime 
start_real = datetime.now()
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
# from keras.layers import Bidirectional
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from nltk.corpus import stopwords
import math
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix
from sklearn.naive_bayes import BernoulliNB
import nltk
from nltk.stem import SnowballStemmer
# set seed
np.random.seed(123)


# In[ ]:


os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'forkserver'

INPUT_PATH = r'../input'


def dameraulevenshtein(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.

    This method has not been modified from the original.
    Source: http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/

    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.

    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.

    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.

    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2

    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


class SymSpell:
    def __init__(self, max_edit_distance=3, verbose=0):
        self.max_edit_distance = max_edit_distance
        self.verbose = verbose
        # 0: top suggestion
        # 1: all suggestions of smallest edit distance
        # 2: all suggestions <= max_edit_distance (slower, no early termination)

        self.dictionary = {}
        self.longest_word_length = 0

    def get_deletes_list(self, w):
        """given a word, derive strings with up to max_edit_distance characters
           deleted"""

        deletes = []
        queue = [w]
        for d in range(self.max_edit_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c + 1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue

        return deletes

    def create_dictionary_entry(self, w):
        '''add word and its derived deletions to dictionary'''
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections,
        # frequency of word in corpus)
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0], self.dictionary[w][1] + 1)
        else:
            self.dictionary[w] = ([], 1)
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1] == 1:
            # first appearance of word in corpus
            # n.b. word may already be in dictionary as a derived word
            # (deleting character from a real word)
            # but counter of frequency of word in corpus is not incremented
            # in those cases)
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list
                    self.dictionary[item][0].append(w)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = ([w], 0)

        return new_real_word_added

    def create_dictionary_from_arr(self, arr, token_pattern=r'[a-z]+'):
        total_word_count = 0
        unique_word_count = 0

        for line in arr:
            # separate by words by non-alphabetical characters
            words = re.findall(token_pattern, line.lower())
            for word in words:
                total_word_count += 1
                if self.create_dictionary_entry(word):
                    unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def create_dictionary(self, fname):
        total_word_count = 0
        unique_word_count = 0

        with open(fname) as file:
            for line in file:
                # separate by words by non-alphabetical characters
                words = re.findall('[a-z]+', line.lower())
                for word in words:
                    total_word_count += 1
                    if self.create_dictionary_entry(word):
                        unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def get_suggestions(self, string, silent=False):
        """return list of suggested corrections for potentially incorrectly
           spelled word"""
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print("no items in dictionary within maximum edit distance")
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        q_dictionary = {}  # items other than string that we've checked

        while len(queue) > 0:
            q_item = queue[0]  # pop
            queue = queue[1:]

            # early exit
            if ((self.verbose < 2) and (len(suggest_dict) > 0) and
                    ((len(string) - len(q_item)) > min_suggest_len)):
                break

            # process queue item
            if (q_item in self.dictionary) and (q_item not in suggest_dict):
                if self.dictionary[q_item][1] > 0:
                    # word is in dictionary, and is a word from the corpus, and
                    # not already in suggestion list so add to suggestion
                    # dictionary, indexed by the word with value (frequency in
                    # corpus, edit distance)
                    # note q_items that are not the input string are shorter
                    # than input string since only deletes are added (unless
                    # manual dictionary corrections are added)
                    assert len(string) >= len(q_item)
                    suggest_dict[q_item] = (self.dictionary[q_item][1],
                                            len(string) - len(q_item))
                    # early exit
                    if (self.verbose < 2) and (len(string) == len(q_item)):
                        break
                    elif (len(string) - len(q_item)) < min_suggest_len:
                        min_suggest_len = len(string) - len(q_item)

                # the suggested corrections for q_item as stored in
                # dictionary (whether or not q_item itself is a valid word
                # or merely a delete) can be valid corrections
                for sc_item in self.dictionary[q_item][0]:
                    if sc_item not in suggest_dict:

                        # compute edit distance
                        # suggested items should always be longer
                        # (unless manual corrections are added)
                        assert len(sc_item) > len(q_item)

                        # q_items that are not input should be shorter
                        # than original string
                        # (unless manual corrections added)
                        assert len(q_item) <= len(string)

                        if len(q_item) == len(string):
                            assert q_item == string
                            item_dist = len(sc_item) - len(q_item)

                        # item in suggestions list should not be the same as
                        # the string itself
                        assert sc_item != string

                        # calculate edit distance using, for example,
                        # Damerau-Levenshtein distance
                        item_dist = dameraulevenshtein(sc_item, string)

                        # do not add words with greater edit distance if
                        # verbose setting not on
                        if (self.verbose < 2) and (item_dist > min_suggest_len):
                            pass
                        elif item_dist <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (self.dictionary[sc_item][1], item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                        # depending on order words are processed, some words
                        # with different edit distances may be entered into
                        # suggestions; trim suggestion dictionary if verbose
                        # setting not on
                        if self.verbose < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            # now generate deletes (e.g. a substring of string or of a delete)
            # from the queue item
            # as additional items to check -- add to end of queue
            assert len(string) >= len(q_item)

            # do not add words with greater edit distance if verbose setting
            # is not on
            if (self.verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

        # queue is now empty: convert suggestions in dictionary to
        # list for output
        if not silent and self.verbose != 0:
            print("number of possible corrections: %i" % len(suggest_dict))
            print("  edit distance for deletions: %i" % self.max_edit_distance)

        # output option 1
        # sort results by ascending order of edit distance and descending
        # order of frequency
        #     and return list of suggested word corrections only:
        # return sorted(suggest_dict, key = lambda x:
        #               (suggest_dict[x][1], -suggest_dict[x][0]))

        # output option 2
        # return list of suggestions with (correction,
        #                                  (frequency in corpus, edit distance)):
        as_list = suggest_dict.items()
        # outlist = sorted(as_list, key=lambda (term, (freq, dist)): (dist, -freq))
        outlist = sorted(as_list, key=lambda x: (x[1][1], -x[1][0]))

        if self.verbose == 0:
            return outlist[0]
        else:
            return outlist

        '''
        Option 1:
        ['file', 'five', 'fire', 'fine', ...]

        Option 2:
        [('file', (5, 0)),
         ('five', (67, 1)),
         ('fire', (54, 1)),
         ('fine', (17, 1))...]  
        '''

    def best_word(self, s, silent=False):
        try:
            return self.get_suggestions(s, silent)[0]
        except:
            return None


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field, start_time=time.time()):
        self.field = field
        self.start_time = start_time

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        print(f'[{time.time()-self.start_time}] select {self.field}')
        dt = dataframe[self.field].dtype
        if is_categorical_dtype(dt):
            return dataframe[self.field].cat.codes[:, None]
        elif is_numeric_dtype(dt):
            return dataframe[self.field][:, None]
        else:
            return dataframe[self.field]


class DropColumnsByDf(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=1, max_df=1.0):
        self.min_df = min_df
        self.max_df = max_df

    def fit(self, X, y=None):
        m = X.tocsc()
        self.nnz_cols = ((m != 0).sum(axis=0) >= self.min_df).A1
        if self.max_df < 1.0:
            max_df = m.shape[0] * self.max_df
            self.nnz_cols = self.nnz_cols & ((m != 0).sum(axis=0) <= max_df).A1
        return self

    def transform(self, X, y=None):
        m = X.tocsc()
        return m[:, self.nnz_cols]


def get_rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))


def split_sub(text):
#     text=re.sub(r'(.)\1+', r'\1\1', text) #removes repeating characters
    return re.sub('[^0-9\'a-zA-Z]+', ' ', text) 


def preprocess_pandas(train, test, start_time=time.time()):
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    print('Train shape : ', train.shape)

    nrow_train = train.shape[0]
    y_train = train[list_classes]
    merge: pd.DataFrame = pd.concat([train, test])

    del train
    del test
    gc.collect()

    merge['comment_text'] = merge['comment_text']         .fillna('')         .str.lower()         .apply(lambda x: split_sub(x))
    print(f'[{time.time() - start_time}] Missing filled.')

    return merge, y_train, nrow_train


def intersect_drop_columns(train: csr_matrix, valid: csr_matrix, min_df=0):
    t = train.tocsc()
    v = valid.tocsc()
    nnz_train = ((t != 0).sum(axis=0) >= min_df).A1
    nnz_valid = ((v != 0).sum(axis=0) >= min_df).A1
    nnz_cols = nnz_train & nnz_valid
    res = t[:, nnz_cols], v[:, nnz_cols]
    return res

def wordCount(text):
    try:
        if text == 'No description yet':
            return 0
        else:
            text = text.lower()
            words = [w for w in text.split(" ")]
            return len(words)
    except: 
        return 0


mp.set_start_method('forkserver', True)

start_time = time.time()
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv') # uncomment for final versioin
#     train, test = train_test_split(train, shuffle=False) #comment for dev
#     y_test =  test[list_classes]#comment for dev
print(f'[{time.time() - start_time}] Finished to load data')
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

train['char_len'] = train['comment_text'].apply(lambda x: len(x))
test['char_len'] = test['comment_text'].apply(lambda x: len(x))
train['desc_len'] = train['comment_text'].apply(lambda x: wordCount(x))
test['desc_len'] = test['comment_text'].apply(lambda x: wordCount(x))

# stuff from the NN Guy
# https://www.kaggle.com/prashantkikani/pooled-gru-with-preprocessing
repl = {
    "&lt;3": " good ",
    ":d": " good ",
    ":dd": " good ",
    ":p": " good ",
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":s": " bad ",
    ":-s": " bad ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}

keys = [i for i in repl.keys()]

new_train_data = []
new_test_data = []
ltr = train["comment_text"].tolist()
lte = test["comment_text"].tolist()
for i in ltr:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_train_data.append(xx)
for i in lte:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_test_data.append(xx)
train["new_comment_text"] = new_train_data
test["new_comment_text"] = new_test_data
print("crap removed")
trate = train["new_comment_text"].tolist()
tete = test["new_comment_text"].tolist()
for i, c in enumerate(trate):
    trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
for i, c in enumerate(tete):
    tete[i] = re.sub('[^a-zA-Z ?!]+', '', tete[i])
train["comment_text"] = trate
test["comment_text"] = tete

stem = SnowballStemmer('english')
def token_stemmer(text):
    tokens = nltk.word_tokenize(text)
    tokens = [stem.stem(item) for item in tokens]
    return ' '.join(tokens)
train['comment_text'] = train['comment_text'].apply(lambda x: token_stemmer(x))
test['comment_text'] = test['comment_text'].apply(lambda x: token_stemmer(x))
print(f'[{time.time() - start_time}] Stemming Completed.')


abuses = pd.read_csv('../input/abusive-words5/abusive_words.csv',  header=None)
def has_abuse(text):
    return sum((abuses[0]).apply(lambda x: str(x) in text))
def abuse_list(text):
    return ' '.join(abuses[0].apply(lambda x: str(x) if str(x) in text else ''))
train['abuse_list'] = train['new_comment_text'].apply(lambda x: abuse_list(x).split())
test['abuse_list'] = test['new_comment_text'].apply(lambda x: abuse_list(x).split())
print(f'[{time.time() - start_time}] Abuses List Completed.')

abuses = pd.read_csv('../input/abusive-words3/abusive_words.csv',  header=None)
train['num_abuses'] = train['new_comment_text'].apply(lambda x: has_abuse(x))
test['num_abuses'] = test['new_comment_text'].apply(lambda x: has_abuse(x))
print(f'[{time.time() - start_time}] Abuses Completed.')


merge, y_train, nrow_train = preprocess_pandas(train, test, start_time)

meta_params = {
               'desc_ngram': (1, 3),
               'desc_max_f': 150000,
               'desc_max_df': 0.5,
               'desc_min_df': 10}

stopwords = frozenset(['the', 'a', 'an', 'is', 'it', 'this', ])

# 'i', 'so', 'its', 'am', 'are'])

from sklearn.base import TransformerMixin #gives fit_transform method for free
class MyLabelEncoder(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelEncoder(*args, **kwargs)
    def fit(self, x, y = None):
        self.encoder.fit(x)
        return self
    def transform(self, x, y = None):
        return self.encoder.transform(x)

vectorizer = FeatureUnion([

#         ('comment_text', Pipeline([
#             ('select', ItemSelector('comment_text', start_time=start_time)),
#             ('transform', HashingVectorizer(
#                 ngram_range=(1, 1),
#                 n_features=2 ** 27,
#                 norm='l2',
#                 lowercase=False,
#                 stop_words ='english',
#             )),
#             ('drop_cols', DropColumnsByDf(min_df=2))
#         ])),
    ('comment_text_2', Pipeline([
            ('select', ItemSelector('comment_text', start_time=start_time)),
            ('transform',   TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='word',
                stop_words='english',
                token_pattern=r'\w{1,}',
                ngram_range=(1, 1),
                max_features=15000
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
    ('comment_text_3', Pipeline([
            ('select', ItemSelector('comment_text', start_time=start_time)),
            ('transform', TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='char',
                stop_words='english',
                ngram_range=(2, 6),
                max_features=20000
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
#         ('desc_len', Pipeline([
#             ('select', ItemSelector('desc_len', start_time=start_time)),
#             ('transform', CountVectorizer(
#                 token_pattern='.+',
#                 min_df=2,
#                 lowercase=False
#             )),
#         ])),
#         ('num_abuses', Pipeline([
#             ('select', ItemSelector('num_abuses', start_time=start_time)),
#             ('transform', CountVectorizer(
#                 token_pattern='.+',
#                 min_df=2,
#                 lowercase=False
#             )),
#         ]))
#         ('desc_len', Pipeline([
#             ('select', ItemSelector('desc_len', start_time=start_time)),
#             ('ohe', OneHotEncoder())
#         ])),
        ('abuse_list', Pipeline([
            ('select', ItemSelector('abuse_list', start_time=start_time)),
            ('ohe', MyLabelEncoder())
        ])),
        ('num_abuses', Pipeline([
            ('select', ItemSelector('num_abuses', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ]))
    ], n_jobs=1)


sparse_merge = vectorizer.fit_transform(merge)
print(f'[{time.time() - start_time}] Merge vectorized')
print(sparse_merge.shape)
tfidf_transformer = TfidfTransformer()

X = tfidf_transformer.fit_transform(sparse_merge)
print(f'[{time.time() - start_time}] TF/IDF completed')

X_train = X[:nrow_train]
print(X_train.shape)

X_test = X[nrow_train:]
del merge
del sparse_merge
del vectorizer
del tfidf_transformer
gc.collect()

X_train, X_test = intersect_drop_columns(X_train, X_test, min_df=1)
print(f'[{time.time() - start_time}] Drop only in train or test cols: {X_train.shape[1]}')
gc.collect()


###########
losses = []
predictions = {'id': test['id']}
from sklearn.svm import LinearSVC
for class_name in list_classes:
    train_target = train[class_name]
#     classifier = RidgeClassifier(solver='auto', fit_intercept=True, alpha=0.4, max_iter=200, normalize=False, tol=0.01)
    classifier = LogisticRegression(solver='sag')
#     classifier = LinearSVC()
#     classifier = MultinomialNB()
#     classifier = RidgeClassifier(tol=1e-2, solver="sag")
#     classifier =  PassiveAggressiveClassifier()
    cv_loss = np.mean(cross_val_score(classifier, X_train, train_target, cv=3, scoring='roc_auc'))
    losses.append(cv_loss)
    print('CV score for class {} is {}'.format(class_name, cv_loss))
#     cv_lossb = np.mean(cross_val_score(classifierb, X_train, train_target, cv=3, scoring='roc_auc'))
#     lossesb.append(cv_lossb)
#     print('CV score for class {} is {}'.format(class_name, cv_loss))
#     print('CV score for class {} is {}'.format(class_name, cv_lossb))


    classifier.fit(X_train, train_target)
#     classifierb.fit(X_train, train_target)
    predictions[class_name] = classifier.predict_proba(X_test)[:, 1]
#     predictions[class_name] = (classifier.predict_proba(X_test)[:, 1] + classifierb.predict_proba(X_test)[:, 1])/2
#     d = classifier.decision_function(X_test)
#     predictions[class_name] =  np.exp(d) / (1 + np.exp(d))

print(f'[{time.time() - start_time}] Predict Ridge completed.')

###########

submission = pd.DataFrame.from_dict(predictions)
submission.to_csv("submission_ridge.csv", index=False)

#uncomment if not developing
print("Evaluating the model on validation data...")
# print(" ROC_AUC Score:", roc_auc_score(y_test, submission[list_classes]))
#0.9737 error for ridge auto
# log regression 0.97482 sag


# In[ ]:





# In[ ]:


merge.head(100)


# In[ ]:





# In[ ]:




