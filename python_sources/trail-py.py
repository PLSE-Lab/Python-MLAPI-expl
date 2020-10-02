# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import log_loss
import time; start_time = time.time()
import xgboost as xgb
import pandas as pd
import numpy as np
import re

#Other Options
import nltk
from collections import Counter
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
import difflib
from nltk.metrics.distance import jaccard_distance, edit_distance

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv', low_memory=False, iterator=True, chunksize=600000)

pos_train = train[train['is_duplicate'] == 1]
neg_train = train[train['is_duplicate'] == 0]
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
train = pd.concat([pos_train, neg_train])

def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.ratio()

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

def quick_clean(s): 
    if isinstance(s, str):
        s = s.lower()
        s = re.sub(r"[^a-z0-9 ]", "_", s)
        s = s.replace("  "," ").strip()
        s = (" ").join([z for z in s.split(" ") if len(z)>2])
        #s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        return s
    else:
        return "null"

class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, df):
        df['zexactly_same'] = (df['question1'] == df['question2']).astype(int)
        df['zduplicated'] = df.duplicated(['question1','question2']).astype(int)
        #df['question1'] = df.question1.map(lambda x: quick_clean(str(x)))
        #df['question2'] = df.question2.map(lambda x: quick_clean(str(x)))
        #print('lengths...')
        df['z_len1'] = df.question1.map(lambda x: len(str(x)))
        df['z_len2'] = df.question2.map(lambda x: len(str(x)))
        df['z_len_diff'] = df['z_len1'] - df['z_len2']
        df['z_word_len1'] = df.question1.map(lambda x: len(str(x).split()))
        df['z_word_len2'] = df.question2.map(lambda x: len(str(x).split()))
        df['z_word_len_diff'] =  df['z_word_len1'] -  df['z_word_len2']
        df['zavg_world_len1'] = df['z_len1'] / df['z_word_len1']
        df['zavg_world_len2'] = df['z_len2'] / df['z_word_len2']
        df['zavg_world_len_diff'] = df['zavg_world_len1'] - df['zavg_world_len2']
        #print('word match...')
        df['z_word_match'] = df.apply(word_match_share, axis=1, raw=True)
        #print('nouns...')
        #df['question1_nouns'] = df.question1.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
        #df['question2_nouns'] = df.question2.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
        #df['z_noun_match'] = df.apply(lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)  #takes long
        #print('difflib...')
        #df['z_match_ratio'] = df.apply(lambda r: diff_ratios(r.question1, r.question2), axis=1)  #takes long
        #print('distances...')
        #df['z_edit_dist'] = df.apply(lambda r: edit_distance(str(r.question1).split(' '), str(r.question2).split(' ')),axis=1)
        #df['z_jaccard_dist'] = df.apply(lambda r: jaccard_distance(set(str(r.question1).split(' ')), set(str(r.question2).split(' '))),axis=1)
        
        df = df.fillna(0.0)
        col = [c for c in df.columns if c[:1]=='z']
        return df[col]

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, df):
        return df[self.key].apply(str)

tfidf = TfidfVectorizer(max_df=1.0, min_df=1, ngram_range=(1, 1), max_features=100)
reg = Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst', Pipeline([('cst1', cust_regression_vals())])),
                        ('txt1', Pipeline([('s1', cust_txt_col(key='question1')), ('tfidf1', tfidf)])),
                        ('txt2', Pipeline([('s2', cust_txt_col(key='question2')), ('tfidf2', tfidf)]))
                        ],
                    transformer_weights = {
                        'cst': 1.0,
                        'txt1': 1.0,
                        'txt2': 1.0
                        },
                n_jobs = -1
                ))])

df = reg.fit_transform(train[['question1','question2']])
x_train, x_valid, y_train, y_valid = train_test_split(df, train['is_duplicate'], test_size=0.2, random_state=0)

params = {}
params["objective"] = "binary:logistic"
params['eval_metric'] = 'logloss'
params["eta"] = 0.02
params["subsample"] = 0.7
params["min_child_weight"] = 1
params["colsample_bytree"] = 0.7
params["max_depth"] = 4
params["silent"] = 1
params["seed"] = 12357

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=50, verbose_eval=100)
print(log_loss(train.is_duplicate, bst.predict(xgb.DMatrix(df))))
df, train = [None, None]
x_train, x_valid, y_train, y_valid, d_train, d_valid = [None, None, None, None, None, None]

id_test = []
y_pred = []
for df_test in test:
    id_test += list(df_test['test_id'].values)
    df_test = reg.transform(df_test[['question1','question2']])
    y_pred += list(bst.predict(xgb.DMatrix(df_test)))
    print('test...', round(((time.time() - start_time)/60),2))
test, df_test = [None, None]

sub = pd.DataFrame({"test_id": id_test, "is_duplicate": y_pred})
sub.to_csv('submission.csv', index=False)
print('Done...', round(((time.time() - start_time)/60),2))

