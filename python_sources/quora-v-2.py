# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 12:05:04 2017
https://www.kaggle.com/heraldxchaos/quora-question-pairs/adventures-in-scikitlearn-and-nltk/run/1039110
@author: User
"""
import numpy as np 
import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
import scipy
import xgboost as xgb
import multiprocessing
import difflib
from nltk.corpus import stopwords
from nltk.metrics import jaccard_distance

#Reading and processing of data
train=pd.read_csv('../input/train.csv').fillna("")
stops = set(stopwords.words("english"))

def build_dict(sentences):
    #Dictionary of train words --> word index: word freq
    print('Building dictionary using train words..')
    wordcount = dict()
    #For each worn in each sentence, cummulate frequency
    for ss in sentences:
        for w in ss:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1    
    worddict = dict()
    for idx, w in enumerate(sorted(wordcount.items(), key = lambda x: x[1], reverse=True)):
        worddict[w[0]] = idx+2  # leave 0 and 1 (UNK)
    return worddict, wordcount

def generate_sequence(sentences, dictionary):
    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in ss]
    return seqs

def normalize(x):
    return x.lower().split()

questions = train['question1'].tolist() + train['question2'].tolist()
tok_questions = [normalize(s) for s in questions]
'''
>>> questions[1]
'What is the story of Kohinoor (Koh-i-Noor) Diamond?'
>>> tok_questions[1]
['what', 'is', 'the', 'story', 'of', 'kohinoor', '(koh-i-noor)', 'diamond?']
'''

worddict, wordcount = build_dict(tok_questions)
print(np.sum(list(wordcount.values())), ' total words ', len(worddict), ' unique words')
'''8944591  total words  201102  unique words'''

def jc(x):
    return jaccard_distance(set(x['s_question1']),set(x['s_question2']))

def cosine_d(x):
    a = set(x['s_question1'])
    b = set(x['s_question2'])
    d = len(a)*len(b)
    if (d == 0):
        return 0
    else: 
        return len(a.intersection(b))/d

tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
tfidf.fit_transform(questions)

def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.quick_ratio()

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
def get_features(df_features):
    
    print('jaccard...')
    df_features['s_question1'] = generate_sequence(df_features['question1'].apply(normalize),worddict)
    '''
    >>> train['s_question1'][1]
    [3, 4, 2, 749, 10, 18691, 40608, 13451]
    '''
    df_features['s_question2'] = generate_sequence(df_features['question2'].apply(normalize),worddict)
    df_features['z_jaccard'] = df_features.apply(jc,axis = 1)
    
    print('cosine....')
    df_features['z_cosine'] = df_features.apply(cosine_d,axis = 1)
    
    print('question lengths....')
    #Question length
    df_features['z_len1'] = df_features.question1.map(lambda x: len(str(x)))
    df_features['z_len2'] = df_features.question2.map(lambda x: len(str(x)))
    #Question number of words
    df_features['z_word_len1'] = df_features.question1.map(lambda x: len(str(x).split()))
    df_features['z_word_len2'] = df_features.question2.map(lambda x: len(str(x).split())) 
    
    print('difflib...')
    #matching the sequences
    df_features['z_match_ratio'] = df_features.apply(lambda r: diff_ratios(r.question1, r.question2), axis=1)  #takes long
    
    print('word match...')
    #percentage of common words in both questions
    df_features['z_word_match'] = df_features.apply(word_match_share, axis=1, raw=True)
    
    '''
    print('tfidf...')
    question1_tfidf = tfidf.transform(df_features.question1.tolist())
    print(type(question1_tfidf))
    question2_tfidf = tfidf.transform(df_features.question2.tolist())
    df_features['z_tfidf_sum1'] = scipy.sparse.csr_matrix(question1_tfidf).sum(axis=1)
    df_features['z_tfidf_sum2'] = scipy.sparse.csr_matrix(question2_tfidf).sum(axis=1)
    df_features['z_tfidf_mean1'] = scipy.sparse.csr_matrix(question1_tfidf).mean(axis=1)
    df_features['z_tfidf_mean2'] = scipy.sparse.csr_matrix(question2_tfidf).mean(axis=1)
    df_features['z_tfidf_len1'] = (question1_tfidf != 0).sum(axis = 1)
    df_features['z_tfidf_len2'] = (question2_tfidf != 0).sum(axis = 1)
    
    '''
    return df_features.fillna(0.0)
df_train = get_features(train)

col = [c for c in df_train.columns if c[:1]=='z']
x_train, x_valid, y_train, y_valid = train_test_split(df_train[col], df_train['is_duplicate'], test_size=0.2, random_state=0)
#XGBoost mosdel
params = {}
params["objective"] = "binary:logistic"
params['eval_metric'] = 'logloss'
params["eta"] = 0.02
params["subsample"] = 0.7
params["min_child_weight"] = 1
params["colsample_bytree"] = 0.7
params["max_depth"] = 4
params["silent"] = 1
params["seed"] = 1632

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=50,verbose_eval=100) #change to higher #s
print('training done')

print(log_loss(df_train.is_duplicate, bst.predict(xgb.DMatrix(df_train[col]))))

#Predicting for test data set
sub = pd.DataFrame() # Submission data frame
sub['test_id'] = []
sub['is_duplicate'] = []
header=['test_id','question1','question2']
test=pd.read_csv('../input/test.csv').fillna("")
df_test = get_features(test)
sub=pd.DataFrame({'test_id':df_test['test_id'], 'is_duplicate':bst.predict(xgb.DMatrix(df_test[col]))})
sub.to_csv('quora_submission_xgb_01.csv', index=False)
