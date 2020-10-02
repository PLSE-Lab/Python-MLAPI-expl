from __future__ import division
import sqlite3, time, csv, re, random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix

'''
This script was inspired by smerity's script "The Biannual Reddit Sarcasm
Hunt." A natural follow-up question is whether we can detect posts with the
/s flag using a BOW model. 

The corpus has about 54m posts, of which about 30k have the /s flag. It is 
impossible to compete with a majority baseline that strong, so instead I've
framed it as a binary classification task with uniform class distribution.
Realistic, no, but enough to see some pronounced trends in the features. A
logistic regression model scores about 72% on unseen data.

Lots of work has been done on irony detection, here are a couple references:
Bamman, Contextualized Sarcasm Detection on Twitter, ICWSM 2015
Wallace, Humans Require Context to Infer Ironic Intent (so Computers 
Probably do, too), ACL 2014
'''

#Parameters
srs_lmt = 30100 #serious posts to train on
sar_lmt = 30100 #sarcastic posts to train on
top_k = 30 #features to display
num_ex = 20 #examples displayed per feature
min_ex = 0 #shortest example displayed
max_ex = 120 #longest example displayed
ovr_ex = True #display longer/shorter examples if we run out

print('Querying DB...\n')
sql_conn = sqlite3.connect('../input/database.sqlite')

sarcasmData = """SELECT subreddit, body, score FROM May2015\
                                WHERE body LIKE '% /s'\
                                LIMIT 30100"""

seriousData = """SELECT subreddit, body, score FROM May2015\
                                WHERE body NOT LIKE '%/s%'\
                                LIMIT 30100"""

sarcasmdata=pd.read_sql(sarcasmData,sql_conn)
serioudata=pd.read_sql(seriousData,sql_conn)
sarcasmdata.to_csv('sarcasmredditfull.csv', encoding='utf-8', index=False)
serioudata.to_csv('serioudataredditfull.csv', encoding='utf-8', index=False)
print('Building Corpora...\n')
#corpus, raw_corpus, srs_corpus = [], [], []
"""
for sar_post in sarcasmData:
    raw_corpus.append(re.sub('\n', '', sar_post[1]))
    cln_post = re.sub('/s|\n', '', sar_post[1]) #Remove /s and newlines
    corpus.append(re.sub(r'([^\s\w]|_)+', '', cln_post)) #and then non-alpha

for srs_post in seriousData:
    srs_corpus.append(re.sub('\n', '', srs_post[1]))
    cln_post = re.sub('\n', '', srs_post[1]) #Remove newlines
    corpus.append(re.sub(r'([^\s\w]|_)+', '', cln_post)) #and then non-alpha
"""

    
    
    
