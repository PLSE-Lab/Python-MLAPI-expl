
from __future__ import division
import sqlite3, time, csv, re, random
import pandas as pd
import operator
from csv import reader

#Parameters
srs_lmt = 10000 #serious posts to train on
sar_lmt = 10000 #sarcastic posts to train on

print('Querying DB...\n')
sql_conn = sqlite3.connect('../input/database.sqlite')

'''
sarcasmData = sql_conn.execute("SELECT subreddit, body, score FROM May2015\
                                WHERE body LIKE '% /s'\
                                LIMIT " + str(sar_lmt))

'''
seriousData = sql_conn.execute("SELECT * FROM May2015 where subreddit_id='t5_2rjz2' or subreddit_id='t5_2r2o9'")
print('Building Corpora...\n')
corpus, sar_corpus, srs_corpus = [], [], []

'''
for sar_post in sarcasmData:
    #cln_post = re.sub('/s|\n', '', sar_post[1]) #Remove /s and newlines
    #cln_post = re.sub(r'([^\s\w]|_)+', '', cln_post)
    #sar_corpus.append(re.sub('\n', '', cln_post))
    sar_corpus.append(sar_post)
    #corpus.append(re.sub(r'([^\s\w]|_)+', '', cln_post)) #and then non-alpha
'''

for srs_post in seriousData:
    re.sub('\n+', '  ', srs_post[17]) #Remove newlines
   # re.sub('\n', '  ', srs_post[19])
   # cln_post = re.sub(r'([^\s\w]|_)+', '', cln_post)
    #srs_corpus.append(re.sub('\n', '"', cln_post))
    srs_corpus.append(srs_post)
    #corpus.append(re.sub(r'([^\s\w]|_)+', '', cln_post)) #and then non-alpha

#print(sar_corpus)
#print(srs_corpus)

df = pd.DataFrame(srs_corpus)
df.to_csv('reddit_got.csv', sep='|', encoding='utf-8')