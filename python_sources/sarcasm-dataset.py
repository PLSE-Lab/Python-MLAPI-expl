from __future__ import division
import sqlite3, time, csv, re, random
import pandas as pd

#Parameters
srs_lmt = 10000 #serious posts to train on
sar_lmt = 10000 #sarcastic posts to train on

print('Querying DB...\n')
sql_conn = sqlite3.connect('../input/database.sqlite')


sarcasmData = sql_conn.execute("SELECT body FROM May2015\
                                WHERE body LIKE '% /s'\
                                LIMIT " + str(sar_lmt))

seriousData = sql_conn.execute("SELECT body FROM May2015\
                                WHERE body NOT LIKE '%/s%'\
                                LIMIT " + str(srs_lmt))
                                
print('Building Corpora...\n')
corpus, sar_corpus, srs_corpus = [], [], []


for sar_post in sarcasmData:
    cln = sar_post[0].replace('\n', " ").replace('\r', " ")
    sar_corpus.append('"'+cln+'"')

for srs_post in seriousData:
    cln = srs_post[0].replace('\n', " ").replace('\r', " ")
    srs_corpus.append('"'+cln+'"')

df1 = pd.DataFrame(sar_corpus)
df1.to_csv('reddit_sarcasm.csv', sep=',', encoding='utf-8')

df = pd.DataFrame(srs_corpus)
df.to_csv('reddit_serious.csv', sep=',', encoding='utf-8')