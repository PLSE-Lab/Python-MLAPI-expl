from __future__ import division
import sqlite3, time, csv, re, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix



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
cur = sql_conn.cursor()

cur.execute("SELECT subreddit, body, score FROM May2015\
                                WHERE body LIKE '% irony'\
                                LIMIT " + str(sar_lmt))


sql_conn2 = sqlite3.connect('irony.db')
sql_conn2.execute("CREATE TABLE irony (subreddit text,body text,score real)")
reddit_irony_dict = {}

for row in cur.fetchall():
    sql_conn2.execute("INSERT into irony values(?,?,?)",row)    

'''
for row in cur.fetchall():
    if row[0] not in reddit_irony_dict:
        reddit_irony_dict[row[0]] = []
    reddit_irony_dict[row[0]].append(row)
    
for key in reddit_irony_dict:
    print (key + " " + str(len(reddit_irony_dict[key])))
'''
    
    
