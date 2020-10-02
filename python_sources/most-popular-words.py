
import sqlite3 
import pandas as pd 
import operator

sql_conn = sqlite3.connect('../input/database.sqlite')


res = pd.read_sql("SELECT body FROM May2015 ORDER BY RANDOM() limit 1000000", sql_conn)

allWords = {}
for comment in res['body']:
    # print(comment)
    words = comment.lower().split()
    for word in words: 
        if word in allWords: 
            allWords[word] += 1
        else:
            allWords[word] = 1
sorted_allWords = sorted(allWords.items(), key=operator.itemgetter(1))
sorted_allWords.reverse()
for word in sorted_allWords:
    print(word)