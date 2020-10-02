import sqlite3
import pandas as pd
import numpy as np
from textblob import TextBlob
from collections import Counter
from flask import jsonify
conn = sqlite3.connect('../input/database.sqlite')

soccer = conn.execute("SELECT subreddit, author_flair_text, body FROM May2015 WHERE lower(subreddit) LIKE 'soccer'")

flair_list = []

for text in soccer:
    if text[1] != '':
        flair_list.append(text[1])


c = Counter(flair_list)

commonOnes = c.most_common(20)

print(commonOnes)

#db = pd.read_sql("SELECT author_flair_text,body FROM May2015 WHERE lower(subreddit) LIKE 'soccer'" , conn)
    
#for t in teams:
#    print (t),
#    senti = {}
#    for row in db.iterrows():
#        if t == row['author_flair_text']:
#           comment = TextBlob(row['body'])
#            comment_sentiment = {'polarity' : comment.polarity , 'subjectivity' : comment.subjectivity}
#            senti.append(Counter(comment_sentiment))   
#    print (senti)        
    
#print (teams)

