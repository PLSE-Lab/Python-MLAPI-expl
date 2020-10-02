import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import codecs
import re

import matplotlib.pyplot
from collections import Counter

sql_conn = sqlite3.connect('../input/database.sqlite')

sql_conn = sqlite3.connect('../input/database.sqlite')
'''
sql_cmd = "SELECT author_flair_text, score, body FROM May2015 WHERE subreddit = 'soccer' AND author_flair_text IS NOT NULL " #

data = pd.read_sql(sql_cmd, sql_conn)

# Python reads as ascii so some characters throw errors if not converted
flair_utf8 = [codecs.encode(i,'utf-8') for i in data.author_flair_text]
data['flair'] = pd.Series(flair_utf8)

flair_counts =  data.flair.value_counts()
top_flairs = flair_counts[flair_counts >= 1000].index

#1: relationship betweeen flairs and upvote: 
#print(data[data.flair.isin(top_flairs)].groupby('flair').score.mean())
#print(flair_counts[flair_counts >= 1000])

comment_utf8 = [codecs.encode(i,'utf-8') for i in data.body]
data['body'] = pd.Series(comment_utf8)

#2: tdf-if of downvoated comments: 



'''

sql_cmd = "SELECT author, subreddit FROM May2015"
data = pd.read_sql(sql_cmd, sql_conn)


post_counts =  data.author.value_counts()
top_posters = post_counts[flair_counts >= 500].index

freq_posters = data[data.author.isin(top_posters)]
#print(flair_counts[flair_counts >= 1000])
#data.to_csv('data.csv')

len(freq_posters)
