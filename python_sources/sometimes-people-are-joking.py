import sqlite3
import pandas as pd
import collections
from sklearn.feature_extraction.text import CountVectorizer

sql_conn = sqlite3.connect('../input/database.sqlite')
df = pd.read_sql('SELECT author, score, gilded, subreddit, body FROM May2015 WHERE LENGTH(body)>100 AND subreddit="Jokes" AND gilded!=0 LIMIT 10000', sql_conn)

# Fifty percent of what people say when they are joking is true.

df.sort(['score', 'gilded'], ascending=[False, False], inplace=True)

ignore = {'The','the','A','a','am','if', 'If','in','In','it','It',
          'of','Of','or','Or','they','They','and','that','That',
          'to','To','I','me','my','he','him','his','she','her','you','are','we',
          'can','want','for','how','just','but','But','because','is','was','not',
          'could','As','as','be','will','this','This','never','know','Could',
          'get','gets','with','up','do','does','Your','My','such','which',
          'not','Not','have','has','them','Much','here','over','about','by','at'}

most_frequent_word={}
for i, row in df.iterrows():
    words = CountVectorizer(stop_words='english').build_tokenizer()(str(row['body']))
    most_frequent_word[i] = collections.Counter(x for x in words if x not in ignore).most_common(3)

print(most_frequent_word)