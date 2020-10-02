import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from bs4 import BeautifulSoup

# This script identifies which communication styles receive highest ranks
# For illustration purposes I defined 3 styles such as Passive, Assertive and Aggressive
# The list of key words must of course be extended

sql_conn = sqlite3.connect('../input/database.sqlite')
df = pd.read_sql("SELECT * FROM May2015 WHERE LENGTH(body) > 5 AND LENGTH(body) < 1000 LIMIT 1000", sql_conn)

col_names = list(df.columns.values)

controversiality = df.controversiality
score = df.score

df.controversiality.hist()
plt.savefig('cont_hist.png')

df.score.hist()
plt.savefig('score.png')


score.plot(x='controversiality', title='controversiality vs score', linestyle='', marker='o')
plt.xlabel('controversiality')
plt.ylabel('score')
plt.show()
plt.savefig('cont_vs_score.png')

word_list = df.body[1:].values

words = []
pos = []
for idx, row in enumerate(word_list):
    # tokenize the text
    words += nltk.tokenize.word_tokenize(word_list[idx])

print(words)


# #separate by parts of speech
# pos.append(nltk.tag.pos_tag(words))

# #define a fxn that makes a dict of all words
# def bag_of_words(words):
#     word_counts = {}
#     for word in words:
#         if word not in word_counts:
#             word_counts[word] = 1
#         else:
#             word_counts[word] = word_counts[word] + 1
#     return word_counts
    
# word_counts = bag_of_words(words)

# train_set, test_set = words[7:], words[:7]
# # classifier = nltk.NaiveBayesClassifier.train(train_set)

# # # classifier = nltk.data.load('classifiers/movie_reviews_NaiveBayes.pickle')

# print(train_set)
