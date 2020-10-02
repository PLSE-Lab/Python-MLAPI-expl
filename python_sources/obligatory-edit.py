"""
This script looks into gilded and edited comments, and finds the 10 main topics
Interestingly, the "Obligatory edit: thanks for the gold kind stranger" is automatically extracted. Twice.
"""
import sqlite3
import pandas as pd
import numpy as np
from numpy.core.defchararray import count
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from time import time

n_topics = 10
n_top_words = 15

def print_top_words(model, feature_names, n_top_words, W):
    # ordering topics
    topic_weights = W.sum(axis=0)
    topic_ranking = np.argsort(topic_weights)[::-1]
    H_ordered = model.components_[topic_ranking]
    topic_weights = topic_weights[topic_ranking]
    
    for topic_idx, (topic, weight) in enumerate(zip(H_ordered, topic_weights)):
        print("____________Topic #%d: (score = %.1f)____________" % (topic_idx, weight))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

t0 = time()
sql_conn = sqlite3.connect('../input/database.sqlite')

res = pd.read_sql("SELECT edited, body, gilded "
                  "FROM May2015 "
                  "WHERE gilded > 0 "
                  "ORDER BY gilded DESC ",
                  #"LIMIT 100000",
                  sql_conn)
                  
# select gilded posts that have been edited
data = res['body'][res['edited'] > 0]
print("Query done in %0.3fs." % (time() - t0))

tfidf = TfidfVectorizer(max_df=0.90, min_df=2)
X_tfidf = tfidf.fit_transform(data)
print(X_tfidf.shape)

nmf = NMF(n_components=n_topics, init='nndsvdar', random_state=42)
W = nmf.fit_transform(X_tfidf)
print("NMF fit done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words, W)

