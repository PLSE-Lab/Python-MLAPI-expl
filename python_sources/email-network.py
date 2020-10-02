# The data comes both as CSV files and a SQLite database

import pandas as pd
import sqlite3

# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')

# It's yours to take from here!
df = pd.read_sql_query("select * from Emails", con)
# print(df.columns.values)
# ['Id' 'DocNumber' 'MetadataSubject' 'MetadataTo' 'MetadataFrom'
#  'SenderPersonId' 'MetadataDateSent' 'MetadataDateReleased'
#  'MetadataPdfLink' 'MetadataCaseNumber' 'MetadataDocumentClass'
#  'ExtractedSubject' 'ExtractedTo' 'ExtractedFrom' 'ExtractedCc'
#  'ExtractedDateSent' 'ExtractedCaseNumber' 'ExtractedDocNumber'
#  'ExtractedDateReleased' 'ExtractedReleaseInPartOrFull' 'ExtractedBodyText'
#  'RawText']

# print(df.count())
# 7945 emails

# print(df[['MetadataSubject', 'MetadataFrom', 'MetadataTo']])

# print(df.MetadataSubject.nunique())
# 4174
import nltk
from nltk.corpus import stopwords

filter_set = set(stopwords.words('english'))
feature_set = set()
for subject in df.MetadataSubject.unique():
    tokens = nltk.word_tokenize(subject)
    tokens = [token for token in tokens if token not in filter_set ]
    for token in tokens:
        feature_set.add(token)

# print(len(feature_set))
# 5293
# print(feature_set)
feature_list = list(feature_set)
feature_set = set(feature_list)

emails = []
persons = []
from__ = []
to__ = []

import numpy as np

for subject in df.MetadataSubject.unique():
    tokens = nltk.word_tokenize(subject)
    from_ = [person.replace(";","") for person in df[df.MetadataSubject == subject].MetadataFrom.values if person != '']
    to_ = [person.replace(";","") for person in df[df.MetadataSubject == subject].MetadataTo.values if person != '']
    from__.append(set(from_))
    to__.append(set(to_))
    persons.append(from_+to_)
    tokens = set([token for token in tokens if token not in filter_set ])
    email = [1 if feature in tokens else 0 for feature in feature_set]
    emails.append(email)

from sklearn.cluster import MiniBatchKMeans

k = 5
X = np.matrix(emails)
km = MiniBatchKMeans(n_clusters=5).fit(X)

top_k_tags_per_cluster = {}
for cluster, center in enumerate(km.cluster_centers_):
    i_top_k = np.argpartition(center, -k)[-k:]
    for i in i_top_k:
        if top_k_tags_per_cluster.get(cluster) is None:
            top_k_tags_per_cluster[cluster] = set()
        top_k_tags_per_cluster[cluster].add(feature_list[i])

predictions = km.predict(X)
clusters = {}
for cluster in predictions:
    if clusters.get(cluster) is None:
        clusters[cluster] = 0
    clusters[cluster] += 1
print(clusters)

import networkx as nx
import matplotlib.pyplot as plt

color = {0:'red', 1:'blue',2:'gray',3:'green',4:'black'}
G = nx.Graph()
for i, cluster in enumerate(predictions):
    for from_person in from__[i]:
        for to_person in to__[i]:
            # print(cluster, ";", ",".join(top_k_tags_per_cluster[cluster]), ";", from_person, ";", to_person)
            G.add_edge(from_person, to_person, color=color[cluster])

nx.draw_random(G)
plt.axis('off')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.savefig("network.png")

