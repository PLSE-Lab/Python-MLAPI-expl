#!/usr/bin/env python
# coding: utf-8

# From https://github.com/ravishchawla/word_2_vec

# In[ ]:


import nltk.data;

from gensim.models import word2vec;

from sklearn.cluster import KMeans;
from sklearn.neighbors import KDTree;

import pandas as pd;
import numpy as np;

import os;
import re;
import logging;
import sqlite3;
import time;
import sys;
import multiprocessing;
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt;
from itertools import cycle;


# In[ ]:


model = word2vec.Word2Vec.load('../input/reddit_word2vec/model_full_reddit');


# In[ ]:


Z = model.wv.vectors


# In[ ]:


print(Z[0].shape)
print(Z[0])


# In[ ]:


def clustering_on_wordvecs(word_vectors, num_clusters):
    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters = num_clusters, init='k-means++');
    idx = kmeans_clustering.fit_predict(word_vectors);
    
    return kmeans_clustering.cluster_centers_, idx;


# In[ ]:


start = time.time();
centers, clusters = clustering_on_wordvecs(Z, 50);
print('Total time: ' + str((time.time() - start)) + ' secs')


# In[ ]:


start = time.time();
centroid_map = dict(zip(model.wv.index2word, clusters));
print('Total time: ' + str((time.time() - start)) + ' secs')


# In[ ]:


def get_top_words(index2word, k, centers, wordvecs):
    tree = KDTree(wordvecs);

    #Closest points for each Cluster center is used to query the closest 20 points to it.
    closest_points = [tree.query(np.reshape(x, (1, -1)), k=k) for x in centers];
    closest_words_idxs = [x[1] for x in closest_points];

    #Word Index is queried for each position in the above array, and added to a Dictionary.
    closest_words = {};
    for i in range(0, len(closest_words_idxs)):
        closest_words['Cluster #' + str(i+1).zfill(2)] = [index2word[j] for j in closest_words_idxs[i][0]]

    #A DataFrame is generated from the dictionary.
    df = pd.DataFrame(closest_words);
    df.index = df.index+1

    return df;


# In[ ]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:


top_words = get_top_words(model.wv.index2word, 20, centers, Z);


# In[ ]:


top_words


# In[ ]:


def display_cloud(cluster_num, cmap):
    wc = WordCloud(background_color="black", max_words=2000, max_font_size=80, colormap=cmap);
    wordcloud = wc.generate(' '.join([word for word in top_words['Cluster #' + str(cluster_num).zfill(2)]]))

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('cluster_' + str(cluster_num), bbox_inches='tight')


# In[ ]:


cmaps = cycle([
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])


col = next(cmaps)
display_cloud(48, col)


# In[ ]:


def get_word_table(table, key, sim_key='similarity', show_sim = True):
    if show_sim == True:
        return pd.DataFrame(table, columns=[key, sim_key])
    else:
        return pd.DataFrame(table, columns=[key, sim_key])[key]


# In[ ]:


get_word_table(model.wv.most_similar_cosmul(positive=['king', 'woman'], negative=['queen']), 'Analogy')


# In[ ]:


model.wv.doesnt_match("apple microsoft samsung tesla".split())


# In[ ]:


model.wv.doesnt_match("trump clinton sanders obama".split())


# In[ ]:


model.wv.doesnt_match("joffrey cersei tywin lannister jon".split())


# In[ ]:


model.wv.doesnt_match("daenerys rhaegar viserion aemon aegon jon targaryen".split())


# In[ ]:


keys = ['musk', 'modi', 'hodor', 'martell', 'apple', 'neutrality', 'snowden', 'batman', 'hulk', 'warriors', 'falcons', 'pizza', ];
tables = [];
for key in keys:
    tables.append(get_word_table(model.wv.similar_by_word(key), key, show_sim=False))


# In[ ]:


pd.concat(tables, axis=1)


# In[ ]:


from gensim.models.keyedvectors import KeyedVectors
model.wv.save_word2vec_format('reddit_word2vec.txt', binary=False)

