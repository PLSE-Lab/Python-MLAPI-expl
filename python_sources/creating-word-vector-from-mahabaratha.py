#!/usr/bin/env python
# coding: utf-8

# **Word Vector from Mahabaratha**

# In[ ]:


from __future__ import absolute_import, division, print_function


# In[ ]:


import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re


# In[ ]:


import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')


# Set up logging

# In[ ]:


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Prepare Corpus
# Load books from files

# In[ ]:


book_filenames = sorted(glob.glob("../input/*.txt"))


# In[ ]:


print("Found books:")
book_filenames


# Combine the books into one string

# In[ ]:


corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()


# Split the corpus into sentences

# In[ ]:


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[ ]:


raw_sentences = tokenizer.tokenize(corpus_raw)


# In[ ]:


#convert into a list of words
#remove unnnecessary,, split into words, no hyphens
#list of words
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words


# In[ ]:


#sentence where each word is tokenized
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


# In[ ]:


print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))


# In[ ]:


token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))


# Train Word2Vec

# In[ ]:


#ONCE we have vectors
#step 3 - build model
#3 main tasks that vectors help with
#DISTANCE, SIMILARITY, RANKING

# Dimensionality of the resulting word vectors.
#more dimensions, more computationally expensive to train
#but also more accurate
#more dimensions = more generalized
num_features = 300
# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
#0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
#random number generator
#deterministic, good for debugging
seed = 1


# In[ ]:


thrones2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)


# In[ ]:


thrones2vec.build_vocab(sentences)


# In[ ]:


print("Word2Vec vocabulary length:", len(thrones2vec.wv.vocab))


# Start training, this might take a minute or two...

# In[ ]:


thrones2vec.train(sentences)


# Save to file, can be useful later

# In[ ]:


if not os.path.exists("trained"):
    os.makedirs("trained")


# In[ ]:


thrones2vec.save(os.path.join("trained", "thrones2vec.w2v"))


# Explore the trained model.

# In[ ]:


thrones2vec = w2v.Word2Vec.load(os.path.join("trained", "thrones2vec.w2v"))


# Compress the word vectors into 2D space and plot them

# In[ ]:


#my video - how to visualize a dataset easily
tsne = sklearn.manifold.TSNE(n_components=3, random_state=0)


# In[ ]:


all_word_vectors_matrix = thrones2vec.syn0


# Train t-SNE, this could take a minute or two...

# In[ ]:


all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)


# Plot the big picture

# In[ ]:


points = pd.DataFrame(
    [
        (word, coords[0], coords[1], coords[2])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[thrones2vec.wv.vocab[word].index])
            for word in thrones2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y", "z"]
)

points.to_csv('Cluster_threeD.csv',header=True, index_label='id')


# In[ ]:


points.head(10)


# In[ ]:


sns.set_context("poster")


# In[ ]:


points.plot.scatter("x", "y", c = "z",s=10, figsize=(12, 12))


# In[ ]:


def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


# People related to Kingsguard ended up together

# In[ ]:


plot_region(x_bounds=(4.0, 9.2), y_bounds=(-0.5, -0.1))


# Food products are grouped nicely as well. Aerys (The Mad King) being close to "roasted" also looks sadly correct

# In[ ]:


plot_region(x_bounds=(0, 1), y_bounds=(4, 4.5))


# Explore semantic similarities between book characters. Words closest to the given word

# In[ ]:


thrones2vec.most_similar("Krishna")


# In[ ]:


thrones2vec.most_similar("Arjuna")


# In[ ]:


thrones2vec.most_similar("Karna")


# In[ ]:


thrones2vec.most_similar("Dronacharya")


# Linear relationships between word pairs

# In[ ]:


def nearest_similarity_cosmul(start1, end1, end2):
    similarities = thrones2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


# In[ ]:


nearest_similarity_cosmul("Krishna", "Karna", "Arjuna")
nearest_similarity_cosmul("Kunti", "Pandu", "Krishna")


# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go


# In[ ]:


scatter = dict(
    mode = "markers",
    name = "y",
    type = "scatter3d",    
    x = points['x'], y = points['y'], z = points['z'],
    marker = dict( size=2, color="rgb(23, 190, 207)" )
)

clusters = dict(
    alphahull = 7,
    name = "y",
    opacity = 0.1,
    type = "mesh3d",    
    x = points['x'], y = points['y'], z = points['z']
)


# In[ ]:


layout = dict(
    title = '3d point clustering',
    scene = dict(
        xaxis = dict( zeroline=False ),
        yaxis = dict( zeroline=False ),
        zaxis = dict( zeroline=False ),
    )
)
fig = dict( data=[scatter, clusters], layout=layout )
# Use py.iplot() for IPython notebook
#py.iplot(fig, filename='3d point clustering')


# In[ ]:




