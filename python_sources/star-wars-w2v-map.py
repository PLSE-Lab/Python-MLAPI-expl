#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Word2Vec is a method or taking a corpus (collection of text) and turning each word into a vector in some high dimensional space (in the example below it will have 100 dimensions, the default for the library we will use), so that words that are somehow similar in the corpus are close together and words that have opposite meanings are far apart.
# 
# This is an unsupervised task and you see it a lot as an alternative to bag of words to represent text.
# 
# In this notebook we're going to use word2vec and PCA (a method of dimensionality reduction) to show how you can plot word similarities from a corpus. Then update those similarities when new data is available and replot it.
# 
# First, lets load our data set (Star Wars!) and have a look at it.

# In[ ]:


import string
import gensim
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")

get_ipython().run_line_magic('matplotlib', 'inline')

df_IV = pd.read_table("../input/SW_EpisodeIV.txt", error_bad_lines=False)
df_V = pd.read_table("../input/SW_EpisodeV.txt", error_bad_lines=False)
df_VI = pd.read_table("../input/SW_EpisodeVI.txt", error_bad_lines=False)


# In[ ]:


pd.set_option('display.max_colwidth', -1)
df_IV.columns = ['text']
df_V.columns = ['text']
df_VI.columns = ['text']

df_IV.head(5)


# As you can see, this has a slightly awkward layout, below I've written a hacky bit of code to get the actual text and split it into lists of words (sentences) which is going to be the desired input for our word2vec model.

# In[ ]:


def prep_text(in_text):
    return in_text.split('"')[3:-1][0].lower().translate(str.maketrans("", "", string.punctuation)).split()


# In[ ]:


df_IV['clean_text'] = df_IV.apply(lambda row: prep_text(row['text']), axis=1)
df_V['clean_text'] = df_V.apply(lambda row: prep_text(row['text']), axis=1)
df_VI['clean_text'] = df_VI.apply(lambda row: prep_text(row['text']), axis=1)
df_IV.head(5)


# Let's define our model. Rather than build word2vec ourselves, were going to use [gensim](https://radimrehurek.com/gensim/models/word2vec.html).
# 
# There are a couple of ways to train gensim's word2vec, here we're going to be retraining our model so will use the `train` function rather than doing it in `Word2Vec`s constructor. There's a nice short blogpost on this [here](https://rutumulkar.com/blog/2015/word2vec**)

# In[ ]:


sentences_iv = df_IV['clean_text']
sentences_v = df_V['clean_text']
sentences_vi = df_VI['clean_text']

# use a min count of 3 since our corpus is quite small
model = gensim.models.Word2Vec(min_count=3, window=5, iter=50)
model.build_vocab(sentences_iv)
model.train(sentences_iv, total_examples=model.corpus_count, epochs=model.epochs)


# Cool - so do we have here?
# 
# Well as I said earlier, the idea behind word2vec is to build a vector for each word in a corpus - this allows you to talk about how close different words are in a meaningful way. Normally when talking about vectors we use Euclidean distance to talk about the relationships between them, but for word2vec [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) makes more sense - similar words have a value close to 1 and opposite words have a value close to -1.
# 
# So let's use that - what are the words most and least similar to `ship`?

# In[ ]:


model.wv.most_similar('ship')


# In[ ]:


model.wv.most_similar(negative=['ship'])


# So as you can see there are stopwords in here - removing these might be something you would try if you were using this model for something important and weren't getting very good results. We're going to keep them because our objective is just to plot some stuff!
# 
# Another thing I should say is gensim has the ability to automatically identify bigrams (things like `New York`)
# 
# # Star Wars characters
# 
# So our general approach will be this: find the position of each of the words we're interested in (Star Wars character names) in our 100 dimensional space, cluster them and then reduce the dimensions down to just 2 and plot it.
# 
# Just as a side note, when doing this with real data I've found it useful to do clustering like this since going from 100 to 2 dimensions is always going to lose some information about where data sits. It's nice to preserve some of that information if you can.

# In[ ]:


characters = ['kenobi', 'han', 'jabba', 'leia', 'luke', 'threepio', 'r2', 'vader', 'wedge', 'yoda', 'chewbacca', 'lando']


# So lets break this down into steps. Firstly, not all our characters appear in our model. That's for one of two reasons - either the name didn't appear at all in our corpus or it didn't appear often enough for a reasonable vector to be created. So let's create a vocab of the words we're interested in

# In[ ]:


vocab = list(model.wv.vocab)
vocab = list(filter(lambda x: x in characters, vocab))
vocab


# Let's then build a list of the vectors representing our words

# In[ ]:


X = model[vocab]


# Now before going further, we want to cluster this data. I'm just going to (slightly arbitrarily) group the data into 3 clusters.
# 
# Note it's important to do this before doing dimensionality reduction otherwise you're losing information

# In[ ]:


cluster_num = 3

kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(X)
cluster = kmeans.predict(X)


# Let's now use PCA to get down to 2 dimensions. Note that along with fitting the data we're going to transform our `X` vector
# 
# We're then going to put that into a dataframe ready to visualise and manipulate

# In[ ]:


pca = PCA(n_components=2, random_state=11, whiten=True)
clf = pca.fit_transform(X)

tmp = pd.DataFrame(clf, index=vocab, columns=['x', 'y'])

tmp.head(3)


# Then let's label and plot that

# In[ ]:


tmp['cluster'] = None
tmp['c'] = None

count = 0
for index, row in tmp.iterrows():
    tmp['cluster'][index] = cluster[count]
    tmp['c'][index] = characters[count]
    count += 1
    
for i in range(cluster_num):
    values = tmp[tmp['cluster'] == i]
    plt.scatter(values['x'], values['y'], alpha = 0.5)

for word, row in tmp.iterrows():
    x, y, cat, character = row
    pos = (x, y)
    plt.annotate(character, pos)
    
plt.axis('off')
plt.title('Episode IV')
plt.show()


# So that's episode IV, what about V and VI?
# 
# Well we can now take our existing model and update it with the corpus from episode V

# In[ ]:


model.build_vocab(sentences_v, update=True)
model.train(sentences_v, total_examples=model.corpus_count, epochs=model.epochs)

vocab = list(model.wv.vocab)
vocab = list(filter(lambda x: x in characters, vocab))
    
X = model[vocab]


# This time when you run your dimensionality reduction make sure you only transform and don't fit a new model, otherwise it will become hard to reason about how the plot from episode iv relates the plot at the end of spisode v

# In[ ]:


clf = pca.transform(X)


# In[ ]:


cluster = kmeans.predict(X)

tmp = pd.DataFrame(clf, index=vocab, columns=['x', 'y'])

tmp['cluster'] = None
tmp['c'] = None

count = 0
for index, row in tmp.iterrows():
    tmp['cluster'][index] = cluster[count]
    tmp['c'][index] = characters[count]
    count += 1
    
for i in range(cluster_num):
    values = tmp[tmp['cluster'] == i]
    plt.scatter(values['x'], values['y'], alpha = 0.5)

for word, row in tmp.iterrows():
    x, y, cat, character = row
    pos = (x, y)
    plt.annotate(character, pos)
    
plt.axis('off')
plt.title("Episodes IV and V")
plt.show()


# and then finally we do the same for episode vi

# In[ ]:


model.build_vocab(sentences_vi, update=True)
model.train(sentences_vi, total_examples=model.corpus_count, epochs=model.epochs)

vocab = list(model.wv.vocab)
vocab = list(filter(lambda x: x in characters, vocab))
    
X = model[vocab]

clf = pca.transform(X)
cluster = kmeans.predict(X)

tmp = pd.DataFrame(clf, index=vocab, columns=['x', 'y'])
    
tmp['cluster'] = None
tmp['c'] = None
count = 0
for index, row in tmp.iterrows():
    tmp['cluster'][index] = cluster[count]
    tmp['c'][index] = characters[count]
    count += 1
    
for i in range(cluster_num):
    values = tmp[tmp['cluster'] == i]
    plt.scatter(values['x'], values['y'], alpha = 0.5)

for word, row in tmp.iterrows():
    x, y, cat, character = row
    pos = (x, y)
    plt.annotate(character, pos)
    
plt.axis('off')
plt.title("Episodes IV, V and VI")
plt.show()


# As you can see, over the life of the Star Wars series the positions of each character moves in a way that can clearly see when plotted
# 
# Sadly there isn't very much text in this dataset (about 1000 lines per episode) to have much confidence in what this plot is telling us, but it's a bit of fun none the less!

# In[ ]:




