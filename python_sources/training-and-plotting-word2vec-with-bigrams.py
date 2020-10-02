#!/usr/bin/env python
# coding: utf-8

# In this kernel we're going to train a word2vec embedding for bigrams using Gensim and then plot the results in 3d using PCA and t-SNE

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


# In[ ]:


import string
import gensim
from gensim.models import phrases, word2vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use('ggplot')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_IV = pd.read_table("../input/SW_EpisodeIV.txt", error_bad_lines=False, delim_whitespace=True, header=0, escapechar='\\')
df_V = pd.read_table("../input/SW_EpisodeV.txt", error_bad_lines=False, delim_whitespace=True, header=0, escapechar='\\')
df_VI = pd.read_table("../input/SW_EpisodeVI.txt", error_bad_lines=False, delim_whitespace=True, header=0, escapechar='\\')

pd.set_option('display.max_colwidth', -1)
df_IV.columns = ['speaker','text']
df_V.columns = ['speaker', 'text']
df_VI.columns = ['speaker', 'text']


# In[ ]:


df_IV.head(4)


# In[ ]:


def prep_text(in_text):
    return in_text.lower().translate(str.maketrans("", "", string.punctuation)).split()


# First let's prep the data

# In[ ]:


df_IV['clean_text'] = df_IV.apply(lambda row: prep_text(row['text']), axis=1)
df_V['clean_text'] = df_V.apply(lambda row: prep_text(row['text']), axis=1)
df_VI['clean_text'] = df_VI.apply(lambda row: prep_text(row['text']), axis=1)
df_IV.head(5)


# and use that to create a large corpus for training

# In[ ]:


df = pd.concat([df_IV, df_V, df_VI])

sentences = df.clean_text.values
# for idx, row in df.iterrows():
#     sentences.append(row['clean_text'])

df.head(5)


# Now we can use [gensim's phrases](https://radimrehurek.com/gensim/models/phrases.html#module-gensim.models.phrases) to find bigrams

# In[ ]:


bigrams = phrases.Phrases(sentences)


# In[ ]:


print(bigrams["this is the death star".split()])


# as you can see, "death star" has been recognised as a bi-gram (indicated by the `_`)
# 
# This gives us something to train a w2v model over. There are a couple of important hyperparameters with gensim you need to think about when training a w2v model (I'm not convinced the defaults are great..):
# 
# - size: this is the dimensionality of the vector space. The rule of thumb here is more dimensions requires more data and time to train, but also can pick up more information about the way words are used. It's typical to see this somewhere in the range of 50 to 300. We don't have that much text here so lets go with just 50
# - min_count: the min number of times a word appears before it's included in the output. 3 is perfectly ok given the size of this corpus
# - iter: this is often overlooked. You can think of it as epochs. The default is 5, which feels very small

# In[ ]:


bigrams[sentences]

model = word2vec.Word2Vec(bigrams[sentences], size=50, min_count=3, iter=20)


# In[ ]:


model.wv.most_similar('death_star')


# ## plotting it
# 
# So let's plot it!
# 
# We're going to use dimensionality reduction to reduce the number of dimensions down from 50 to 3. Specifally we're going to use PCA and t-SNE
# 
# First we need the words in the embedding

# In[ ]:


vocab = list(model.wv.vocab)
len(vocab)


# 1108 is probably too many words to display nicely in a plot - we will thin this down a bit soon.
# 
# We need to now find the vector representing each word, we can do this like so

# In[ ]:


X = model[vocab]


# Lets train PCA over that

# In[ ]:


pca = PCA(n_components=3, random_state=11, whiten=True)
clf = pca.fit_transform(X)

tmp = pd.DataFrame(clf, index=vocab, columns=['x', 'y', 'z'])

tmp.head(3)


# I'm now going to take a random sample of words from the vocab, just 150. Another way to thin this down might be to only pick nouns or only pick the most common words

# In[ ]:


tmp = tmp.sample(150)


# In[ ]:


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(tmp['x'], tmp['y'], tmp['z'], alpha = 0.5)

for word, row in tmp.iterrows():
    x, y, z = row
    pos = (x, y, z)
    ax.text(x, y, z, s=word, size=8, zorder=1, color='k')
    
plt.title('w2v map - PCA')
plt.show()


# Lets print the same thing with t-SNE

# In[ ]:


tsne = TSNE(n_components=3, random_state=11)
clf = tsne.fit_transform(X)

tmp = pd.DataFrame(clf, index=vocab, columns=['x', 'y', 'z'])

tmp.head(3)


# In[ ]:


tmp = tmp.sample(150)


# In[ ]:


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(tmp['x'], tmp['y'], tmp['z'], alpha = 0.5)

for word, row in tmp.iterrows():
    x, y, z = row
    pos = (x, y, z)
    ax.text(x, y, z, s=word, size=8, zorder=1, color='k')
    
plt.title('w2v map - t-SNE')
plt.show()


# In[ ]:




