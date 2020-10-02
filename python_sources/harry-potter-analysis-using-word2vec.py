#!/usr/bin/env python
# coding: utf-8

# In[27]:


#Goal : create word vector from Game of Throne dataset
from __future__ import absolute_import, division, print_function # for dependency python 2 to 3
# For word encoding
import codecs
# Regex
import glob
# Concurrency
import multiprocessing
# Dealing with operating system like reading files
import os
# Pretty Printing
import pprint
# Regular Expression
import re
# Natural Language  Toolkit
import nltk
from nltk.corpus import stopwords
# WOrd 2 vec
from gensim.models import Word2Vec
# Dimensional Reductionality
import sklearn.manifold
#math
import numpy as np
#plotting
import matplotlib.pyplot as plt
#data processing 
import pandas as pd
# Data Visualization
import seaborn as sns
from sklearn.manifold import TSNE

get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


book_filenames = sorted(glob.glob("../input/*.txt"))


# In[29]:


print("Books Found :")
book_filenames


# In[30]:


corpus_raw = ""
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with open(book_filename, "rb") as infile:
        corpus_raw += str(infile.read())
        
        print("Corpus is now {0} characters long". format(len(corpus_raw)))
        print()
        


# In[31]:


text = corpus_raw

# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = text.strip()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)


# In[32]:


# Preparing the dataset
sentences = nltk.sent_tokenize(text)


# In[33]:


sentences


# In[34]:


sentences = [nltk.word_tokenize(sentence) for sentence in sentences]


# In[35]:


sentences


# In[36]:


for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]


# In[37]:


sentences


# In[38]:


# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)


# In[39]:


model


# In[40]:


words = model.wv.vocab


# In[41]:


words


# In[42]:


# Finding Word Vectors
vector = model.wv['harry']


# In[43]:


vector


# In[44]:


# Most similar words
similar = model.wv.most_similar('mcgonagall')


# In[45]:


similar


# In[46]:


#distance, similarity, and ranking
def nearest_similarity_cosmul(start1, end1, end2):
    similarities = model.wv.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0] 
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


# In[47]:


nearest_similarity_cosmul("harry", "professor", "snape")
nearest_similarity_cosmul("dumbledore", "elder", "wand")
nearest_similarity_cosmul("lupin", "james", "sirius")


# In[48]:


X = model[model.wv.vocab]


# In[49]:


X


# **Plot Word Vectors Using PCA**

# In[50]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
result = pca.fit_transform(X)
plt.scatter(result[:, 0], result[:, 1])


# In[52]:


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(20, 20)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


# In[ ]:


tsne_plot(model)


# In[ ]:





# In[ ]:




