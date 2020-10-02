#!/usr/bin/env python
# coding: utf-8

# # NLP Preprocessing Pipeline
# 
# Greetings! In this notebook we will review common preprocessing techniques that applied to natural language data.
# 
# For this exploration, we will use the Gutenberg corpus, a collection of classic, free-to-use books.

# In[ ]:


import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, gutenberg
from nltk.stem.porter import *
import gensim
from gensim.models.phrases import Phraser, Phrases
from gensim.models.word2vec import Word2Vec
import string
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.io import output_notebook, output_file
from bokeh.plotting import show, figure
# nltk.download('gutenberg') # Need to download resource the first time
# nltk.download('stopwords') # Need to download resource the first time


# ### List Available Texts

# In[ ]:


gutenberg.fileids()


# ### Load a Text

# In[ ]:


alice = gutenberg.raw('carroll-alice.txt')


# ### Tokenize
# 
# Tokenization is used to break the corpus into sentences and individual words within these sentences.

# In[ ]:


sentences_v1 = sent_tokenize(gutenberg.raw()) # Break text into sentences
sentences_v1[0:5]


# In[ ]:


print(word_tokenize(sentences_v1[1])) # Break a sentence into words


# In[ ]:


sentences_v2 = gutenberg.sents('carroll-alice.txt') # Gutenberg library comes with built in sents() method
sentences_v2


# ### Convert to Lowercase
# 
# For small datasets, all characters are converted to lowercase. For larger datasets, where there are many examples of words with differing capitalizations, it can be helpful to skip this step.

# In[ ]:


def lowercase(corpus):
    sentences_lower = []

    for sentence in corpus:
        sentences_lower.append([w.lower() for w in sentence])

    return sentences_lower

sentences_lower = lowercase(sentences_v2)
sentences_lower[0:2]


# ### Remove Stopwords
# 
# The choice to remove stopwords is context dependent. Sentiment analysis often benefits from including stopwords such as "not", in order to differentiate phrases like "not good".

# In[ ]:


stopwords = stopwords.words('english')
stopwords[0:10]


# In[ ]:


def remove_stopwords(corpus):
    no_stopwords = []

    for sentence in corpus:
        no_stopwords.append([w for w in sentence if w not in stopwords])
    
    return no_stopwords

no_stopwords = remove_stopwords(sentences_v2)
no_stopwords[0:2]


# ### Remove Punctuation
# 
# Punctuation is almost always removed, unless specific punctuation is important for representation learning (ex: "?" fo question answering tasks).

# In[ ]:


punctuation = string.punctuation
punctuation


# In[ ]:


# Slow
def remove_punctuation(corpus):
    no_punctuation = []

    for sentence in corpus:
        no_punctuation.append([c for c in sentence if c not in list(punctuation)])

    return no_punctuation

no_punctuation = remove_punctuation(sentences_v2)
no_punctuation[0:2]


# In[ ]:


# Faster
def remove_punctuation(corpus):
    no_punctuation = []

    for sentence in corpus:
        no_punctuation.append([word.translate(str.maketrans("","",punctuation)) for word in sentence if word.translate(str.maketrans("","",punctuation)) != ""])

    return no_punctuation

no_punctuation = remove_punctuation(sentences_v2)
no_punctuation[0:2]


# ### Stemming
# 
# Truncating words down to their stems is often used with smaller datasets because words with similar meanings are grouped into a single token.

# In[ ]:


def apply_stemming(corpus):
    stemmer = PorterStemmer()
    stems = []

    for sentence in corpus:
        stems.append([stemmer.stem(w) for w in sentence])
        
    return stems

stems = apply_stemming(sentences_v2)
stems[0:2]


# ### Handle Bi-Grams
# 
# Some words that co-occur frequently are better interpreted as one token, rather than two.

# In[ ]:


# Train the detector on the corpus; apply a min_count and threshold to refine
phrases = Phrases(sentences_lower)
# Create a dictionary for parsing bi-grams
bigram = Phraser(phrases)


# In[ ]:


# Print all bigrams (long list)
# bigram.phrasegrams 


# In[ ]:


# Sort bigrams by score
sorted_bigrams = {k:v for k,v in sorted(bigram.phrasegrams.items(), key=lambda item: item[1], reverse=True)}


# In[ ]:


# Print top 10 bigrams by score (notice that none of these are actual bi-grams, so we'd want to tweak the parameters of Phrases() to cut out the noise)
for i, (k, v) in enumerate(sorted_bigrams.items()):
    if i < 10:
        print(k,v)


# In[ ]:


def apply_bigrams(corpus):
    bigram = Phraser(Phrases(corpus))
    sentences_bigrams = []

    for sentence in corpus:
        sentences_bigrams.append(bigram[sentence])
        
    return sentences_bigrams

sentences_bigrams = apply_bigrams(sentences_v2)
sentences_bigrams[0:2]


# ### Apply Word2Vec
# 
# Word2Vec produces word embeddings, i.e. it converts text into vectors within high dimensional space, which can then be passed into a machine learning model.
# 
# The larger the corpus, the more accurate are the numerical representations of word features in vectorspace.
# 
# Below, the entirety of the Gutenberg corpus is cleaned and passed into the Word2Vec model.

# In[ ]:


# Prepare corpus
tokenized_corpus = gutenberg.sents()
clean_corpus = remove_punctuation(remove_stopwords(lowercase(tokenized_corpus)))


# In[ ]:


# size = number of dimensions
# sg = skip-gram or CBOW architecture
# window = number of context words to consider
# iter = number of epochs
# min-count = number of times word must appear in corpus in order to fit into word vector space
# workers = number of processing cores

# Run Word2Vec
model = Word2Vec(sentences=clean_corpus, size=64, sg=1, window=10, iter=5, min_count=10, workers=4)


# In[ ]:


# Number of words in our corpus (after min count threshold applied)
len(model.wv.vocab)


# In[ ]:


# View entire vocabulary (long list)
# model.wv.vocab


# In[ ]:


# View a word's location in n-dimensional space (here, 64-dimensional)
model.wv['queen']


# In[ ]:


# View similar words
model.wv.most_similar("queen", topn=3)


# In[ ]:


# Extract a 2D represention of the word vector space
tsne = TSNE(n_components=2, n_iter=1000)
wv_2d = tsne.fit_transform(model.wv[model.wv.vocab])


# In[ ]:


# Compile a dataframe with x,y coords
word_coords = pd.DataFrame(wv_2d, columns=["x", "y"])
word_coords["token"] = model.wv.vocab.keys()
word_coords.head()


# In[ ]:


# Plot 2D vectorspace (not very helpful, but still cool)
sns.scatterplot(x="x", y="y", data=word_coords)
plt.title("2D Representation of Entire Vector Space")
plt.show()


# In[ ]:


# Plot an interactive chart of a sample of the word vectors within the vectorspace (for speed & clarity)
output_notebook()
corpus_sample = word_coords.sample(1000)
p = figure(plot_width=800, plot_height=800)
_ = p.text(x=corpus_sample["x"], y=corpus_sample["y"], text=corpus_sample["token"])
show(p)


# ### Conclusion
# 
# I hope you enjoyed this quick review of NLP preprocessing techniques (especially the interactive word vectorspace plot at the end - that's my favorite!). 
# 
# Please leave a comment below if this helped you out or if you have suggestions for improvement.
# 
# And until next time, happy coding :)
