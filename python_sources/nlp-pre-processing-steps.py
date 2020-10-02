#!/usr/bin/env python
# coding: utf-8

# # Best Practices for Preprocessing Natural Language Data

# In this notebook, we improve the quality of our Project Gutenberg word vectors by adopting best-practices for preprocessing natural language data.
# 
# **N.B.:** Some, all or none of these preprocessing steps may be helpful to a given downstream application. 

# #### Load dependencies

# In[ ]:


# the initial block is copied from creating_word_vectors_with_word2vec.ipynb
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from sklearn.manifold import TSNE
import pandas as pd
from bokeh.io import output_notebook, output_file
from bokeh.plotting import show, figure
import string
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


nltk.download('punkt')


# In[ ]:


nltk.download('stopwords')


# #### Load data

# In[ ]:


nltk.download('gutenberg')


# In[ ]:


from nltk.corpus import gutenberg


# In[ ]:


len(gutenberg.fileids())


# In[ ]:


gutenberg.fileids()


# In[ ]:


gberg_sent_tokens = sent_tokenize(gutenberg.raw())


# In[ ]:


gberg_sent_tokens[0:5]


# In[ ]:


gberg_sent_tokens[1]


# In[ ]:


word_tokenize(gberg_sent_tokens[1])


# In[ ]:


word_tokenize(gberg_sent_tokens[1])[14]


# In[ ]:


gberg_sents = gutenberg.sents()


# In[ ]:


gberg_sents[0:5]


# In[ ]:


len(gutenberg.words())


# In[ ]:


len(gutenberg.sents())


# #### Iteratively preprocess a sentence

# ##### a tokenized sentence: 

# In[ ]:


gberg_sents[5]


# ##### to lowercase: 

# In[ ]:


# CODE HERE
[w.lower() for w in gberg_sents[5]]


# ##### remove stopwords and punctuation: 

# In[ ]:


stpwrds = stopwords.words('english') + list(string.punctuation)


# In[ ]:


stpwrds


# In[ ]:


# CODE HERE
[w.lower() for w in gberg_sents[5] if w not in stpwrds]


# #### stem words: 

# In[ ]:


stemmer = PorterStemmer()


# In[ ]:


# CODE HERE
[stemmer.stem(w.lower()) for w in gberg_sents[5] if w not in stpwrds]


# ##### handle bigram collocations:

# In[ ]:


phrases = Phrases(gberg_sents) # train detector


# In[ ]:


bigram = Phraser(phrases) # create a more efficient Phraser object for transforming sentences


# In[ ]:


bigram.phrasegrams # output count and score of each bigram


# In[ ]:


"Jon lives in New York City".split()


# In[ ]:


# CODE HERE
bigram["Jon lives in New York City".split()]


# #### Preprocess the corpus

# In[ ]:


lower_sents = []
for s in gberg_sents:
    lower_sents.append( [w.lower() for w in s if w not in list(string.punctuation)] )


# In[ ]:


lower_sents[0:5]


# In[ ]:


lower_bigram = Phraser(Phrases(lower_sents))


# In[ ]:


lower_bigram.phrasegrams # miss taylor, mr woodhouse, mr weston


# In[ ]:


lower_bigram["jon lives in new york city".split()]


# In[ ]:


lower_bigram = Phraser(Phrases(lower_sents, min_count=32, threshold=64))
lower_bigram.phrasegrams


# In[ ]:


# as in Maas et al. (2001):
# - leave in stop words ("indicative of sentiment")
# - no stemming ("model learns similar representations of words of the same stem when data suggests it")
clean_sents = []
for s in lower_sents:
    clean_sents.append(lower_bigram[s])


# In[ ]:


clean_sents[0:9]


# In[ ]:


clean_sents[6] # could consider removing stop words or common words


# #### Run word2vec

# In[ ]:


# max_vocab_size can be used instead of min_count (which has increased here)
model = Word2Vec(sentences=clean_sents, size=64, sg=1, window=10, min_count=10, seed=42, workers=8)
model.save('../clean_gutenberg_model.w2v')


# #### Explore model

# In[ ]:


# skip re-training the model with the next line:  
model = gensim.models.Word2Vec.load('../clean_gutenberg_model.w2v')


# In[ ]:


len(model.wv.vocab) # 17k with raw data


# In[ ]:


len(model['dog'])


# In[ ]:


model['dog']


# In[ ]:


model.most_similar('dog')


# In[ ]:


model.most_similar('think')


# In[ ]:


model.most_similar('day')


# In[ ]:


model.doesnt_match("morning afternoon evening dog".split())


# In[ ]:


model.similarity('morning', 'dog')


# In[ ]:


model.most_similar('ma_am') 


# In[ ]:


model.most_similar(positive=['father', 'woman'], negative=['man']) 


# #### Reduce word vector dimensionality with t-SNE

# In[ ]:


tsne = TSNE(n_components=2, n_iter=1000)


# In[ ]:


X_2d = tsne.fit_transform(model[model.wv.vocab])


# In[ ]:


coords_df = pd.DataFrame(X_2d, columns=['x','y'])
coords_df['token'] = model.wv.vocab.keys()


# In[ ]:


coords_df.to_csv('../clean_gutenberg_tsne.csv', index=False)


# #### Visualise 

# In[ ]:


coords_df = pd.read_csv('../clean_gutenberg_tsne.csv')


# In[ ]:


coords_df.head()


# In[ ]:


_ = coords_df.plot.scatter('x', 'y', figsize=(12,12), marker='.', s=10, alpha=0.2)


# In[ ]:


output_notebook()


# In[ ]:


subset_df = coords_df.sample(n=5000)


# In[ ]:


p = figure(plot_width=800, plot_height=800)
_ = p.text(x=subset_df.x, y=subset_df.y, text=subset_df.token)


# In[ ]:


show(p)


# In[ ]:


# output_file() here

