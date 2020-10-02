#!/usr/bin/env python
# coding: utf-8

# # Synopsis Visualization 
# 
# I modified this code from https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html by adding visualization to the resulting topics.

# ## Import libraries
# 
# Here I used LDA model from gensim. Personally I find it simpler to implement than using sklearn.

# In[ ]:


from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models
import pandas as pd
import gensim
import pyLDAvis.gensim


# ## Initiating Tokenizer and Stemmer
# 
# In addition, I create a list of words which are common in this context to be removed before performing topic modeling.

# In[ ]:


pattern = r'\b[^\d\W]+\b'
tokenizer = RegexpTokenizer(pattern)
en_stop = get_stop_words('en')
lemmatizer = WordNetLemmatizer()

# remove certain words
tobe_removed = []


# ## Read the data and make it into a list

# In[ ]:


# Input from csv
df = pd.read_csv('../input/datasynopsis-all-share-new.csv',sep='|')

# sample data
print(df['Synopsis'].head(2))


# ## Perform Tokenization, words removal, and Lemmatization

# In[ ]:


# list for tokenized documents in loop
texts = []

# loop through document list
for i in df['Synopsis'].iteritems():
    # clean and tokenize document string
    raw = str(i[1]).lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [raw for raw in tokens if not raw in en_stop]
    
    # remove some words from tokens
    new_stopped_tokens = [raw for raw in stopped_tokens if not raw in tobe_removed]
    
    # lemmatize tokens
    lemma_tokens = [lemmatizer.lemmatize(tokens) for tokens in new_stopped_tokens]
    
    # remove
    new_lemma_tokens = [raw for raw in lemma_tokens if not len(raw) == 1]
    
    # add tokens to list
    texts.append(new_lemma_tokens)

# sample data
print(texts[0])


# ## Create term dictionary and document-term matrix

# In[ ]:


# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]


# ## Generate LDA model
# 
# Here I used pre-determined number of topics. It will better calculating perplexity to find the optimum number of topics.

# In[ ]:


ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=10, num_words=5))


# ## Visualize the topic model
# 
# Using pyLDAvis, we can create an interactive visualization.

# In[ ]:


pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

