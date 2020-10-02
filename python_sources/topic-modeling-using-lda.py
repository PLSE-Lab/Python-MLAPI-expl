#!/usr/bin/env python
# coding: utf-8

# # Topic Modeling 
# 
# Topic modeling is a statistical model to discover the abstract "topics" that occur in a collection of documents.  
# It is commonly used in text document. But nowadays, in social media analysis, topic modeling is an emerging research area.  
# One of the most popular algorithms used is Latent Dirichlet Allocation which was proposed by  
# [David Blei et al in 2003](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf).   
# Here, I want to perform topic modeling for the upvoted kaggle dataset. 
# 
# Some notes on topic modeling:   
# * To determine the number topics, it is common to use [elbow method](https://en.wikipedia.org/wiki/Elbow_method_(clustering) with [perplexity score](http://qpleple.com/perplexity-to-evaluate-topic-models/) as its cost function.   
# * To evaluate the models, we can calculate [topic coherence](http://qpleple.com/topic-coherence-to-evaluate-topic-models/).   
# * Finally, to interpret the topics, as studied in social science research, there is [triangulation method](http://www.federica.eu/users/9/docs/amaturo-39571-01-Triangulation.pdf).  

# ## Import libraries
# 
# I used LDA model from gensim. Other option is using sklearn.

# In[ ]:


from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models
import pandas as pd
import gensim
import pyLDAvis.gensim


# ## Initiating Tokenizer and Lemmatizer
# 
# Initiate the tokenizer, stop words, and lemmatizer from the libraries.
# 
# * Tokenizer is used to split the sentences into words.  
# * Lemmatizer (a quite similar term to Stemmer) is used to reduce words to its base form.   
# The simple difference is that Lemmatizer considers the meaning while Stemmer does not. 
# 

# In[ ]:


pattern = r'\b[^\d\W]+\b'
tokenizer = RegexpTokenizer(pattern)
en_stop = get_stop_words('en')
lemmatizer = WordNetLemmatizer()


# In[ ]:


from nltk.corpus import stopwords 


# In[ ]:


remove_words = list(stopwords.words('english'))


# In[ ]:


# remove_words


# ## Read the data

# In[ ]:


get_ipython().system('ls ../input/news-category-dataset/')


# In[ ]:


# Input from csv
df = pd.read_json('../input/news-category-dataset/News_Category_Dataset_v2.json',lines = True)

# sample data


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df = df.sample(frac = 0.1)


# ## Perform Tokenization, Words removal, and Lemmatization

# In[ ]:


df["Description"] = df["headline"]+". " +df["short_description"]


# In[ ]:


# list for tokenized documents in loop
texts = []

# loop through document list
for i in df['Description'].iteritems():
    # clean and tokenize document string
    raw = str(i[1]).lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [raw for raw in tokens if not raw in en_stop]
    
    # remove stop words from tokens
    stopped_tokens_new = [raw for raw in stopped_tokens if not raw in remove_words]
    
    # lemmatize tokens
    lemma_tokens = [lemmatizer.lemmatize(tokens) for tokens in stopped_tokens_new]
    
    # remove word containing only single char
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
# I used pre-determined number of topics. It will better calculating perplexity to find the optimum number of topics.    
# *top_topics* shows the sorted topics based on the topic coherence.

# In[ ]:


ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=15, id2word = dictionary, passes=20)
import pprint
pprint.pprint(ldamodel.top_topics(corpus,topn=5))


# ## Visualize the topic model
# 
# Using pyLDAvis, we can create an interactive visualization.

# In[ ]:


pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

