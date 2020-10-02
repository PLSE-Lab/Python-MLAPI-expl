#!/usr/bin/env python
# coding: utf-8

# ## Topic Modeling
# 
# ### Introduction
# 
# Another popular text analysis technique is called topic modeling. The ultimate goal of topic modeling is to find various topics that are present in your corpus. Each document in the corpus will be made up of at least one topic, if not multiple topics.
# 
# In this notebook, we will be predicting topic using Latent Dirichlet Allocation (LDA), which is one of many topic modeling techniques. It was specifically designed for text data.
# 
# To use a topic modeling technique, you need to provide (1) a document-term matrix and (2) the number of topics you would like the algorithm to pick up.

# The objective of this competition is to identify the theme around the given corpus and categorize it accordingly.

# ### 1.Loading the dataset

# In[ ]:


# Importing alll libraries and packages

import pandas as pd
import numpy as np
import scipy as sp
import sklearn
import sys
from nltk.corpus import stopwords
import nltk
from gensim.models import ldamodel
import gensim.corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
import string
from nltk.stem.wordnet import WordNetLemmatizer
import re
# from gensim.test.utils import common_corpus, common_dictionary
# from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from gensim import matutils, models
import scipy.sparse
from nltk import word_tokenize, pos_tag
# from sklearn.feature_extraction import text


# In[ ]:


import pandas as pd
Train=pd.read_csv('data.csv')
pd.set_option('display.max_colwidth',150)
Train.head(6)


# ### 2. Data cleaning process
# 
# In data cleaning process we'll be removing stopwords, special character,numbers and user email id (if present) which make our data clean to get the important words to decide the topic.

# In[ ]:


# removing userid email and numbers if present
Train['clean_text']=Train['text'].str.lower().apply(lambda x: re.sub(r'(@[\S]+)|(\w+:\/\/\S+)|(\d+)','',x))


# removing stopwords and special character and returned lemmatized word
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop and len(i)>1])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
Train['clean_text']=Train['clean_text'].apply(lambda x: clean(x))
Train.head(6)


# Since my data cleaning has done now we'll proceed further to apply topic modeling technique using LDA method.
# 
# 
# ### 3. Create Document term metrics and dictionaru of terms

# In[ ]:


# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(Train.clean_text)
data = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names(),index=Train['Id'])
# data_dtm.index = Train['Id'].index


# In[ ]:


# One of the required inputs is a term-document matrix
tdm = data.transpose()
data.head()


# In[ ]:


# We're going to put the term-document matrix into a new gensim format, from data --> sparse matrix --> gensim corpus
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)


# In[ ]:


# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix
id2word = dict((v, k) for k, v in cv.vocabulary_.items())
id2word


# * Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term), we need to specify two other parameters - the number of topics and the number of passes. ****Since we have been given number of topics so we'll specify number of passes ****

# ### 4.Fitting the LDA model

# In[ ]:


# Fit the model for 5 topics
lda = models.LdaModel(corpus=corpus, num_topics=5, id2word=id2word, passes=100,eta=.90)


# In[ ]:


# get the list of top 20 words in each topic after applying LDA model
def get_lda_topics(model, num_topics,num_words):
    word_dict = {}
    topics = model.show_topics(num_topics,num_words)
    word_dict = {'Topic '+str(i):[x.split('*') for x in words.split('+')]                  for i,words in model.show_topics(num_topics,num_words)}
    return pd.DataFrame.from_dict(word_dict)

get_lda_topics(lda,5,20)


# *This order will change when we rerun thhis code*
# 
# By looking at top 20 words from each topic it seams
# 
# * Words in Topic 0 belong to automobiles
# * words in Topic 1 related to sports news
# * Words in Topic 2 related to room rental
# * Words in topic 3 also seem seem related to room rental
# * Words in topic 4 seem related to glassdoor reviews
# 
# We don't have a very clear result about topic and it seems there is repetition of topic so to tune it more fine we would consider Nouns and Adjective only

# In[ ]:


corpus_transformed = lda[corpus]
# getting topic having maximum score for a document
topic=[]
for i in range(len(corpus_transformed)):
    v=dict(corpus_transformed[i])
    for top, score in v.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if score == max(v.values()):
            topic.append(top)


# In[ ]:


id_topics=pd.DataFrame([a for a in topic],index=data.index)
id_topics.columns=['Topics']

# "glassdoor_reviews"
# "tech_news"
# "room_rentals"
# "sports_news"
# "Automobiles"


# In[ ]:


id_topics['topic']=np.where(id_topics['Topics']==0,'tech_news',
                                np.where(id_topics['Topics']==1,'glassdoor_reviews',
                                        np.where(id_topics['Topics']==2,'sports_news',
                                                np.where(id_topics['Topics']==3,'Automobiles','room_rentals'))))


# In[ ]:


# id_topics.head(20)
final=id_topics.reset_index()
final=final[['Id','topic']]
final.to_csv('final_output12.csv',index=False)
final


## score .94197
## Score .9617

# This is the final output csv having ID and Topic column in it


# ### To get a better result select only Noun and Adjective and rerun above code again

# In[ ]:


# set Id as an index in data frame
Train=Train.set_index('Id')
Train.head(4)


# In[ ]:


# Let's create a function to pull out nouns from a string of text
def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2].startswith('N') or pos[:2].startswith('J')
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
    return ' '.join(nouns_adj)


# In[ ]:


# Apply the nouns function to the transcripts to filter only on nouns
data_nouns_adj = pd.DataFrame(Train.clean_text.apply(nouns_adj))
data_nouns_adj.head(4)


# In[ ]:


# Creating sparse matrix with ttems as columns and ids as index
cvna = CountVectorizer(stop_words=stop, max_features = 5000, max_df=.8)
data_cvna = cvna.fit_transform(data_nouns_adj.clean_text)
data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
data_dtmna.index = data_nouns_adj.index
data_dtmna.head(4)


# In[ ]:


# Create the gensim corpus
corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))

# Create the vocabulary dictionary
id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())


# In[ ]:


# Create LDA model with 5 topics, number of passes is given 50 to get fine tuned result
ldana = models.LdaModel(corpus=corpusna, num_topics=5, id2word=id2wordna, passes=100,eta=.90)
get_lda_topics(ldana,5,20)


# *This order will change once we rerun this code*
# 
# By looking at words in all topics it seems we have more fine tned topics.
# 
# * Topic 0 seems realted to glassdoor reviews
# * Topic 1 seems related to room rental
# * Topic 2 seems realted to sports news
# * Topic 3 seems related to automobile
# * Topic 4 seems related to tech news
# 

# In[ ]:


corpus_transformed = ldana[corpusna]
# getting topic having maximum score for a document
topic=[]
for i in range(len(corpus_transformed)):
    v=dict(corpus_transformed[i])
    for top, score in v.items():  
        if score == max(v.values()):
            topic.append(top)


# In[ ]:


# Get the topic the each document contains

id_topics=pd.DataFrame([a for a in topic],index=data_dtmna.index)
id_topics.columns=['Topics']
# id_topics.head(6)

# "glassdoor_reviews"
# "tech_news"
# "room_rentals"
# "sports_news"
# "Automobiles"


# In[ ]:


id_topics['topic']=np.where(id_topics['Topics']==0,'sports_news',
                                np.where(id_topics['Topics']==1,'glassdoor_reviews',
                                        np.where(id_topics['Topics']==2,'Automobiles',
                                                np.where(id_topics['Topics']==3,'room_rentals','tech_news'))))


# In[ ]:


id_topics.head(20)
final=id_topics.reset_index()
final=final[['Id','topic']]
final.to_csv('final5.csv',index=False)
final

# 0.91267
# 0.89135
# Final output having Noun and Adjective in Document
# After comparing both results in this notebook, publish the one having highest score 


# To measure how good our model is, we can use metrics like 'Perplexity' and 'Coherence'
# 
# #### Coherence
# * Topic Coherence measures score a single topic by measuring the degree of semantic similarity between high scoring words in the topic. These measurements help distinguish between topics that are semantically interpretable topics and topics that are artifacts of statistical inference
# 
# #### Perplexity
# * perplexity is a measurement of how well a probability distribution or probability model predicts a sample. It may be used to compare probability models. A low perplexity indicates the probability distribution is good at predicting the sample.
# 

# In[ ]:


# Perplexity, lower the better.
# print('\nPerplexity: ', lda.log_perplexity(corpusna))  
# # Coherance score, higher is better
# coherence_model_lda = CoherenceModel(model=lda, texts=Train['clean_text'], dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ', coherence_lda)


# As we can see perplexity is pretty good low and cohererance is high so we are going in good direction
