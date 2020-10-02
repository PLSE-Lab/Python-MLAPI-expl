#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# In this notebook I am exploring few NLP papers dataset. I am doing text analysis on paper Abstracts & Titles using nltk for text preprocessing, Topic modeling using LDA & NMF, Paper similarity analysis using TF-IDF vectors. 
# 
# Please share your valuable feedback & upvote if you learn something new today from this analysis. 
# 

# In[1]:


import pandas as pd 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
import gensim.corpora as corpora
import re
import numpy as np
from pprint import pprint
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)  


# In[2]:


papers = pd.read_csv('../input/201812_CL_Github.csv')


# In[3]:


papers.head()


# In[4]:


papers.shape
#Total 106 papers given


# In[5]:


papers.info()


# #### Data Preprocessing

# In[6]:


#Removing symbols from Abstracts

papers['Abstract_Cleaned'] = papers.apply(lambda row: (re.sub("[^A-Za-z0-9' ]+", ' ', row['Abstract'])),axis=1)


# In[7]:


# Tokenization

papers['Abstract_Cleaned'] = papers.apply(lambda row: (word_tokenize(row['Abstract_Cleaned'])), axis = 1)


# In[8]:


# Removing Stopwords

stop_words = set(stopwords.words('english'))
papers['Abstract_Cleaned'] = papers.apply(lambda row: ([w for w in row['Abstract_Cleaned'] if w not in stop_words]),axis=1)


# In[9]:


# Lemmatization

lmtzr = WordNetLemmatizer()
papers['Abstract_Cleaned'] = papers.apply(lambda row: ([lmtzr.lemmatize(w) for w in row['Abstract_Cleaned']]), axis=1)


# In[10]:


papers['Abstract_Cleaned'][1][:20]


# #### Building LDA Topic Model

# In[11]:


# Creating Dictionary and Corpus

dictionary = corpora.Dictionary(papers['Abstract_Cleaned'])
texts = papers['Abstract_Cleaned']
corpus = [dictionary.doc2bow(text) for text in papers['Abstract_Cleaned']]


# In[12]:


# Building LDA Model

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[13]:


# Printing the Keywords in topics

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[14]:


# Visualizing topics using pyLDAvis

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
vis


# #### Abstracts TF-IDF
# 

# In[15]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

abstracts = papers['Abstract'].values

count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(abstracts)
tfidf_vectorizer = TfidfTransformer().fit(counts)
tfidf_abstracts = tfidf_vectorizer.transform(counts)


# In[16]:


tfidf_abstracts.shape


# #### Affinity Propogation

# In[17]:


#Testing affinity propogation on abstracts tfidf(experimental)

from sklearn.cluster import AffinityPropagation

X = tfidf_abstracts
clustering = AffinityPropagation().fit(X)
clustering 


# In[18]:


abstract_affinity_clusters = list(clustering.labels_)
abstract_affinity_clusters


# In[19]:


len(set(abstract_affinity_clusters))


# Lets try topic modeling again with number of topics equal to clusters found by affinity propogation algorithm

# In[20]:


# Building LDA Model

lda_model_17 = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=17, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[21]:


# Visualizing topics using pyLDAvis

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model_17, corpus, dictionary)
vis


# #### Mapping Top Topics & Abstracts

# But above visualization shows 9 dominant topic clusters

# Lets check paper titles whose abstracts have maximum correspondence with top topic words, took below ideas from https://towardsdatascience.com/improving-the-interpretation-of-topic-models-87fd2ee3847d

# In[22]:


from sklearn.decomposition import LatentDirichletAllocation

lda_model = LatentDirichletAllocation(n_topics=9, max_iter=10, learning_method='online', learning_offset=50.,random_state=0).fit(tfidf_abstracts)
lda_W = lda_model.transform(tfidf_abstracts)
lda_H = lda_model.components_
#Not sure how to find lda_W & lda_H using gensim lda model


# In[23]:


def display_topics(H, W, feature_names, title_list, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print('\n',"Topic %d:" % (topic_idx))
        print("Top Words: "," ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print(title_list[doc_index])
            
no_top_words = 15
no_top_documents = 4     
title_list = papers['Title'].tolist()
tf_feature_names = count_vectorizer.get_feature_names()
display_topics(lda_H, lda_W, tf_feature_names, title_list, no_top_words, no_top_documents)


# In[24]:


from sklearn.decomposition import NMF

nmf_model = NMF(n_components=9, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf_abstracts)
nmf_W = nmf_model.transform(tfidf_abstracts)
nmf_H = nmf_model.components_


# In[25]:


display_topics(nmf_H, nmf_W, tf_feature_names, title_list, no_top_words, no_top_documents)


# #### Visualizing Similarity among Abstracts

# In[26]:


from sklearn.metrics.pairwise import cosine_similarity, paired_cosine_distances


# In[27]:


tfidf_abstracts_sim = []

for i in range(0,tfidf_abstracts.shape[0]):
    sim_array = cosine_similarity(tfidf_abstracts,tfidf_abstracts[i])
    sim_array = sim_array.flatten()
    tfidf_abstracts_sim.append(sim_array)
    
tfidf_abstracts_sim[0]


# In[28]:


tfidf_abstracts_sim_matrix =  np.array(tfidf_abstracts_sim)
tfidf_abstracts_sim_matrix.shape


# In[29]:


import seaborn as sns

sns.set(rc={'figure.figsize':(18.7,15.27)})
sns.heatmap(tfidf_abstracts_sim_matrix)


# Testing a new semi-supervised learning approach to overcome limitations of Topic Models. This approach was published on medium blog: https://medium.com/pew-research-center-decoded/overcoming-the-limitations-of-topic-models-with-a-semi-supervised-approach-b947374e0455
# 
# This approach is based on ideas from paper "Anchored Correlation Explanation: Topic Modeling with Minimal Domain Knowledge" which were implemented in open source project "corextopic". 

# In[30]:


get_ipython().system('pip install corextopic')


# In[31]:


from corextopic import corextopic as ct


# In[32]:


anchors = []
model = ct.Corex(n_hidden=5, seed=36)
model = model.fit(
    tfidf_abstracts,
    words=tf_feature_names
)


# In[33]:


for i, topic_ngrams in enumerate(model.get_topics(n_words=10)):
    topic_ngrams = [ngram[0] for ngram in topic_ngrams if ngram[1] > 0]
    print("Topic #{}: {}".format(i+1, ", ".join(topic_ngrams)))


# In[34]:


# Anchors designed to nudge the model towards measuring specific genres
anchors = [
    ["adversarial","anomaly"],
    ["entity extraction","ner"],
    ["fake","news"],
    ["bias","dimensionality"],
    ["toxic","attack"]
]
anchors = [
    [a for a in topic if a in tf_feature_names]
    for topic in anchors
]

model = ct.Corex(n_hidden=5, seed=40)
model = model.fit(
    tfidf_abstracts,
    words=tf_feature_names,
    anchors=anchors, # Pass the anchors in here
    anchor_strength=3 # Tell the model how much it should rely on the anchors
)


# In[35]:


for i, topic_ngrams in enumerate(model.get_topics(n_words=10)):
    topic_ngrams = [ngram[0] for ngram in topic_ngrams if ngram[1] > 0]
    print("Topic #{}: {}".format(i+1, ", ".join(topic_ngrams)))

