#!/usr/bin/env python
# coding: utf-8

# # BERT + BM25 News Search Engine
# 
# * BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document, regardless of their proximity within the document. 
# 
# * Semantic search is a use case for BERT where pre-trained word vectors can be used as is, without any fine tuning.
# * In this notebook, we ensemble the traditional function BM25 with BERT semantic search by taking the average.

# ## Import data

# In[ ]:


import pandas as pd 
news = pd.read_csv('/kaggle/input/cbc-news-coronavirus-articles-march-26/news.csv')
news = news.drop(['Unnamed: 0'], axis=1)
news.sample(3)


# ## BM25

# In[ ]:


get_ipython().system('pip install rank_bm25')


# ### Parse sentences

# In[ ]:


import nltk.data
corpus = []
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
for article in news.text.apply(lambda row : row.lower()):
    corpus.extend(tokenizer.tokenize(article))


# ### Set simple grammar

# In[ ]:


corpus = list(set([sentence for sentence in corpus if len(sentence) > 50]))


# ### Tokenize documents

# In[ ]:


from rank_bm25 import BM25Okapi
tokenized_docs = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_docs)


# ### Enter query

# In[ ]:


query = "trudeau"
tokenized_query = query.split(" ")
doc_scores = bm25.get_scores(tokenized_query)


# ### Print the top 10 related sentences

# In[ ]:


bm25.get_top_n(tokenized_query, corpus, n=10)


# ## Using BERT

# In[ ]:


get_ipython().system('pip install -U sentence-transformers')


# ### Print the top 10 related sentences

# In[ ]:


from sentence_transformers import SentenceTransformer
import scipy.spatial

embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# Corpus with example sentences
corpus = corpus
corpus_embeddings = embedder.encode(corpus)

# Query sentences:
queries = ['trudeau']
query_embeddings = embedder.encode(queries)

closest_n = 10
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nThe most similar sentences in corpus:")

    for idx, distance in results[0:closest_n]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))


# ## Ensemble

# ### Calculate sum

# In[ ]:


ensemble = [a + b for a, b in zip(distances, doc_scores)]


# ### Print the adjusted top 10 sentences
# Compared to BM25, the ordering has changed. 

# In[ ]:


for i in range(10):
    print(corpus[ensemble.index(sorted(ensemble, reverse=True)[i])])

