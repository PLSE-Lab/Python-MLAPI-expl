#!/usr/bin/env python
# coding: utf-8

# # Simple topic modeling with NMF/LDA
# 
# <img src="https://i.imgur.com/KrLVpBQ.png">
# 
# # Table of contents
# 
# [<h3>1. Presentation of the data</h3>](#1)
# 
# [<h3>2. Topic modeling with NMF</h3>](#2)
# 
# [<h3>3. Topic modeling with LDA</h3>](#3)

# # 1. Presentation of the data<a class="anchor" id="1"></a>
# 
# Around 42.000 Tweets from Donald Trump between 2009 and 2020.
# 
# <strong><u>Data Content: </u></strong><br>
# <br><br>- <strong>id: </strong> Unique tweet id
# <br><br>- <strong>link: </strong>Link to tweet
# <br><br>- <strong>content: </strong>Text of tweet
# <br><br>- <strong>date: </strong>Date of tweet
# <br><br>- <strong>retweets: </strong>Number of retweets
# <br><br>- <strong>favorites: </strong>Number of favorites
# <br><br>- <strong>mentions: </strong>Accounts mentioned in tweet

# In[ ]:


import pandas as pd
df = pd.read_csv("/kaggle/input/trump-tweets/realdonaldtrump.csv")
df.head(3)


# In[ ]:


print(f"Number of tweets: {df.shape[0]}")


# # 2. Topic modeling with NMF<a class="anchor" id="2"></a>

# In[ ]:


# Add a few words to the stop words to avoid websites
from nltk.corpus import stopwords
stop_words = stopwords.words("english")+["http","https","www", "com"]

# Convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stop_words, ngram_range=(1,2))
tm = tfidf.fit_transform(df["content"])


# In[ ]:


# Non-Negative Matrix Factorization (NMF)
# Find two non-negative matrices (W, H) whose product 
# approximates the non- negative matrix X.
from sklearn.decomposition import NMF
nmf = NMF(n_components=10, random_state=0)

# fit the transfomed content with NMF
nmf.fit(tm)

# display the result
for index,topic in enumerate(nmf.components_):
    print(f"The top 20 words for topic # {index}")
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-20:]])
    print("\n")


# # 3. Topic modeling with LDA<a class="anchor" id="3"></a>
# 
# In natural language processing, the latent Dirichlet allocation (LDA) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. For example, if observations are words collected into documents, it posits that each document is a mixture of a small number of topics and that each word's presence is attributable to one of the document's topics. LDA is an example of a topic model and belongs to the machine learning toolbox and in wider sense to the artificial intelligence toolbox. (<a href="https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation">source</a>)

# In[ ]:


# Add a few words to the stop words to avoid websites
from nltk.corpus import stopwords
stop_words = stopwords.words("english")+["http","https","www", "com"]

# Convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stop_words, ngram_range=(1,2))
tm = tfidf.fit_transform(df["content"])


# In[ ]:


from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components = 10, n_jobs = -1, random_state = 0)

# fit the transfomed content with LDA
LDA.fit(tm)

# display the result
for index,topic in enumerate(LDA.components_):
    print(f"The top 20 words for topic # {index}")
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-20:]])
    print("\n")


# Thank you
