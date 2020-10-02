#!/usr/bin/env python
# coding: utf-8

# # Nafisur Rahman
# nafisur21@gmail.com<br>
# https://www.linkedin.com/in/nafisur-rahman

# # Clustering Problem in NLP
# * Identifying the genre of a book
# * Recognizing the themes or topics in an article
# * Topic modeling
# * Clustering text Data
# * Modeling Text Topics

# ## Topic Modeling in NLP

# ## Clustering text data (Article) using Kmeans algorithm

# ## Recognizing the themes or topics in an article

# ### Part 2:- Clustering
# In part1 we have saved all the article from given blogpost (http://doxydonkey.blogspot.in/) into a tab seperated file called "allposts.csv".<br>
# In this part, we are loading "allposts.csv" file into pandas dataframe and doing cluster analysis. 

# Loading the libraries

# In[1]:


import pandas as pd
import nltk


# Loading the dataset

# In[2]:


dataset=pd.read_csv('../input/allposts.csv',sep='\t',quoting=3)


# In[3]:


dataset.head()


# In[4]:


df=dataset[['post']]
df.head()


# In[5]:


len(df)


# In[6]:


df.info()


# Removing the missing values

# In[7]:


df=df.dropna()
len(df)


# Removing column without any text

# In[8]:


df[df['post']=='"'].head()


# In[9]:


l=df[df['post']=='"'].index


# In[10]:


df=df.drop(labels=l)
len(df)


# Resetting the index

# In[11]:


df = df.reset_index(drop=True)
len(df)


# In[12]:


df1=df
df.head()


# #### NLP Preprocessing task

# In[13]:


from nltk.corpus import stopwords
from string import punctuation
import re
from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
ls=WordNetLemmatizer()
cstopwords=set(stopwords.words('english')+list(punctuation))


# In[14]:


text_corpus=[]
for i in range(0,len(df)):
    review=re.sub('[^a-zA-Z]',' ',df['post'][i])
    #review=df['post'][i]
    review=[ls.lemmatize(w) for w in word_tokenize(str(review).lower()) if w not in cstopwords]
    review=' '.join(review)
    text_corpus.append(review)
    
len(text_corpus)


# ### NLP Features extraction

# In[15]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[16]:


cv=CountVectorizer()


# In[17]:


X1=cv.fit_transform(text_corpus).toarray()


# In[18]:


X1.shape


# In[19]:


tfidfvec=TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english')
X2=tfidfvec.fit_transform(text_corpus).toarray()
X2.shape


# ### Kmean Clustering Algorithm

# In[20]:


from sklearn.cluster import KMeans
from nltk.probability import FreqDist


# #### Finding number of cluster

# wcss=[]
# for i in range(1,10):
#     kmeans=KMeans(n_clusters=i)
#     kmeans.fit(X1)
#     wcss.append(kmeans.inertia_)
#     
# import matplotlib.pyplot as plt
# %matplotlib inline
# plt.plot(range(1,10),wcss)

# In[21]:


from scipy.cluster import hierarchy as hc


# In[22]:


hc.dendrogram(hc.linkage(X1,method='ward'))


# For simplicity we are only taking number of cluster=3

# In[23]:


km=KMeans(n_clusters=3)
km.fit(X1)
km.labels_


# In[24]:


df1['labels']=km.labels_
df1['processed']=text_corpus
df1.head()


# In[25]:


km.n_clusters


# Most frequent words in each cluster

# In[26]:


for i in range(km.n_clusters):
    df2=df1[df['labels']==i]
    df2=df2[['processed']]
    words=word_tokenize(str(list(set([a for b in df2.values.tolist() for a in b]))))
    dist=FreqDist(words)
    print('Cluster :',i)
    print('most common words :',dist.most_common(30))


# Most frequent unique words in each cluster

# In[27]:


text={}
for i,cluster in enumerate(km.labels_):
    oneDocument = df1['processed'][i]
    if cluster not in text.keys():
        text[cluster] = oneDocument
    else:
        text[cluster] += oneDocument


# In[28]:


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import nltk


# In[29]:


_stopwords = set(stopwords.words('english') + list(punctuation)+["million","billion","year","millions","billions","y/y","'s","''","``"])


# In[30]:


keywords = {}
counts={}
for cluster in range(3):
    word_sent = word_tokenize(text[cluster].lower())
    word_sent=[word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)
    keywords[cluster] = nlargest(100, freq, key=freq.get)
    counts[cluster]=freq


# In[31]:


unique_keys={}
for cluster in range(3):   
    other_clusters=list(set(range(3))-set([cluster]))
    keys_other_clusters=set(keywords[other_clusters[0]]).union(set(keywords[other_clusters[1]]))
    unique=set(keywords[cluster])-keys_other_clusters
    unique_keys[cluster]=nlargest(10, unique, key=counts[cluster].get)


# In[32]:


unique_keys


# ### Topics Modeling

# 1. Cluster 0= Related to startup and Fund raising plan
# 2. Cluster 1= Stock performance related
# 3. Cluster 2= Social media and advertising related

# In[ ]:




