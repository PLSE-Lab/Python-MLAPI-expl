#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


get_ipython().system('pip install biobert-embedding')


# In[ ]:


from biobert_embedding.embedding import BiobertEmbedding
#from sentence_transformers import SentenceTransformer
model = BiobertEmbedding()
#model = SentenceTransformer('bert-base-nli-mean-tokens')


# In[ ]:


sentences = list(df.title.dropna())


# In[ ]:


sentence_embeddings=[]
for i in range(len(sentences)):
    sentence_embeddings.append(model.sentence_vector(sentences[i]).numpy())
#sentence_embeddings = model.encode(sentences)


# In[ ]:


# Import required packages
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[ ]:


mms = MinMaxScaler()
mms.fit(sentence_embeddings)
data_transformed = mms.transform(sentence_embeddings)


# In[ ]:


Sum_of_squared_distances = []
K = range(1,25)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)


# In[ ]:


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=10, random_state=0).fit(data_transformed)
label = kmeans.labels_
clusters = {'Title': sentences, 'Label': label}
dfc = pd.DataFrame(clusters)
dfc.head()


# In[ ]:


label_freq = {i:list(dfc.Label).count(i) for i in set(list(dfc.Label))}
print(label_freq)


# In[ ]:


dfc.to_csv('title_clusters_BioBERT.csv')
#dfc.to_csv('title_clusters.csv')


# In[ ]:


import seaborn as sns
sns.countplot(dfc.Label)


# In[ ]:


from langdetect import detect
dfc['is_en']=dfc['Title'].apply(lambda x : detect(x)=='en')
sns.countplot(dfc.is_en)


# In[ ]:


import re
from textblob import Word
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from nltk import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
import string, collections
from nltk.util import ngrams
from gensim.parsing.preprocessing import STOPWORDS


# In[ ]:


def freq_bigrams(x, terms):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    esBigrams = ngrams(all_words, 2)
    esBigramFreq = collections.Counter(esBigrams)
    bigrams = esBigramFreq.most_common(500) 
    tb=[]
    for bi in bigrams:
        if len([i for i in bi[0] if i not in STOPWORDS])==2:
            tb.append(bi)
    d = pd.DataFrame({'bigram':[x[0] for x in tb[:terms]], 'count':[x[1] for x in tb[:terms]]})
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "bigram", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()


# In[ ]:


print('Overall distribution :' )
freq_bigrams(dfc.Title, 10)
for i in range(10):
    print('Class ', i,  ' distribution :' )
    freq_bigrams(dfc[dfc.Label==i].Title, 10)


# In[ ]:


dfc['Title'] = dfc['Title'].str.lower()
dfc['Title'] = dfc['Title'].apply(lambda x : re.sub(r'\.+', ". ", x))
dfc['Title'] = dfc['Title'].str.replace("[^a-zA-Z#]", " ")
dfc['Title'] = dfc['Title'].apply(lambda x: remove_stopwords(x))
dfc['Title'] = dfc['Title'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
dfc['Title'] = dfc['Title'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# In[ ]:


def freq_words(x, terms):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    
    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
    # selecting top N most frequent words
    d = words_df.nlargest(columns="count", n = terms) 
    #print(d)
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()


# In[ ]:


#Distribution of most frequent words in the 'Titles' data-set
print('Overall distribution :' )
freq_words(dfc.Title, 20)
for i in range(10):
    print('Class ', i,  ' distribution :' )
    freq_words(dfc[dfc.Label==i].Title, 20)


# In[ ]:




