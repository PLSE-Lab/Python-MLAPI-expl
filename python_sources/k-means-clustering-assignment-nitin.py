#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
np.random.seed(2018)
import nltk
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/department-of-justice-20092018-press-releases/"))
data=os.listdir("../input/20-newsgroup-original/20_newsgroup/20_newsgroup/soc.religion.christian/")
# Any results you write to the current directory are saved as output.


# In[ ]:


TEXT_DATA_DIR = r'../input/20-newsgroup-original/20_newsgroup/20_newsgroup/'
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)


# In[ ]:


print('Original Topics-----------------------------------------------------')
for index,ele in labels_index.items():
    print(ele," ",index)


# In[ ]:


def remove(list1): 
    pattern = '[_:.)><@!(\\n\[\]/0-9]'
    list1 = [re.sub(pattern, ' ', i) for i in list1] 
    list1=[re.sub(r'\b\w{1,3}\b', ' ',i) for i in list1]
    return list1
texts=remove(texts)


# ### K means clustering

# In[ ]:


tfidf = TfidfVectorizer(
    min_df = 0.002,
    max_df = 0.1,
    max_features = 10000,
    stop_words = 'english',
)
tfidf.fit(texts)
text = tfidf.transform(texts)
df = pd.DataFrame(text.toarray(),columns=[tfidf.get_feature_names()])
df.iloc[:5]


# In[ ]:


#Finding the optimal cluster through Elbow method
def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    
find_optimal_clusters(text, 20)


# In[ ]:


#To see the sum of squared error for given cluster size
MiniBatchKMeans(n_clusters=20, init_size=1024, batch_size=2048, random_state=20).fit(text).inertia_


# In[ ]:


clusters = MiniBatchKMeans(n_clusters=20, init_size=1024, batch_size=2048, random_state=20).fit_predict(text)


# In[ ]:


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    topic_df=pd.DataFrame()
    for i,r in df.iterrows():
        print('\nTopic {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
        return topic_df[i]=[labels[t] for t in np.argsort(r)[-n_terms:]]    
get_top_keywords(text, clusters, tfidf.get_feature_names(), 10)

#To join the clusters in original data


# In[ ]:


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    topic_df=pd.DataFrame()
    for i,r in df.iterrows():
        topic_df[i]=[labels[t] for t in np.argsort(r)[-n_terms:]]    
    return topic_df
df1=(get_top_keywords(text, clusters, tfidf.get_feature_names(), 10))


# In[ ]:


print('topics from K-Means')
df1


# **LDA**

# In[ ]:


stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in texts] 


# In[ ]:


# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


# In[ ]:


# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=20, id2word = dictionary, passes=20)


# In[ ]:


def get_lda_topics(model, num_topics):
    word_dict = {}
    for i in range(num_topics):
        words = model.show_topic(i, topn = 10)
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words]
    return pd.DataFrame(word_dict)


# In[ ]:


get_lda_topics(ldamodel, 20)


# In[ ]:


n_samples = 2000
n_features = 8000
n_components = 20
n_top_words = 10


def print_top_words(model, feature_names, n_top_words):
    topic_df=pd.DataFrame()
    for topic_idx, topic in enumerate(model.components_):
#         message = "Topic #%d: " % topic_idx
#         message += " ".join([feature_names[i]
#                              for i in topic.argsort()[:-n_top_words - 1:-1]])
#         print(message)
        topic_df[topic_idx]=[feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return topic_df
    print()


# In[ ]:


# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')

tfidf = tfidf_vectorizer.fit_transform(texts)


# In[ ]:


# Fit the NMF model
nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)

tfidf_feature_names = tfidf_vectorizer.get_feature_names()
df2=print_top_words(nmf, tfidf_feature_names, n_top_words)


# In[ ]:


print("\nTopics in NMF model (Frobenius norm):")
df2

