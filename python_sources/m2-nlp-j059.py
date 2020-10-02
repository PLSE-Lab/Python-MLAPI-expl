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
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;


# In[ ]:


#From sklearn
import scipy as sp;
import sklearn;
import sys;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#From nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


# In[ ]:


get_ipython().system('pip install transformers')


# In[ ]:


a=pd.read_csv("/kaggle/input/10k-german-news-articles/Annotations.csv")
b=pd.read_csv("/kaggle/input/10k-german-news-articles/Posts.csv")
c=pd.read_csv("/kaggle/input/10k-german-news-articles/Categories.csv")
d=pd.read_csv("/kaggle/input/10k-german-news-articles/Newspaper_Staff.csv")
e=pd.read_csv("/kaggle/input/10k-german-news-articles/Articles.csv")
f=pd.read_csv("/kaggle/input/10k-german-news-articles/CrossValSplit.csv")
g=pd.read_csv("/kaggle/input/10k-german-news-articles/Annotations_consolidated.csv")


# In[ ]:


a


# In[ ]:


b


# In[ ]:


c


# In[ ]:


d


# In[ ]:


e


# In[ ]:


f


# In[ ]:


g


# In[ ]:


ag = a.merge(g, on='ID_Post')


# In[ ]:


ag


# In[ ]:


be=b.merge(e, on='ID_Article')
be


# In[ ]:


be.isna().sum()


# In[ ]:


be.info()


# In[ ]:


be.dropna(inplace = True)


# In[ ]:


#taking only 5000 values
be=be.iloc[0:5000]


# In[ ]:


#Removing HTML tags
import re
tags = re.compile(r'<[^>]+>')
def html_del(sent):
    t1 = tags.sub('', sent)
    return t1


# In[ ]:


#pipelining
from nltk.tokenize.treebank import TreebankWordDetokenizer
def pipeline_ger(text):
    text=re.sub('[^a-zA-Z]', " ", str(text))
    text=text.lower()
    stop_words_ge = set(stopwords.words('german'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words_ge]
    text = TreebankWordDetokenizer().detokenize(filtered_sentence)
    return text


# In[ ]:


#applying pipline
df = pd.DataFrame(be, columns = ['ID_Post', 'ID_Article','Headline','Body_x','Title','Body_y'])
df["Title"] = df['Title'].apply(pipeline_ger)
df["Headline"] = df['Headline'].apply(pipeline_ger)
df["Body_x"] = df['Body_x'].apply(pipeline_ger)
df['Body_y'] = df['Body_y'].apply(html_del)
df


# In[ ]:


df_content = pd.DataFrame(df, columns = ['Body_x','Body_y','Title'])
df_content#body_x=posts,body_y=articles


# In[ ]:


df_content.info()
#completed till here


# In[ ]:


lemmat = WordNetLemmatizer()
vect = TfidfVectorizer()


# In[ ]:


model_contentt =  vect.fit_transform(df_content['Title'])
model_contentt = pd.DataFrame(model_contentt.toarray(), columns=vect.get_feature_names())
nmf_model_contentt = NMF(n_components=10, init='random', random_state=0)
W_contentt = nmf_model_contentt.fit_transform(model_contentt)
H_contentt = nmf_model_contentt.components_


# In[ ]:


model_contentx =  vect.fit_transform(df_content['Body_x'])
model_contentx = pd.DataFrame(model_contentx.toarray(), columns=vect.get_feature_names())
nmf_model_contentx = NMF(n_components=10, init='random', random_state=0)
W_contentx = nmf_model_contentx.fit_transform(model_contentx)
H_contentx = nmf_model_contentx.components_


# In[ ]:


model_contenty =  vect.fit_transform(df_content['Body_y'])
model_contenty = pd.DataFrame(model_contenty.toarray(), columns=vect.get_feature_names())
nmf_model_contenty = NMF(n_components=10, init='random', random_state=0)
W_contenty = nmf_model_contenty.fit_transform(model_contenty)
H_contenty = nmf_model_contenty.components_


# In[ ]:


lda=LatentDirichletAllocation(n_components=10)


# In[ ]:


ldamodel_contentt=lda.fit_transform(model_contentt)
ldacomp_contentt=lda.components_
ldacomp_contentt


# In[ ]:


ldamodel_contentx=lda.fit_transform(model_contentx)
ldacomp_contentx=lda.components_
ldacomp_contentx


# In[ ]:


ldamodel_contenty=lda.fit_transform(model_contenty)
ldacomp_contenty=lda.components_
ldacomp_contenty


# In[ ]:


def display_topics(H, W, feature_names, no_top_words, no_top_documents):
   for topic_idx, topic in enumerate(H):
       print("Topic %d:" % (topic_idx))
       print(" ".join([feature_names[i]
                       for i in topic.argsort()[:-no_top_words - 1:-1]]))
       top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]


# In[ ]:


display_topics(H_contentt, W_contentt, vect.get_feature_names(), 7,7)


# In[ ]:


display_topics(H_contentx, W_contentx, vect.get_feature_names(), 7,7)


# In[ ]:


display_topics(H_contenty, W_contenty, vect.get_feature_names(), 7,7)


# In[ ]:


def print_topics(model, vect, n_top_words):
    words = vect.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


# In[ ]:


print_topics(lda,vect,10)

