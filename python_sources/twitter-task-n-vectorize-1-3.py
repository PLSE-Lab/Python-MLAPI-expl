#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))
import re
import logging
import time
import warnings
import gensim
import sys
warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

from bs4 import BeautifulSoup 

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.cluster import KMeans

from gensim.models import Word2Vec 

from nltk.tokenize import WordPunctTokenizer, TweetTokenizer

import nltk
from nltk.corpus import stopwords 


# In[ ]:


train = pd.read_csv('../input/train.csv',encoding='latin1',sep=',')
test = pd.read_csv('../input/test.csv',encoding='latin1',sep=',')


# In[ ]:


stop_words = set(stopwords.words('english'))
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)


# In[ ]:


def stage_one(text):
    text = BeautifulSoup(text).get_text()   
    reg = re.sub("[^a-zA-Z]", " ", text) 
    low = reg.lower().split()  
    meaningful_words = ''
    meaningful_words = tokenizer.tokenize(" ".join( low ))
    return( " ".join( meaningful_words ))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clean_train = []\nfor i in range( 0, train["SentimentText"].size ):                                                          \n    if i % 10000==0:\n        print(i,\'samples is readt from\',train["SentimentText"].size)\n    clean_train.append( stage_one( train["SentimentText"][i] ))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'vectorizer = CountVectorizer(analyzer = "word",   \\\n                             tokenizer = None,    \\\n                             preprocessor = None, \\\n                             stop_words = stop_words,   \\\n                             max_features =1500, \\\n                             ngram_range=(1, 3)) \nfeatures = vectorizer.fit_transform(clean_train).toarray()\n\ndist = np.sum(features, axis=0)\nvocab = vectorizer.get_feature_names()\n    \nforest = ExtraTreesClassifier(n_estimators = 200, n_jobs = -1) \nforest = forest.fit( features, train["Sentiment"] )')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clean_test = [] \nfor i in range(0,len(test["SentimentText"])):\n    if i % 10000==0:\n        print(i,\'samples is readt from\',train["SentimentText"].size)\n    clean_review = stage_one( test["SentimentText"][i] )\n    clean_test.append( clean_review )\n\ntest_features = vectorizer.transform(clean_test)\ntest_features = test_features.toarray()\n\nresult = forest.predict(test_features)')


# In[ ]:


output = pd.DataFrame( data={"ItemID":test["ItemID"], "Sentiment":result} )
output.to_csv("n_vectorize_1_3 submit.csv", index=False, quoting=3 )


# In[ ]:





# In[ ]:




