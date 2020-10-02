#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
import gensim.corpora as corpora
from gensim.models import coherencemodel
from gensim.utils import simple_preprocess
import nltk


# In[ ]:


file = open('../input/sport.txt').read()


# In[ ]:


file


# In[ ]:


def get_summary(text,pct):
    return summarize(text,ratio=pct,split=True)


# In[ ]:


def get_keywords(text,lemme):
    res = keywords(text, ratio=0.1,
                   words=None,
                   split=False,
                   scores=False,
                   pos_filter=('NN', 'JJ'),
                   lemmatize=lemme,
                   deacc=False)
    res = res.split('\n')
    return res


# In[ ]:


get_summary(file,0.3)


# In[ ]:


get_keywords(file,True)


# In[ ]:




