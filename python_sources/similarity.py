#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[ ]:


from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import multiprocessing

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import xlrd
# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import bigrams


from time import time  # To time our operations
from collections import defaultdict  # For word frequency

import spacy  # For preprocessing


# In[ ]:


df = pd.read_csv('../input/examplesss/examples.csv')
df


# In[ ]:


df.isnull().sum()


# In[ ]:


df = df.dropna().reset_index(drop=True)
df.isnull().sum()


# In[ ]:


import pickle

with open("../input/pickledglove300d22mforkernelcompetitions/glove.2M.840B.300d.pkl","rb") as f:
    embeddings_dict_glove = pickle.load(f)
    
print(len(embeddings_dict_glove))


# In[ ]:


all_vec = []
for sentence in df['pulse_questions']:
    sentence_vec = np.zeros((300,))
    number = 0
    for word in sentence.split():
        if embeddings_dict_glove.get(word) is not None:
            sentence_vec = sentence_vec + embeddings_dict_glove.get(word)
            number = number + 1
    
    sentence_vec = sentence_vec/number
    all_vec.append(sentence_vec)
    
all_vec = np.array(all_vec)
print(all_vec.shape)


# In[ ]:


# saving the file
pd.DataFrame(all_vec).to_csv("sentence_vectors.csv",index=False)
print("Done writing the file")

