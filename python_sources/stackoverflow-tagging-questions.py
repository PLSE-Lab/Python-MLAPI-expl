#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

import warnings

import pickle
import time

import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.cluster import KMeans


import logging

from scipy.sparse import hstack

warnings.filterwarnings("ignore")
plt.style.use('bmh')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[58]:


np.random.seed(seed=11)


# In[59]:


import os 
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/Questions.csv", encoding="ISO-8859-1")
df.head()


# In[ ]:


tags = pd.read_csv("../input/Tags.csv", encoding="ISO-8859-1", dtype={'Tag': str})
tags.head()


# In[ ]:


df.info()


# In[ ]:


tags.info()


# In[ ]:


tags['Tag'] = tags['Tag'].astype(str)


# In[ ]:


grouped_tags = tags.groupby("Id")["Tag"].apply(lambda tags: ' '.join(tags))


# In[ ]:


grouped_tags.head()


# In[ ]:


grouped_tags.reset_index()


# In[ ]:


grouped_tags_final = pd.DataFrame({'Id':grouped_tags.index, 'Tags':grouped_tags.values})


# In[ ]:


grouped_tags_final.head(5)


# In[ ]:


df.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate'], inplace=True)


# In[ ]:


df = df.merge(grouped_tags_final, on='Id')


# In[ ]:


df.head(5)


# Keeping Data Only With score > 5 because --> 
# 1. Less resources will be needed 
# 2. Quality of data is better with higher score 

# In[ ]:


new_df = df[df['Score']>5]


# In[ ]:


new_df.head()


# In[ ]:


new_df.isnull().sum()
# No missing value present 


# In[ ]:


print('Dupplicate entries: {}'.format(new_df.duplicated().sum()))
# No Duplicates Present 


# In[ ]:


new_df.drop(columns=['Id', 'Score'], inplace=True)


# It is a good idea to keep only a few important tags because User can manually add more tags later. Our main purpose is to recommend articles. 

# In[ ]:


new_df.head()


# In[ ]:


new_df['Tags'] = new_df['Tags'].apply(lambda x: x.split())


# In[ ]:


new_df['Tags']


# In[ ]:


all_tags = [item for sublist in new_df['Tags'].values for item in sublist]


# In[ ]:


len(all_tags)


# In[ ]:


my_set = set(all_tags)
unique_tags = list(my_set)
len(unique_tags)

