#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_curve, confusion_matrix, roc_auc_score, auc, recall_score, precision_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from scipy.stats import logistic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LassoCV, Lasso, ElasticNet, LassoLars
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_curve, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import Counter
from itertools import tee, islice
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
from nltk.corpus import stopwords
import string
import re
import pickle

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')


# In[4]:


news = pd.read_json('../input/News_Category_Dataset_v2.json', lines=True)


# In[5]:


news.shape


# In[6]:


news.category.value_counts()


# In[7]:


news.category[news.category=='THE WORLDPOST'] = 'WORLDPOST'
news.category[news.category=='GREEN'] = 'ENVIRONMENT'
news.category[news.category=='CULTURE & ARTS'] = 'ARTS'
news.category[news.category=='COMEDY'] = 'ENTERTAINMENT'
news.category[(news.category=='BLACK VOICES') | (news.category=='LATINO VOICES') | (news.category=='QUEER VOICES')] = 'VOICES'
news.category[news.category=='STYLE'] = 'STYLE & BEAUTY'
news.category[news.category=='ARTS & CULTURE'] = 'ARTS'
news.category[news.category=='COLLEGE'] = 'EDUCATION'
news.category[news.category=='SCIENCE'] = 'TECH'
news.category[news.category=='WEDDINGS'] = 'GOOD NEWS'
news.category[news.category=='TASTE'] = 'FOOD & DRINK'
news.category[(news.category=='PARENTING') | (news.category=='FIFTY')] = 'PARENTS'
news.category[news.category=='WORLD NEWS'] = 'WORLDPOST'


# In[8]:


news.category.value_counts()


# In[9]:


news.head()


# In[10]:


news.category.nunique()


# In[11]:


news.authors.nunique()


# In[12]:


news[news.authors == 'Suzy Strutner'].category.value_counts()


# In[13]:


def textClean(text):
    text = re.sub("[^a-zA-Z]", ' ',text).strip()
    text = nltk.word_tokenize(text.lower())
    stops = set(stopwords.words("english"))
    stops = list(stops) + ['amp']
    text = " ".join([lemmatizer.lemmatize(w) for w in text if w not in stops])
    return(text)


# In[14]:


get_ipython().run_cell_magic('time', '', "news['Clean_headline'] = news.headline.map(textClean)\nnews['Clean_description'] = news.short_description.map(textClean)")


# In[15]:


y = news.category
train, test, y_train, y_test = train_test_split(news, y, test_size=0.2, random_state=0, stratify= y)


# In[16]:


tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df = 5)

tfidf_train = tfidf_vectorizer.fit_transform(train['Clean_headline'])
tfidf_test = tfidf_vectorizer.transform(test['Clean_headline'])


# In[17]:


get_ipython().run_cell_magic('time', '', "from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score\n\nLR = LogisticRegression('l1')\nLR.fit(tfidf_train, y_train)\ny_predlr=LR.predict(tfidf_test)\nprint('LR test: ',accuracy_score(y_test, y_predlr))\nprint('LR train:',accuracy_score(y_train, LR.predict(tfidf_train)))\nprint('KAPPA SCORE: ',cohen_kappa_score(y_test,y_predlr))")


# In[18]:


from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(tfidf_train, y_train)
y_pred = nb.predict(tfidf_test)
print('LR test: ',accuracy_score(y_test, y_pred))
print('LR train:',accuracy_score(y_train, nb.predict(tfidf_train)))
print('KAPPA SCORE: ',cohen_kappa_score(y_test,y_pred))


# In[ ]:




