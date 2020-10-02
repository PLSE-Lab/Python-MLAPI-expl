#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
#print(os.listdir("../input"))
data = pd.read_csv('../input/train.csv')
data.shape


# In[7]:


#data[data['target']==1].head()
data['target'].value_counts() / data.shape[0] * 100


# In[8]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
insincere_rows = data[data['target'] == 1]
wc = WordCloud(background_color='white').generate(' '.join(insincere_rows['question_text']))
plt.imshow(wc)


# In[9]:


from sklearn.model_selection import train_test_split
train, validate = train_test_split(data, test_size=0.3, random_state=1)
train.shape, validate.shape


# In[12]:


import nltk
def clean_sentence(doc, stopwords, stemmer):
    words = doc.split(' ')
    words_clean = [stemmer.stem(word) for word in words if word not in stopwords]
    return ' '.join(words_clean)
    
def clean_documents(docs_raw):
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = nltk.stem.PorterStemmer()
    docs = docs_raw.str.lower().str.replace('[^a-z ]', '')
    docs_clean = docs.apply(lambda doc: clean_sentence(doc, stopwords, stemmer))
    return docs_clean
train_docs_clean = clean_documents(train['question_text'])
train_docs_clean.head()


# In[15]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=10).fit(train_docs_clean)
dtm = vectorizer.transform(train_docs_clean)


# In[18]:


from sklearn.tree import DecisionTreeClassifier
model_df = DecisionTreeClassifier(max_depth=10).fit(dtm, train['target'])


# In[19]:


validate_docs_clean = clean_documents(validate['question_text'])
dtm_validate = vectorizer.transform(validate_docs_clean)
dtm_validate


# In[20]:


validate_pred = model_df.predict(dtm_validate)
from sklearn.metrics import f1_score
f1_score(validate['target'], validate_pred)


# In[21]:


from sklearn.naive_bayes import MultinomialNB
model_nb = MultinomialNB().fit(dtm, train['target'])
validate_pred = model_nb.predict(dtm_validate)
f1_score(validate['target'], validate_pred)


# In[23]:


test = pd.read_csv('../input/test.csv')
docs_clean = clean_documents(test['question_text'])
dtm_test = vectorizer.transform(docs_clean)
dtm_test


# In[24]:


test_pred = model_nb.predict(dtm_test)


# In[26]:


sample_submission = pd.read_csv('../input/sample_submission.csv')
submission = pd.DataFrame({'qid': test['qid'],
                           'prediction': test_pred})
submission[['qid', 'prediction']].to_csv('submission.csv', index=False)

