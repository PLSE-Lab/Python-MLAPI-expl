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

data=pd.read_csv('../input/train.csv')
data.shape

# Any results you write to the current directory are saved as output.


# In[ ]:


data.head()


# In[ ]:


data[data['target']==1].head()


# In[ ]:


data['target'].value_counts()/data.shape[0]*100  # Highly imbalanced data


# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
insincere_rows=data[data['target']==1]
wc=WordCloud(background_color='white').generate(' '.join(insincere_rows['question_text']))
plt.imshow(wc)


# In[ ]:


from sklearn.model_selection import train_test_split  # Splitting for train and validation
train,validate=train_test_split(data,test_size=0.3,random_state=1)
train.shape,validate.shape


# In[ ]:


import nltk

def clean_sentence(doc, stopwords, stemmer):
    words=doc.split(' ')
    words_clean=[stemmer.stem(word) for word in words if word not in stopwords]
    return ' '.join(words_clean)

def clean_documents(docs_raw):
    stopwords=nltk.corpus.stopwords.words('english')
    stemmer=nltk.stem.PorterStemmer()
    docs=docs_raw.str.lower().str.replace('[^a-z ]','')
    docs_clean=docs.apply(lambda doc: clean_sentence(doc, stopwords, stemmer))
    return docs_clean

train_docs_clean=clean_documents(train['question_text'])
train_docs_clean.head()


# In[ ]:


# Convert to document matrix

from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(min_df=10).fit(train_docs_clean)
dtm=vectorizer.transform(train_docs_clean)


# In[ ]:


dtm


# In[29]:


total_values= 914285*19550             # Sparcity is 99.96%
non_zer_values=5406880
non_zer_values/total_values *100
(total_values-non_zer_values)/total_values*100 


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model_df=DecisionTreeClassifier(max_depth=10).fit(dtm,train['target'])


# In[30]:


validate_docs_clean=clean_documents(validate['question_text'])
dtm_validate=vectorizer.transform(validate_docs_clean)
dtm_validate


# In[31]:


validate_pred=model_df.predict(dtm_validate)
from sklearn.metrics import f1_score
f1_score(validate['target'],validate_pred)


# f1 score for Decision tree is 0.261

# In[32]:


from sklearn.naive_bayes import MultinomialNB
model_nb=MultinomialNB().fit(dtm,train['target'])
validate_pred=model_nb.predict(dtm_validate)
f1_score(validate['target'],validate_pred)


# f1 score for MultinomialNB is 0.542

# In[33]:


test=pd.read_csv('../input/test.csv')
test.head()


# In[34]:


docs_clean=clean_documents(test['question_text'])
dtm_test=vectorizer.transform(docs_clean)
dtm_test


# In[35]:


test_pred=model_nb.predict(dtm_test)


# In[37]:


sample_submission=pd.read_csv('../input/sample_submission.csv')
sample_submission.head()


# In[38]:


submission=pd.DataFrame({'qid':test['qid'],'prediction':test_pred})
submission[['qid','prediction']].to_csv('submission.csv',index=False)


# In[ ]:




