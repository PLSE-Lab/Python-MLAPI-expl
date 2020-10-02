#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import time
import re
import string
import nltk
from nltk.corpus import stopwords

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', "def clean_text(text):\n    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation\n    and remove words containing numbers.'''\n    text = text.lower()\n    text = re.sub('\\[.*?\\]', '', text)\n    text = re.sub('https?://\\S+|www\\.\\S+', '', text)\n    text = re.sub('<.*?>+', '', text)\n    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)\n    text = re.sub('\\n', '', text)\n    text = re.sub('\\w*\\d\\w*', '', text)\n    return text\n\n# Applying the cleaning function to both test and training datasets\ntrain['text'] = train['text'].apply(lambda x: clean_text(x))\ntest['text'] = test['text'].apply(lambda x: clean_text(x))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "tokenizer = nltk.tokenize.RegexpTokenizer(r'\\w+')\ntrain['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))\ntest['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'def remove_stopwords(text):\n    """\n    Removing stopwords belonging to english language\n    \n    """\n    words = [w for w in text if w not in stopwords.words(\'english\')]\n    return words\n\n\ntrain[\'text\'] = train[\'text\'].apply(lambda x : remove_stopwords(x))\ntest[\'text\'] = test[\'text\'].apply(lambda x : remove_stopwords(x))')


# In[ ]:


get_ipython().run_cell_magic('time', '', "def combine_text(list_of_text):\n    combined_text = ' '.join(list_of_text)\n    return combined_text\n\ntrain['text'] = train['text'].apply(lambda x : combine_text(x))\ntest['text'] = test['text'].apply(lambda x : combine_text(x))")


# In[ ]:


lemmatizer=nltk.stem.WordNetLemmatizer()

train['text'] = train['text'].apply(lambda x: lemmatizer.lemmatize(x))
test['text'] = test['text'].apply(lambda x: lemmatizer.lemmatize(x))


# In[ ]:


X_train = train['text']
y_train = train['target']


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 


# In[ ]:


vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 2), norm='l2')

svc_clf = Pipeline([('tfidf', vectorizer),
                      ('svc_clf', LinearSVC())])


lr_clf = Pipeline([('tfidf', vectorizer),
                      ('lr_clf', LogisticRegression())])


# In[ ]:


svc_clf.fit(X_train,y_train)
lr_clf.fit(X_train,y_train)


# In[ ]:


test


# In[ ]:


svc_pred = svc_clf.predict(test['text'])
lr_pred = lr_clf.predict(test['text'])


# In[ ]:


submission_svc_pred=pd.DataFrame({"id":sub['id'],"target":svc_pred})
submission_lr_pred=pd.DataFrame({"id":sub['id'],"target":lr_pred})


# In[ ]:


submission_svc_pred.to_csv("submission_svc.csv",index=False)
submission_lr_pred.to_csv("submission_lr.csv",index=False)

