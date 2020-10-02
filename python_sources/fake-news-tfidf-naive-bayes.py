#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import nltk as nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# In[ ]:


df=pd.read_csv('../input/fake_or_real_news.csv')


# In[ ]:


df.head()


# In[ ]:


train, test = train_test_split(df, test_size = 0.2)
train.columns.values
train.head()


# In[ ]:


def refineWords(s):
    letters_only = re.sub("[^a-zA-Z]", " ", s) 
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    print( " ".join( meaningful_words ))
    return( " ".join( meaningful_words ))

train["title"] = train["title"].apply(refineWords)
train["text"] = train["text"].apply(refineWords)

train_two = train.copy()
train.head()


# In[ ]:


test["title"] = test["title"].apply(refineWords)
#PB added this line
test["text"] = test["text"].apply(refineWords)

test_two = test.copy()
test.head()


# In[ ]:


XTrain = train['text']
YTrain = train['label']

XTrain.head() 


# In[ ]:


YTrain.head()


# In[ ]:


vectorizer = CountVectorizer().fit(XTrain)
XTrain_vectorized = vectorizer.transform(XTrain)

print('Vocabulary len:', len(vectorizer.get_feature_names()))
print('Longest word:', max(vectorizer.vocabulary_, key=len))


# In[ ]:


transformer = TfidfTransformer(smooth_idf = False)
tfidf = transformer.fit_transform(XTrain_vectorized)


# In[ ]:


model = MultinomialNB(alpha=0.1)
model.fit(XTrain_vectorized, YTrain)


# In[ ]:


XTest = test['text']
YTest = test['label']


# In[ ]:


YPred = model.predict(vectorizer.transform(XTest))
print('Accuracy: %.2f%%' % (accuracy_score(YTest, YPred) * 100))

