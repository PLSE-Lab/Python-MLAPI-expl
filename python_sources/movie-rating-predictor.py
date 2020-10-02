#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


d1 = pd.read_csv("../input/Sample_submission.csv")
Xt = pd.read_csv("../input/Test.csv")
d2 = pd.read_csv("../input/Train.csv") 


# In[ ]:


d2.head()


# In[ ]:


d1.head()


# In[ ]:


Xt.head()


# In[ ]:


df = d2.values


# In[ ]:


x = df[:, 0]
print(x[:2])
print(x.shape)


# In[ ]:


y = df[:, 1]
print(y[:2])
print(y.shape)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le = LabelEncoder()


# In[ ]:


y = le.fit_transform(y)
y


# In[ ]:


from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# In[ ]:


tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()


# In[ ]:


def getCleanReview(review):
    
    review = review.lower()
    review = review.replace("<br /><br />"," ")
    
    #Tokenize
    tokens = tokenizer.tokenize(review)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    
    cleaned_review = ' '.join(stemmed_tokens)
    
    return cleaned_review


# In[ ]:


x = [getCleanReview(i) for i in x]


# In[ ]:


print(x[:2])


# In[ ]:


xt = Xt.values
xt = xt.reshape(-1, )
xt.shape


# In[ ]:


xt[:2]


# In[ ]:


xt = [getCleanReview(i) for i in xt]


# In[ ]:


xt[:2]


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(ngram_range=(2, 2))


# In[ ]:


tf.fit(x)
x = tf.transform(x)
xt = tf.transform(xt)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


mb = MultinomialNB()
mb.fit(x,y)


# In[ ]:


pred = mb.predict(xt)


# In[ ]:


d1['label'] = ['pos' if each == 1 else 'neg' for each in pred ]


# In[ ]:


d1.to_csv('submission.csv', index=None)


# In[ ]:




