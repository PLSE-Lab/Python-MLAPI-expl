#!/usr/bin/env python
# coding: utf-8

# # Basic model using classic NLP approach
# 
# This model is based on Latent Semantic Analysis (`TruncatedSVD`) and Linear Discriminant Analysis. Also uses `casual_tokenize` tokenizer from NLTK.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.pipeline import make_pipeline, TransformerMixin, make_union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier

from nltk.tokenize import casual_tokenize


# In[ ]:


SEED = 1337
np.random.seed(SEED)


# In[ ]:


train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")


# In[ ]:


train_df.shape, test_df.shape


# In[ ]:


train_df.head()


# In[ ]:


train_df.target.value_counts()


# In[ ]:


class Centerer(TransformerMixin):
    
    def __init__(self):
        self.mean = None
    
    def fit(self, X, y=None):
        self.mean = X.mean(axis=0)
        return self
        
    def transform(self, X, y=None):
        return X - self.mean


# In[ ]:


nb = MultinomialNB()


# In[ ]:


tfidf = TfidfVectorizer(tokenizer=casual_tokenize)


# In[ ]:


class TransformerMultinomialNB(MultinomialNB, TransformerMixin):
    def transform(self, X, y=None):
        return self.predict_proba(X)
    
class TransformerLDA(LDA, TransformerMixin):
    def transform(self, X, y=None):
        return self.predict_proba(X)


# In[ ]:


lda_lsa_pipe = make_pipeline(    
    Centerer(),
    TruncatedSVD(n_components=64, n_iter=100, random_state=SEED),
    TransformerLDA(n_components=1)
)


# In[ ]:


pipe = make_pipeline(
    TfidfVectorizer(tokenizer=casual_tokenize),
    make_union(TransformerMultinomialNB(), lda_lsa_pipe),
    RidgeClassifier()
)


# In[ ]:


# scores = cross_val_score(pipe, train_df.text, train_df.target, cv=3, scoring="f1")
# scores.mean().round(3)


# In[ ]:


pipe.fit(train_df.text, train_df.target)


# In[ ]:


test_df['target'] = pipe.predict(test_df.text)


# In[ ]:


test_df[['id', 'target']].to_csv("submission.csv", index=False)

