#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder 
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

import nltk

from nltk.stem.porter import PorterStemmer

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json") 


# In[ ]:


train.head(3)


# In[ ]:


train["ingredients"] = [", ".join(ingredients) for ingredients in train.ingredients]
test["ingredients"] = [", ".join(ingredients) for ingredients in test.ingredients]
target_enc = LabelEncoder()
y = target_enc.fit_transform(train.cuisine)


# In[ ]:


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = [PorterStemmer().stem(word) for word in tokens]
    return(stems)
tfidf = TfidfVectorizer(tokenizer=tokenize) #this doesn't help.


# In[ ]:


X = tfidf.fit_transform(train.ingredients)


# In[ ]:


train.shape


# In[ ]:


model = make_pipeline(TfidfVectorizer(), LinearSVC(C = 0.5))
model.fit(train.ingredients, y)


# In[ ]:


svd = TruncatedSVD(n_components=300)


# In[ ]:


X_proj = svd.fit_transform(X)


# In[ ]:


np.cumsum(svd.explained_variance_ratio_)[-1]


# In[ ]:


model = LinearSVC(C = 0.5)
#model = LogisticRegression()


# In[ ]:


cross_val_score(model, X_proj, y)


# In[ ]:





# In[ ]:


cross_val_score(model, train.ingredients, y)


# In[ ]:


preds = model.predict(test.ingredients)
preds = target_enc.inverse_transform(preds)


# In[ ]:


solution = pd.DataFrame({"id":test.id, "cuisine":preds})


# In[ ]:


solution.head(3)


# In[ ]:


solution.to_csv("solution1.csv", index = False)

