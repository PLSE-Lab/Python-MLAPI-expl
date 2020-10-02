#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Predicting Part-of-Speech Tags

# In[ ]:


import spacy
# Load the small English model
nlp = spacy.load('en_core_web_sm')
# Process a text
doc = nlp("She ate the pizza")
# Iterate over the tokens
for token in doc:
    # Print the text and the predicted part-of-speech tag
    print(token.text, token.pos_)


# # Predicting Syntactic Dependencies

# In[ ]:


for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)


# # Predicting Syntactic Dependencies

# In[ ]:


for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)


# # Predicting Named Entities

# In[ ]:


# Process a text
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
# Iterate over the predicted entities
for ent in doc.ents:
    # Print the entity text and its label
    print(ent.text, ent.label_)


# In[ ]:




nlp = spacy.load('en')

doc = nlp("my friend Mary has worked at Google since 2009")

for ent in doc.ents:
    print(ent.text, ent.label_)


# # Explain Method

# In[ ]:


print("GPE  :-", spacy.explain('GPE'))
print("NNP  :-", spacy.explain('NNP'))
print("DOBJ :-", spacy.explain('dobj'))


# In[ ]:


get_ipython().system('python -m spacy download en_core_web_md')


# # Document Similarity

# In[ ]:


import spacy
# Load a larger model with vectors
nlp = spacy.load('en_core_web_md')
# Compare two documents
doc1 = nlp("I like fast food")
doc2 = nlp("I like pizza")
print(doc1.similarity(doc2))


# In[ ]:


# Compare two tokens
doc = nlp("I like pizza and pasta")
token1 = doc[2]
token2 = doc[4]
print("Token 1, Token 2 : ",token1," , ",token2)
print(token1.similarity(token2))


# In[ ]:


# Compare a document with a token
doc = nlp("I like pizza")
token = nlp("soap")[0]
print(doc.similarity(token))


# In[ ]:


# Compare a span with a document
span = nlp("I like pizza and pasta")[2:5]
doc = nlp("McDonalds sells burgers")
print("span :- ",span)
print("doc :- ",doc)
print(span.similarity(doc))


# # Using Word Vectors to Predict Similarity in spaCy
# * Similarity is determined using word vectors
# * Multi-dimensional meaning representations of words
# * Generated using an algorithm like Word2Vec and lots of text
# * Can be added to spaCy's statistical models
# * Default: cosine similarity, but can be adjusted
# * Doc and Span vectors default to average of token vectors
# * Short phrases are better than long documents with many irrelevant word

# In[ ]:


# Load a larger model with vectors
nlp = spacy.load('en_core_web_md')
doc = nlp("I have a banana")
# Access the vector via the token.vector attribute
print(doc[3].vector)

