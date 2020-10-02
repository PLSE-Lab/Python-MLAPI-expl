#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

from gensim import corpora, models
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
import spacy
from spacy import displacy
from spacy.matcher import Matcher
from spacy.tokens import Span
import en_core_web_lg
from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/train.csv", usecols=[0,1,2])
data.sample(5)


# ## Target

# Inspect distribution of target variable

# In[ ]:


data.target.describe()


# In[ ]:


sns.distplot(data.target)


# Convert continuous target variable to binary variable

# In[ ]:


data.loc[:,"target_binary"] = np.where(data.target < 0.5, 0, 1)


# In[ ]:


data.target_binary.value_counts()


# ## Features

# In[ ]:


# divide data into two dataframes: positive and negative
positive = data[data.target_binary == 0]
positive = positive.sample(100000)

negative = data[data.target_binary == 1]
negative = negative.sample(100000)


# In[ ]:


# convert content of positive comment_text column to one single string
positive_string = " ".join([word for word in positive.comment_text])
print(len(positive_string))
print(positive_string[:100])


# In[ ]:


wordcloud = WordCloud(max_font_size=50, background_color="white").generate(positive_string)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


# convert content of negative comment_text column to one single string
negative_string = " ".join([word for word in negative.comment_text])
print(len(negative_string))
print(negative_string[:100])


# In[ ]:


wordcloud = WordCloud(max_font_size=50, background_color="white").generate(negative_string)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# ## Preprocessing

# In[ ]:


sample = data.sample(1000)


# ### Use spaCy functionalities to preprocess the comments

# In[ ]:


# load spaCy's english large language model
nlp = spacy.load("en_core_web_lg", disable=["ner"])


# I am disabling the pipeline step "NER" in order to be able to set my own NER tags later. These are remaining steps names and classes, which the en_core_web_lg pipeline executes after configuration:

# In[ ]:


nlp.pipeline


# spaCy's Matcher class offers the opportunity to search for words, defined by attributes and rules:
# * Official docs: https://spacy.io/usage/rule-based-matching#matcher
# * Rule Explorer: https://explosion.ai/demos/matcher

# Set up Matcher class

# In[ ]:


# instantiate Matcher
matcher = Matcher(nlp.vocab)

# define pattern
pattern = [{"IS_ALPHA": True,
            "IS_STOP": False,
            "LENGTH": {">": 1},
            "LENGTH": {"<=": 20}
           }]

# add pattern to matcher
matcher.add("Cleaning", None, pattern)


# Apply matcher on comments

# In[ ]:


# initialize empty list for proccessed texts
texts = []

for idx, row in sample.iterrows():
    # get nlp doc of comment text
    doc = nlp(row.comment_text)
    
    # apply matcher on doc
    matches = matcher(doc)
    
    # initialize empty list for matched tokens
    token_matches = []
    
    for match_id, start, end in matches:
        # add custom entitiy "MATCH" to doc.ents
        doc.ents = list(doc.ents) + [Span(doc, start, end, label="MATCH")]  
    
        # get lemma for matched tokens and write to data frame
        token_matches.append(doc[start:end].lemma_.lower())
        sample.loc[idx, "comment_preprocessed"] = " ".join(token_matches)
    
    # append processed comment to list of texts
    texts.append(token_matches)


# Use displaCy to inspect matched tokens

# In[ ]:


displacy.render(doc, style="ent", options={"ents": ["MATCH"]})


# In[ ]:


sample[["comment_text", "comment_preprocessed"]].sample(10)


# ### Use gensim functionalities to vectorize the preprocessed comment text

# Convert list of comments to gensim dictionary 

# In[ ]:


dictionary = corpora.Dictionary(texts)


# In[ ]:


print("The dictionary consists of {} different tokens. In total, {} documents were processed.".format(dictionary.num_pos, dictionary.num_docs))


# Convert dictionary to tfidf-weighted corpus and serialize corpus

# In[ ]:


# get bow representation for each text
corpus_bow = [dictionary.doc2bow(text) for text in texts]

# serialize corpus
corpora.MmCorpus.serialize("corpus.mm", corpus_bow)

# get tfidf representation for each text
corpus_tfidf = models.TfidfModel(corpus_bow)


# Inspect resulting tfidf corpus: list of tuples constisting of token id and tfidf weight.

# In[ ]:


#for document in corpus_tfidf[corpus_bow]:
#    for token in document:
#        print(token)


# In[ ]:


for token, id in dictionary.token2id.items():   
    if id == 6007:
        print(token)


# ## Machine Learning
# Use simple NaivesBayes to challenge different preprocessing techniques.

# In[ ]:


# instantiate NB model
clf = MultinomialNB()

# fit classifier on data
clf.fit(corpus_tfidf[corpus_bow], list(sample.target_binary.values))

