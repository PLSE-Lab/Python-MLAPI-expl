#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing pandas library to read csv file.
import pandas as pd


# In[ ]:


# Reading file
quora = pd.read_csv('../input/quora_questions.csv')


# In[ ]:


# Checking top 5 row from the dataframe
quora.head()


# In[ ]:


#Preporcessing

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

dtm = tfidf.fit_transform(quora['Question'])

dtm


# In[ ]:


#Non-negative Matrix Factorization (NMF)
from sklearn.decomposition import NMF

nmf_model = NMF(n_components=20, random_state=42)

nmf_model.fit(dtm)


# In[ ]:


#Print the top 15 most common words for each of the 20 topic.
for index, topic in enumerate(nmf_model.components_):
    print(f"THE TOP 15 WORDS FOR TOPIC #{index}")
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


# In[ ]:


#Adding new column to original quora dataframe that labels each questions into one of the 20 topic categories.

topic_results = nmf_model.transform(dtm)

quora['Topic'] = topic_results.argmax(axis= 1)


# In[ ]:


quora.head(15)

