#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import recall_score, precision_score, f1_score


# In[ ]:


DATA_JSON_FILE = '../input/email-text-data.json'


# In[ ]:


data = pd.read_json(DATA_JSON_FILE)


# In[ ]:


data.tail() 


# In[ ]:


data.shape


# In[ ]:


data.sort_index(inplace=True)
data.tail()


# In[ ]:


vectorizer = CountVectorizer(stop_words='english')


# In[ ]:


all_features = vectorizer.fit_transform(data.MESSAGE)


# In[ ]:


all_features.shape


# In[ ]:


len(vectorizer.vocabulary_)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(all_features, data.CATEGORY, test_size=0.3, random_state=88)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


classifier = MultinomialNB() 


# In[ ]:


classifier.fit(X_train, y_train)


# In[ ]:


nr_correct = (y_test == classifier.predict(X_test)).sum() 
print(f'{nr_correct} documents classfied correctly')


# In[ ]:


nr_incorrect = y_test.size - nr_correct
print(f'Number of documents incorrectly classified is {nr_incorrect}')


# In[ ]:


fraction_wrong = nr_incorrect / (nr_correct + nr_incorrect)
print(f'The (testing) accuracy of the model is {1-fraction_wrong:.2%}')


# In[ ]:


classifier.score(X_test, y_test)


# In[ ]:


recall_score(y_test, classifier.predict(X_test))


# In[ ]:


precision_score(y_test, classifier.predict(X_test))


# In[ ]:


f1_score(y_test, classifier.predict(X_test))


# In[ ]:


example = ['get viagra for free now!', 
          'need a mortgage? Reply to arrange a call with a specialist and get a quote', 
          'Could you please help me with the project for tomorrow?', 
          'Hello Jonathan, how about a game of golf tomorrow?', 
          'Ski jumping is a winter sport in which competitors aim to achieve the longest jump after descending from a specially designed ramp on their skis. Along with jump length, competitor\'s style and other factors affect the final score. Ski jumping was first contested in Norway in the late 19th century, and later spread through Europe and North America in the early 20th century. Along with cross-country skiing, it constitutes the traditional group of Nordic skiing disciplines.'
          ]


# In[ ]:


doc_term_matrix = vectorizer.transform(example)


# In[ ]:


classifier.predict(doc_term_matrix)


# In[ ]:




