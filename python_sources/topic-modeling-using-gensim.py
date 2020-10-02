#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_csv("../input/abcnews-date-text.csv", error_bad_lines=False)


# In[3]:


data.head()


# In[7]:


text = data[['headline_text']]


# In[8]:


text.head()


# In[9]:


text['index'] = text.index


# In[10]:


text.head()


# In[11]:


documents = text


# In[12]:


documents.head()


# In[13]:


print("Total length of the documents: {}".format(len(documents)))


# ## Data Pre-Processing

# In[15]:


# importing the gensim and nltk libraries

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import nltk
np.random.seed(42)


# In[22]:


def preprocessing(sentence):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(sentence, pos='v'))

def preprocess(sentence):
    result = []
    
    for token in gensim.utils.simple_preprocess(sentence):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(preprocessing(token))
            
    return result


# In[18]:


sample = documents[documents['index'] == 4310].values[0][0]

print("Sample document is selected for pre-processing: {}".format(sample))


# In[20]:


words = []

for word in sample.split(' '):
    words.append(word)
    
print("Words found after splitting the sample document: {}".format(words))


# In[23]:


print("Tokenized and lemmatized document: {}".format(preprocess(sample)))


# In[24]:


# pre-processing all the documents

preprocessed_documents = documents['headline_text'].map(preprocess)


# In[25]:


preprocessed_documents[:10]


# In[26]:


# creating a dictionary from the above processed documents

dictionary = gensim.corpora.Dictionary(preprocessed_documents)


# In[28]:


count = 0

for key, value in dictionary.iteritems():
    print("Key: {} and Value: {}".format(key, value))
    count += 1
    
    if count > 10:
        break


# In[29]:


# filter out extreme tokens in the document

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)


# In[30]:


bag_of_words = [dictionary.doc2bow(document=document) for document in preprocessed_documents]


# In[31]:


bag_of_words[4310]


# In[35]:


## preview of bag of words of our sample preprocessed document

sample_bag_of_words = bag_of_words[4310]

for i in range(len(sample_bag_of_words)):
    print("Word: {} (\"{}\") appears: {} times.".format(sample_bag_of_words[i][0], dictionary[sample_bag_of_words[i][0]], sample_bag_of_words[i][1]))


# # TF-IDF

# In[38]:


from gensim import corpora, models

tfidf = models.TfidfModel(bag_of_words)
corpus_tfidf = tfidf[bag_of_words]

from pprint import pprint

for document in corpus_tfidf:
    pprint(document)
    break


# # Running LDA using Bag of words 

# In[39]:


# training our model using gensim LdaMulticore

model = gensim.models.LdaMulticore(bag_of_words, num_topics=10, id2word=dictionary, passes=2, workers=2)


# In[40]:


for index, topic in model.print_topics(-1):
    print("Topic: {} \n Words: {}".format(index, topic))


# In[41]:


tfidf_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)

for index, topic in tfidf_model.print_topics(-1):
    print("Topic: {}, Words: {}".format(index, topic))


# ## Evaluation by classifing simple document using LDA Bag of words model

# In[42]:


preprocessed_documents[4310]


# In[43]:


for index, score in sorted(tfidf_model[bag_of_words[4310]], key=lambda tup: -1 * tup[1]):
    print("\nScore: {} \t \nTopic: {}".format(score, tfidf_model.print_topics(index, 10)))


# ### Testing on unseen document

# In[44]:


test_document = "How a Pentgon deal became an identity crisis for Google"

bag_of_words_vector = dictionary.doc2bow(preprocess(test_document))


# In[47]:


for index, score in sorted(tfidf_model[bag_of_words_vector], key=lambda tup: -1 * tup[1]):
    print("Score: {} \t Topic: {}\n".format(score, tfidf_model.print_topics(index, 5)))


# In[ ]:




