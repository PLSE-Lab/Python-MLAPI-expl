#!/usr/bin/env python
# coding: utf-8

# **I will use tensor flow to make search more relevent with abstract part of the given json data. I have divided this problem in two parts.**
# 1. Use NLP based search for better accuracy -  TF not working in this notebook, need to try again . if it fails
# 2. Go to standard search using keywords / REG expression.
# 3. Using Whoosh search for keywords based but it needs internet access.
# 4. Hybrid search using NLP + REG expression 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os
 
import pprint
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Load all data required 
path="../input/CORD-19-research-challenge/2020-03-13/"
cord_all_sources=pd.read_csv(path+"all_sources_metadata_2020-03-13.csv")
cord_all_sources.shape


# In[ ]:


cord_all_sources.head(5)


# In[ ]:


# Remove nulls from abstract 
cord_all_sources['abstract'].dropna()


# In[ ]:


cord_all_sources.keys()


# Actual articles files - json

# In[ ]:


biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/'
filenames = os.listdir(biorxiv_dir)
print("Number of articles retrieved from biorxiv:", len(filenames))

all_files = []

for filename in filenames:
    filename = biorxiv_dir + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)


file = all_files[0]
print("Dictionary keys:", file.keys())


# In[ ]:


cord_all_sources['abstract'].head(10)
cord_all_sources['sha'].head(10)


# In[ ]:


# TF  lib not working so starting with simple search and will add TF once its working
# I am going to combine two data files with metadata for better search results and without much efforts to Original  file system.
# Idea is to make all the data as it is, without much efforts to clean or face any performance issues. 
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


def search_covid_data(querystring,key):
    abstract= []
    abstract = (cord_all_sources['abstract'].dropna())
    for i in range(len(abstract)):
        if key == 'any':
            if any(word in str(abstract[1:11]) for word in querystring):
                paper_id = i
                
        elif key=='all':
             if all(word in str(abstract[1:11]) for word in querystring):
                paper_id = i
                
    paper_id_a = cord_all_sources['sha'][paper_id]
     
    
    if file['paper_id'] == paper_id_a:
        print(file['body_text'][0]['text'])
    else:
        print("No Paper ID found or NULL ID. But here is actual text")
        print(cord_all_sources['title'][0])
    



# In[ ]:


# simple search without any heavy lib but accuarcy may not be good
search_covid_data("The geographic spread of 2019 novel coronaviru",'all')
 


# In[ ]:


import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
 
import string
import unicodedata
import sys
from sklearn.metrics import precision_score

def NLPSearch(sSearchString):
    # TF not working - need to look 
     with open(filenames) as json_data:
     data = json.load(json_data)

    categories = list(data.keys())
    words = []
    docs = []

    for each_category in data.keys():
        for each_sentence in data[each_category]:

            each_sentence = remove_punctuation(each_sentence)
            print(each_sentence)

            w = nltk.word_tokenize(each_sentence)
            print("tokenized words: ", w)
            words.extend(w)
            docs.append((w, each_category))

            words = [stemmer.stem(w.lower()) for w in words]
            words = sorted(list(set(words)))

#print(words)
#print(docs)


training = []
output = []

output_empty = [0] * len(categories)


for doc in docs:

    bow = []

    token_words = doc[0]

    token_words = [stemmer.stem(word.lower()) for word in token_words]

    for w in words:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1

    training.append([bow, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])


tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)


model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=2000, batch_size=10, show_metric=True)
model.save('model.tflearn')

model.load('model.tflearn')
sent_1 = "Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.",
"Prevalence of asymptomatic shedding and transmission (e.g., particularly children)."




def get_tf_record(sentence):
    global words
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return(np.array(bow))


# we can start to predict the results for each of the 4 paper
print(categories[np.argmax(model.predict([get_tf_record(sent_1)]))])


# In[ ]:


# pip install Whoosh Search


# In[ ]:


get_ipython().system('pip install Whoosh')


# In[ ]:


from whoosh.fields import Schema, TEXT, ID
from whoosh import index


# 
