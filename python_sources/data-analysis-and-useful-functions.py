#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import io


# **Competition rules in the nutshell:**
# 
# * This competition does not allow external data.
# * Be aware that this is being run as a Kernels Only Competition, requiring that all submissions be made via a Kernel output.
# * Both your training and prediction should fit in a single Kernel. 
# * GPUs are enabled for this competition. If you use GPUs, you will be limited to 2 hours of run time. If you do not use GPUs, you will be limited to 6 hours of run time. 
# * No internet access enabled
# * No multiple data sources enabled
# * No custom packages
# * Submission file must be named "submission.csv"

# **Competition goal**
# > In this competition you will be predicting whether a question asked on Quora is sincere or not.
# 

# **Data that you can use:**
# 
# 1. train.csv 
# 2. test.csv
# 3. sample_submission.csv
# 4. embeddings:
#     *  google
#     *  glove
#     *  paragram
#     * wiki

# **1. Train **
# 
# *train.csv*

# In[ ]:


train = pd.read_csv("../input/train.csv")
train.head()


# In[ ]:


#print((train['target'] == 1).sum())
#print((train['target'] == 0).sum())

x = np.arange(2)
values = [(train['target'] == 0).sum(), (train['target'] == 1).sum()]
fig, ax = plt.subplots()
plt.bar(x, values)
plt.xticks(x, ('0', '1'))
plt.title('target column in train.csv')
plt.show()


# Let's see sample questions that attack the rules and are marked as 1:

# In[ ]:


pd.set_option('display.max_colwidth', -1)
train[train['target'] == 1].sample(10)


# And sample properly formed questions:

# In[ ]:


pd.set_option('display.max_colwidth', -1)
train[train['target'] == 0].sample(10)


# **4. Embeddings**
# 
# To start working with the data you can use this functions to read files:

# 
# *GoogleNews-vectors-negative300*
# 
# Usage:

# In[ ]:


from gensim.models.keyedvectors import KeyedVectors  #import this to read binary file, isn't it against rule "No custom packages" ?

def loadGoogleModel(pathToFile):
    googleModel = KeyedVectors.load_word2vec_format(pathToFile, binary=True)
    return googleModel

pathToGoogleFile = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

# googleModel = loadGoogleModel(pathToGoogleFile)
# result = googleModel.most_similar(positive=['dog'], topn=5) #you can also put the negative words ex. negative=['cat'], topn - number of top examples in return
# print(result)


#  *GloVe: Global Vectors for Word Representation*
# 
# Usage:

# In[ ]:


def loadGloveModel(pathToFile):
    print("Loading Glove Model")
    f = open(pathToFile,'r')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done, words loaded!")
    return model

pathToGloveFile = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"

# gloveModel = loadGloveModel(pathToGloveFile)
# print(gloveModel['frog'])


# *Paragram embeddings*
# 
# pretrained model from Cognitive Computation Group, Univeristy Pensylvania 
# 
# Usage:

# In[ ]:


def loadParagramModel(pathToFile):
    print("Loading Paragram Model")
    f = open(pathToFile,'r')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done, words loaded!")
    return model

pathToParagramFile = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"

# paragramModel = loadParagramModel(pathToParagramFile)
# print(paragramModel['frog'])


# *English word vectors: wiki-news*
# 
# Usage:

# In[ ]:


def loadWikiModel(pathToFile):
    print("Loading Wiki Model")
    f = open(pathToFile,'r')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done, words loaded!")
    return model

pathToWikiFile = "../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec"

# wikiModel = loadWikiModel(pathToWikiFile)  
# print(wikiModel['frog'])


# **I will probably proceed to develop this kernel soon. Good luck!** 
