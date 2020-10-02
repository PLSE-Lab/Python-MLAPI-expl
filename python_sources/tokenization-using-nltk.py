#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# **Tokenization:**
# Tokenization is a task of breking a text into words or sentences.

# **Data Description:** We will be using product review data for learning tokenization. Below we can see that there is a column called "reviews.text" which contains test for the reviews, we will do tokenization on this column.

# In[ ]:


import pandas as pd
data = pd.read_csv("../input/GrammarandProductReviews.csv")
data.head()


# Instance of the text on which we will do tokenization.

# In[ ]:


data["reviews.text"][0]


# **NLTK (Natural Language Toolkit):** NLTK is a platform which provides easy to use interfaces and functions to carry out variouf text processing to work with natural language text. [[source]](https://www.nltk.org/)

# In[ ]:


import nltk


# In[ ]:


nltk.word_tokenize(data["reviews.text"][0])


# In[ ]:


data["reviews.text"][0]


# In[ ]:


# Tokenize using the white spaces
nltk.tokenize.WhitespaceTokenizer().tokenize(data["reviews.text"][0])


# In[ ]:


# Tokenize using Punctuations
nltk.tokenize.WordPunctTokenizer().tokenize(data["reviews.text"][0])


# In[ ]:


#Tokenization using grammer rules
nltk.tokenize.TreebankWordTokenizer().tokenize(data["reviews.text"][0])


# 

# **Token Normalization**
# 1. Stemming: A process of removing and replacing suffixes to get to the root form of the word, which is called stem.
# 2. Lemmatization: returns the base or dictionary form of a word.

# In[ ]:


#Original Sentence
data["reviews.text"][0]


# In[ ]:


#STEMMING
words  = nltk.tokenize.WhitespaceTokenizer().tokenize(data["reviews.text"][0])
df = pd.DataFrame()
df['OriginalWords'] = pd.Series(words)
#porter's stemmer
porterStemmedWords = [nltk.stem.PorterStemmer().stem(word) for word in words]
df['PorterStemmedWords'] = pd.Series(porterStemmedWords)
#SnowBall stemmer
snowballStemmedWords = [nltk.stem.SnowballStemmer("english").stem(word) for word in words]
df['SnowballStemmedWords'] = pd.Series(snowballStemmedWords)
df


# In[ ]:


#LEMMATIZATION
words  = nltk.tokenize.WhitespaceTokenizer().tokenize(data["reviews.text"][0])
df = pd.DataFrame()
df['OriginalWords'] = pd.Series(words)
#WordNet Lemmatization
wordNetLemmatizedWords = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in words]
df['WordNetLemmatizer'] = pd.Series(wordNetLemmatizedWords)
df

