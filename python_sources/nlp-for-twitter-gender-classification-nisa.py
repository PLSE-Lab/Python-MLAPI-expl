#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing for Twitter User Gender Classification

# Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding.
# 
# In this tutorial, I am going to implement Natural Language Process to understand whether the post written on Twitter is written by a man or woman.

# ![](https://enterprisetalk.com/wp-content/uploads/2020/05/IBM-Integrates-Watson-Platform-in-Project-Debater-NLP-Technology.jpg)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Let's import dataset:

# In[ ]:


whole_dataset = pd.read_csv("/kaggle/input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv", encoding = "latin1")


# We are going to use gender and description columns. Let's concat them.

# In[ ]:


dataset = pd.concat([whole_dataset.gender, whole_dataset.description], axis = 1)


# In[ ]:


dataset.head(20)


# As you can see, we have some missing values (NaN). Let's delete this missing rows.

# In[ ]:


dataset = dataset.dropna()


# In[ ]:


dataset.head(20)


# In order to classify our data, we need to get rid of string values.
# 
# female -> 1   male -> 0

# In[ ]:


dataset.gender = [1 if person == "female" else 0 for person in dataset.gender]


# In[ ]:


dataset.head(10)


# Now, I am going to clean my data with using Regular Expression library.(Regular Expression Library is using for searching a pattern)

# In[ ]:


import re


# In[ ]:


first_description = dataset.description[4]
first_description


# In[ ]:


first_description = re.sub("[^a-zA-z]", " ",first_description)
first_description


# Using Regular Expression Library, I deleted ":)" symbol.

# Also, in computer language, "BAND" and "band" words are understood differently. So I am going to convert all letters into lowercase. 

# In[ ]:


first_description = first_description.lower()
first_description


# In this part, I am going to clean all irrelavent words. For example, if we have a sentence like: "I go to the school every day." we don't need some words ("the", "to" etc.) while classifying if a sentence was written by a male or female. I am going to get rid of them.

# In[ ]:


import nltk # natural language took kit
nltk.download("stopwords")
from nltk.corpus import stopwords


# I am going to split my descriptions one by one then check them if it is a stop word or not.

# In[ ]:


first_description = nltk.word_tokenize(first_description)
first_description


# Using word_tokenize method instead of split is more beneficial. Because, for example if you have a word like "shouldn't". split method cannot divide it into two parts but word_tokenize divide it into two parts : should and n't. 

# In[ ]:


first_description = [word for word in first_description if not word in set(stopwords.words("english"))]


# In[ ]:


first_description


# This part, I am going to find root of letters (lemmatization) in order to do classification.

# In[ ]:


import nltk as nlp

lemma = nlp.WordNetLemmatizer()
first_description = [lemma.lemmatize(i) for i in first_description]


# In[ ]:


first_description


# Some words have changed: chiefs -> chief, memories -> memory

# Now, I am going to make a sentence using these words.

# In[ ]:


first_description = " ".join(first_description)


# In[ ]:


first_description


# I have showed you how to claen dataset with using only one sentence. But we should implement this method for whole dataset. Let'S do it together. We need only a for loop:

# In[ ]:


description_list = []
for description in dataset.description:
    description = re.sub("[^a-zA-z]", " ",description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    description = [word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(i) for i in description]
    description = " ".join(description)
    description_list.append(description)


# In[ ]:





# CountVectoizer:
# 
# ![](https://iksinc.files.wordpress.com/2019/12/screen-shot-2019-12-11-at-9.47.32-pm.png?w=1100)

# The CountVectorizer provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

max_features = 5000
count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english")


# In[ ]:


sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()


# In[ ]:


print("5000 most common words: ", count_vectorizer.get_feature_names())


# In[ ]:


x = sparce_matrix


# In[ ]:


x


# Let's implement Naive Bayes Method for the Machine Learning part to make predictions.

# In[ ]:


y = dataset.iloc[:,0].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)


# In[ ]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)


# In[ ]:


predictions = nb.predict(x_test)


# In[ ]:


print("Accuracy: ", nb.score(predictions.reshape(-1,1), y_test))

