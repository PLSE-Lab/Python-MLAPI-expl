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


# In[ ]:


df = pd.read_csv('/kaggle/input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv')


# In[ ]:


df.head()


# In[ ]:


import nltk


# In[ ]:


nltk.download('stopwords')
nltk.download('punkt')


# In[ ]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[ ]:


tokenizer = nltk.RegexpTokenizer(r"\w+")


# In[ ]:


def remove_stopwords(text):
    #text_tokens = word_tokenize(text)
    #tokenizer = nltk.RegexpTokenizer(r"\w+")
    text_tokens = tokenizer.tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    words = [word for word in tokens_without_sw if word.isalpha()]
    filtered_sentence = (" ").join(words)
    return filtered_sentence


# In[ ]:


test_text = "Nick likes to play football, however he is not too fond of tennis....."


# In[ ]:


ouput_text = remove_stopwords(test_text)
print(ouput_text)


# In[ ]:


df['Review Text'] = df['Review Text'].astype(str) 


# In[ ]:


df['Review Text'] = df['Review Text'].head(100).apply(remove_stopwords)


# In[ ]:


df.head(5)


# In[ ]:


text = df['Review Text'].head(100)


# In[ ]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


# Create our vectorizer
vectorizer = CountVectorizer()


# In[ ]:


vectorizer.fit(text)


# In[ ]:


print(vectorizer.vocabulary_)


# In[ ]:


df['vector'] = df['Review Text'].apply(vectorizer.transform())


# In[ ]:


new_df = df['Review Text'].head(100)


# In[ ]:


new_df.describe()


# In[ ]:


new_df.head()


# In[ ]:


lst_vect =[]


# In[ ]:


type(new_df.iloc[0])


# In[ ]:


for i in range(0,100):
    vec = vectorizer.transform([new_df.iloc[i]]).toarray()
    lst_vect.append(vec)
                


# In[ ]:


print(lst_vect[1])


# In[ ]:


len(lst_vect)


# In[ ]:


X = np.array(lst_vect)


# In[ ]:


X.shape


# In[ ]:


X_new = X.reshape((100,1017))


# In[ ]:


y = df['Rating'].head(100).to_numpy()


# In[ ]:


y[0]


# In[ ]:


y.shape


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


clf = MultinomialNB()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.33)


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


y_predict = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


conf_mat = confusion_matrix(y_test,y_predict)


# In[ ]:


print(conf_mat)


# In[ ]:


df_3 = pd.read_csv('/kaggle/input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv')


# In[ ]:


df_4 = df_3.head(5)['Review Text']


# In[ ]:


df_4.head()


# In[ ]:


pos_lst = []


# In[ ]:


for i in range(0,5):
    token_words = word_tokenize(df_4.iloc[i])
    pos_tag = nltk.pos_tag(token_words)
    pos_lst.append(pos_tag)


# In[ ]:


print(pos_lst[0])


# In[ ]:


import sys, time
from nltk import tokenize
from nltk.parse import ViterbiParser
from nltk.grammar import toy_pcfg1, toy_pcfg2


# In[ ]:


# Define two demos.  Each demo has a sentence and a grammar.
    demos = [
        ("I saw the man with my telescope", toy_pcfg1),
        ("the boy saw Jack with Bob under the table with a telescope", toy_pcfg2),
    ]


# In[ ]:


parser = ViterbiParser(toy_pcfg1)


# In[ ]:


text ="I saw the man with my telescope"


# In[ ]:


text_split =text.split()


# In[ ]:


parses = parser.parse_all(text_split)


# In[ ]:


print(parses)


# In[ ]:


my_parsers=[]


# In[ ]:


df_4.iloc[0].split()


# In[ ]:


# parser with tou PCF1
for i in range(0,5):
    tokens = df_4.iloc[i].split()
    parses_x = parser.parse_all(text_split)
    my_parsers.append(parses_x)


# In[ ]:


parser_2 = ViterbiParser(toy_pcfg2)


# In[ ]:


my_parsers_2=[]


# In[ ]:


# parser with tou PCF1
for i in range(0,5):
    tokens = df_4.iloc[i].split()
    parses_x_2 = parser.parse_all(text_split)
    my_parsers_2.append(parses_x_2)


# In[ ]:




