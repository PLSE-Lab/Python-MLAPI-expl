#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Importing Libraries.

# In[ ]:


import numpy as np
import pandas as pd 
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
np.random.seed(2018)

stemmer = SnowballStemmer("english")


# In[ ]:


questions = pd.read_csv('../input/questions.csv')
answers = pd.read_csv('../input/answers.csv')
professionals = pd.read_csv('../input/professionals.csv')
emails = pd.read_csv('../input/emails.csv')
questions.head()


# Creating merged tables for question answers and the professionals who answered questions.

# In[ ]:


question_answers = questions.merge(right=answers, how='inner', left_on='questions_id', right_on='answers_question_id')


# In[ ]:


qa_professionals = question_answers.merge(right=professionals, left_on='answers_author_id', right_on='professionals_id')
qa_professionals.head()


# Hashtag Extraction

# In[ ]:


def extract_hashtags(x):
   
    a = x.split()
 
    hash_tags = [i for i in a if i.startswith("#")]
    

    result = ' '.join(hash_tags) + " "
    
    return result


question_answers['question_hash_tags'] = question_answers['questions_body'].apply(extract_hashtags)

question_answers.head()


# Extracting only the "questions_body" cells

# In[ ]:


df_question_body = question_answers[[ 'questions_id','questions_body',]]
df_question_body['index'] = df_question_body.index
df_question_body.head()


# The Yummy Part, Part I: Preliminary Text Prep

# In[ ]:


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[ ]:


processed_questions = df_question_body[df_question_body['index'] == 0].values[0][1]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(processed_questions))
pdf_question_body = df_question_body['questions_body'].map(preprocess)


# In[ ]:



dictionary = gensim.corpora.Dictionary(pdf_question_body)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 100:
        break


# In[ ]:




