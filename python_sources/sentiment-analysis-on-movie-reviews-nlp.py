#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.simplefilter('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_csv('../input/train.tsv',sep='\t')


# In[3]:


data.info()


# In[4]:


data.head()


#  The sentiment labels are:
# 
# 0 - negative; 
# 1 - somewhat negative; 
# 2 - neutral; 
# 3 - somewhat positive; 
# 4 - positive

# In[5]:


print(data.iloc[0]['Phrase'],'Sentiment - ',data.iloc[0]['Sentiment'])


# The part of the sentence above which says 'but none of which amounts to much of a story' corresponds to a neagtive sentiment which is correctly indicated in the sentiment label '1'. Hence if we remove this phrase from the sentence as is done below, we get a neutral sentiment as the label!

# In[6]:


print(data.iloc[1]['Phrase'],'Sentiment - ',data.iloc[1]['Sentiment'])


# Addition of a word like 'for' has shifted the sentiment from negative to neutral as shown below. Hence it is not recommended to use stopwords filtering here as we are not analysing full messages here, but phrases from the same sentence. A stopword like 'but' or 'not' can really alter the sentiment and hence filtering out them will be counterproductive.

# In[7]:


print(data.iloc[32]['Phrase'],'Sentiment - ',data.iloc[32]['Sentiment'])
print('\n')
print(data.iloc[33]['Phrase'],'Sentiment - ',data.iloc[33]['Sentiment'])


# Lets clean the phrases by removing punctuation marks and splitting them into a list

# In[8]:


import string
string.punctuation


# In[9]:


from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()

def own_analyser(phrase):
    phrase = phrase.split()
    for i in range(0,len(phrase)):
        k = phrase.pop(0)
        if k not in string.punctuation:
                phrase.append(lm.lemmatize(k).lower())    
    return phrase


# In[10]:


data.columns


# In[11]:


X = data['Phrase']
y = data['Sentiment']


# In[12]:


from sklearn.model_selection import train_test_split
phrase_train,phrase_test,sentiment_train,sentiment_test = train_test_split(X,y,test_size=0.3)


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# Using pipeline feature of sklearn - 

# In[14]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([('BOW',CountVectorizer(analyzer=own_analyser)),
                    ('tfidf',TfidfTransformer()),
                    ('classifier',MultinomialNB())])


# In[15]:


pipeline.fit(phrase_train,sentiment_train)


# In[16]:


predictions = pipeline.predict(phrase_test)


# In[17]:


from sklearn.metrics import classification_report


# In[18]:


print(classification_report(sentiment_test,predictions))


# In[19]:


test_data = pd.read_csv('../input/test.tsv',sep='\t')


# In[20]:


test_data.head()


# In[21]:


test_predictions = pipeline.predict(test_data['Phrase'])


# In[22]:


phrase_id = test_data['PhraseId'].values


# In[23]:


test_predictions.shape


# In[24]:


final_answer = pd.DataFrame({'PhraseId':phrase_id,'Sentiment':test_predictions})


# In[25]:


final_answer.head()


# In[26]:


filename = 'Sentiment Analysis - NaiveBayes.csv'

final_answer.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:




