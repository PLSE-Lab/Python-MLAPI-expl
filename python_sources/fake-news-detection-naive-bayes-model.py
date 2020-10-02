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


truenews = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
fakenews = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')


# In[ ]:


fakenews.head()


# In[ ]:


truenews.head()


# In[ ]:


fakenews.describe()


# In[ ]:


truenews.describe()


# In[ ]:


truenews['True/Fake']='True'
fakenews['True/Fake']='Fake'


# In[ ]:


# Combine the 2 DataFrames into a single data frame
news = pd.concat([truenews, fakenews])
news["Article"] = news["title"] + news["text"]
news.sample(frac = 1) #Shuffle 100%


# In[ ]:


#Data Cleaning
from nltk.corpus import stopwords
import string


# In[ ]:


def process_text(s):

    # Check string to see if they are a punctuation
    nopunc = [char for char in s if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Convert string to lowercase and remove stopwords
    clean_string = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_string


# In[ ]:


# Tokenize the Article
#rerun, takes LOOOONG
news['Clean Text'] = news['Article'].apply(process_text)


# In[ ]:


news.sample(5)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


bow_transformer = CountVectorizer(analyzer=process_text).fit(news['Clean Text'])

print(len(bow_transformer.vocabulary_)) #Total vocab words


# In[ ]:


#Bag-of-Words (bow) transform the entire DataFrame of text
news_bow = bow_transformer.transform(news['Clean Text'])


# In[ ]:


print('Shape of Sparse Matrix: ', news_bow.shape)
print('Amount of Non-Zero occurences: ', news_bow.nnz)


# In[ ]:


sparsity = (100.0 * news_bow.nnz / (news_bow.shape[0] * news_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))


# In[ ]:


#TF-IDF


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(news_bow)
news_tfidf = tfidf_transformer.transform(news_bow)
print(news_tfidf.shape)


# In[ ]:


#Train Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
fakenews_detect_model = MultinomialNB().fit(news_tfidf, news['True/Fake'])


# In[ ]:


#Model Evaluation
predictions = fakenews_detect_model.predict(news_tfidf)
print(predictions)


# In[ ]:


from sklearn.metrics import classification_report
print (classification_report(news['True/Fake'], predictions))


# In[ ]:


from sklearn.model_selection import train_test_split

news_train, news_test, text_train, text_test = train_test_split(news['Article'], news['True/Fake'], test_size=0.3)

print(len(news_train), len(news_test), len(news_train) + len(news_test))


# In[ ]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=process_text)),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', MultinomialNB()),  
])
pipeline.fit(news_train,text_train)


# In[ ]:


prediction = pipeline.predict(news_test)


# In[ ]:


print(classification_report(prediction,text_test))

