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


#importing the libraries
import pandas as pd
import numpy as np


# Importing the train, test and stopword datasets

# In[ ]:


news_train_test = pd.read_csv('/kaggle/input/odia-news-dataset/train.csv')
news_validate = pd.read_csv('/kaggle/input/odia-news-dataset/valid.csv')
print(news_train_test.head(5))


# In[ ]:


#importing the stopword dictionary. You can create your own dictionary.
stopwords = pd.read_csv('/kaggle/input/stopwords-odia/stopwords.csv')
sw_arr = stopwords.to_numpy()
print(sw_arr)


# Now we need to clean the headlines. We have to remove punctuations and symbols from our news corpus. We will create a method to remove punctuations and symbols. Later we will call this method.

# In[ ]:


def removePunctuations(headline):
    headline = headline.replace(',',' ')
    headline = headline.replace(':',' ')
    headline = headline.replace(';',' ')
    headline = headline.replace('.',' ')
    headline = headline.replace('\'','')
    headline = headline.replace('-',' ')
    return headline;


# Now we will iterate the news corpus and call removePunctuations(headline) and then will remove the stopwords from each headlines. We will append the filtered texts to an array. 

# In[ ]:


news_arr = []
for headline in news_train_test['headings'] :
    filtered_news_string = ''
    headline=removePunctuations(headline)
    for word in headline.split(' '):
        if word not in sw_arr:
            filtered_news_string = filtered_news_string+word+' '
    news_arr.append(filtered_news_string)


# Now as we have the filtered texts in an array. Now we have to add this array to our news corpus to get the label mapping.

# In[ ]:


#creating a new dataframe to store the filtered news array headlines
dataset_new = pd.DataFrame(news_arr, columns=['filter_news'])
print(dataset_new.head(5))

#Now we will concat the filtered news dataset with our original news corpus so that we can get the labels against filtered headlines
news_train_test = pd.concat([news_train_test, dataset_new], axis = 1)
print(news_train_test.columns)


# Now we have our news corpus got cleaned. We can further clean it using **Lemmatizing** long words or by **stemming** them.
# Now we are going to extract the vectors out of the words from our news corpus.

# In[ ]:


#importing CountVectorizer to create vectors
from sklearn.feature_extraction.text import CountVectorizer

#we will vectorize each words in a documents.these vectors will be our features to train the model
vectorizer = CountVectorizer(analyzer = "word",max_features = 1700)
x= vectorizer.fit_transform(news_train_test['filter_news']).toarray()

#now we will store our features in x
print(x)


# In[ ]:


#our target variable
print(news_train_test['label'].unique)


# Our target variable i.e. label column is in text format. So we need to encode it. We will use **LabelEncoder** class from **sklearn.preprocessing** library to encode our target value.

# In[ ]:


#importing library
from sklearn.preprocessing import LabelEncoder
cat_encoder = LabelEncoder()

#encoding the label column from our news corpus as type category
y = cat_encoder.fit_transform(news_train_test['label'].astype('category'))
print(y)


# Now we will split our corpus into train data and test data. To split the corpus we will use **train_test_split** from **sklearn.model_selection** package

# In[ ]:


#importing the library
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# Now we have our training set as x_train, y_train and test set as x_test, y_test. We can check the shape of our train and test set. You can check the shape of your train and test test. Number of rows in x_train and y_train should always match.

# In[ ]:


print('Shape of train data')
print(x_train.shape)
print(y_train.shape)

print('Shape of test data')
print(x_test.shape)
print(y_test.shape)


# Now we will build our classification model. As we are working on text documents and we have 1700 features, it is not advisable to use any random classifier. When we have a comparatively huge number of columns we will use Multinomial Naive Bayes. So we need to import **MultinomialNB** classifier from **sklearn.naive_bayes**

# In[ ]:


#Model building Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()

#fitting our train data with our classifier to create the model
classifier.fit(x_train, y_train)

print('Training data accuracy')
print(classifier.score(x_train , y_train))


# Now we will predict the label of our test dataset.

# In[ ]:


y_pred = classifier.predict(x_test)
print('Test data accuracy')
print(classifier.score(x_test , y_test))

