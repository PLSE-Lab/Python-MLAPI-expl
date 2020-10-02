#!/usr/bin/env python
# coding: utf-8

# Sentimental analysis on twitter data.
# 
# The objective is to find positivity in the tweets and also to check how much factual oriented are the tweets.
# 
# This can be achieved by sentimental analysis on tweets.
# 
# We use inbuilt python library to enable us perfrom sentimental analysis.
# ![image.png](attachment:image.png)

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


data = pd.read_csv("/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv",encoding='latin-1')
data.head()


# Twitter data is read in the csv format.It does not have columns at first so the first step will be giving column names.

# In[ ]:


DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "TweetText"]
data.columns = DATASET_COLUMNS
data.head()


# Data is given columns like target which is the class label.
# Followed by twitter ids.
# Followed by data and flag.
# The next column will be the column for analysis that is TweetText.

# In[ ]:


data.drop(['ids','date','flag','user'],axis = 1,inplace = True)


# We should drop cells which are not needed for analysis to simplify the process.
# 
# Hence cells like ids,date,flag and user is dropped since we have to analyse the tweets only.

# In[ ]:


data.head()


# Data just has tweets and targets after dropping unwanted columns

# Text analysis also requires preprocesing just like numerical analysis.So few text pre-processing measures are used to clean the data.

# In[ ]:


data['CleanTweet'] = data['TweetText'].str.replace("@", "") 
data.head()


# First we have to remove punctuation marks to cleanse text data.

# In[ ]:


data['CleanTweet'] = data['CleanTweet'].str.replace(r"http\S+", "") 
data.head()


# In[ ]:


data['CleanTweet'] = data['CleanTweet'].str.replace("[^a-zA-Z]", " ") 
data.head()


# Next pre-processing technique would be removal of stop words. Stop words are the words which occur most frequently but do not add much meaning to the text other than grammatical usage.So it can be removed.

# In[ ]:


import nltk
stopwords=nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    clean_text=' '.join([word for word in text.split() if word not in stopwords])
    return clean_text

data['CleanTweet'] = data['CleanTweet'].apply(lambda text : remove_stopwords(text.lower()))
data.head()


# After stop words are removed we next tokenize the content 

# In[ ]:


data['CleanTweet'] = data['CleanTweet'].apply(lambda x: x.split())
data.head()


# Stemming is the process of removing few characters from the word so that it is made to reach root form of the word.

# In[ ]:


# from nltk.stem.porter import * 
# stemmer = PorterStemmer() 
# data['CleanTweet'] = data['CleanTweet'].apply(lambda x: [stemmer.stem(i) for i in x])
# data.head()


# Next we stitch back the words.

# In[ ]:


# data['CleanTweet'] = data['CleanTweet'].apply(lambda x: ' '.join([w for w in x]))
# data.head()


# Removing words with less than 3 characters

# In[ ]:


# data['CleanTweet'] = data['CleanTweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# data.head()


# Visulaizing most frequent words in tweets.

# In[ ]:


# import matplotlib.pyplot as plt
# all_words = ' '.join([text for text in data['CleanTweet']])

# from wordcloud import WordCloud 
# wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 

# plt.figure(figsize=(10, 7)) 
# plt.imshow(wordcloud, interpolation="bilinear") 
# plt.axis('off') 
# plt.show()


# ![image.png](attachment:image.png)

# We use Textblob function to check the positivity and negativity of tweets.

# In[ ]:


# from textblob import TextBlob


# In[ ]:


# data['polarity'] = data['CleanTweet'].apply(lambda CleanTweet: TextBlob(CleanTweet).sentiment[0])
# data.head()


# ![image.png](attachment:image.png)

# In[ ]:


# data['subjectivity'] = data['CleanTweet'].apply(lambda CleanTweet: TextBlob(CleanTweet).sentiment[1])
# data.head()


# ![image.png](attachment:image.png)

# In[ ]:


# data['polarityinwords'] = np.where(data.polarity>0.000000000,'positive','negative')
# data.head()


# ![image.png](attachment:image.png)

# In[ ]:


# data['subjectivityinwords'] = np.where(data.subjectivity>0.5,'factual','personalopinion')

# data.head()


# ![image.png](attachment:image.png)

# In[ ]:


# data['polarityinwords'].value_counts()


# ![image.png](attachment:image.png)

# In[ ]:


# data['subjectivityinwords'].value_counts()


# ![image.png](attachment:image.png)

# Word cloud for positive tweets alone

# In[ ]:


# positive_words =' '.join([text for text in data['CleanTweet'][data['polarityinwords'] =='positive']]) 
# wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)

# plt.figure(figsize=(10, 7)) 
# plt.imshow(wordcloud, interpolation="bilinear") 
# plt.axis('off') 
# plt.show()


# Word Cloud for negative tweets

# In[ ]:


# depressive_words =' '.join([text for text in data['CleanTweet'][data['polarityinwords'] =='negative']]) 
# wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(depressive_words)

# plt.figure(figsize=(10, 7)) 
# plt.imshow(wordcloud, interpolation="bilinear") 
# plt.axis('off') 
# plt.show()


# > There is more negativity than positivity.
# 
# > Being a social media platform it is correct that almost 75% the tweets examined are of personal opinion and it holds good for majority of the tweets too.
