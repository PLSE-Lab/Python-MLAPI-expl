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


review = pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')


# In[ ]:


review.head()


# In[ ]:


review.isnull().sum()


# In[ ]:


##copy the original and drop NA values
review_orginal = review.copy()


# In[ ]:


review.dropna()


# In[ ]:


review = review.drop(['reviewerID','asin','reviewerName'],axis=1)


# In[ ]:


def rating(x):
    if x == 5:
        return('Excellent')
    elif x==4:
        return('Super')
    elif x==3:
        return('Need to inprove')
    elif x==2:
        return('Requires more changes')
    elif x==1:
        return('Waste')


# In[ ]:


## Converting ratings from starts to text
review['overall'] = review['overall'].apply(lambda x: rating(x))


# In[ ]:


review['overall'].value_counts()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x='overall',data=review)
plt.show()


# In[ ]:


##looks like we should covert 'SUPER' and 'Need to improve' ratings to Excellent and try to retain good feedback from 'requires more changes'
## and waste 


# In[ ]:


review_5 = review[review['overall'] == 'Excellent']
review_43 = review[(review['overall'] == 'Super') | (review['overall'] == 'Need to improve')]
review_21 = review[(review['overall'] == 'Required more changes' ) | (review['overall'] =='Waste')]


# In[ ]:


review_43.head()


# In[ ]:


## Reviewing rating comments 
from wordcloud import WordCloud, STOPWORDS
stopwords = STOPWORDS
wordcloud1 = WordCloud(width=800,height=400,stopwords=stopwords, min_font_size=10,max_words=150).generate(' '.join(review_43['summary']))

wordcloud2 = WordCloud(width=800,height=400,stopwords=stopwords, min_font_size=10,max_words=150).generate(' '.join(review_5['summary']))

wordcloud3 = WordCloud(width=800,height=400,stopwords=stopwords, min_font_size=10,max_words=150).generate(' '.join(review_21['summary']))


# In[ ]:


### most comments given 
plt.figure(figsize=(15,10))
plt.imshow(wordcloud1)
plt.axis('off')
plt.title('Reviwes given by 4 & 3 stars')
plt.show()


# In[ ]:


### most comments given 
plt.figure(figsize=(15,10))
plt.imshow(wordcloud2)
plt.axis('off')
plt.title('Reviwes given by 5 stars')
plt.show()


# In[ ]:


### most comments given 
plt.figure(figsize=(15,10))
plt.imshow(wordcloud3)
plt.axis('off')
plt.title('Reviwes given by 1 & 2 stars')
plt.show()


# In[ ]:


review_5.head(5)


# In[ ]:


## 10 sample reviews of TOP ratings
review_5[['reviewText','summary']].sample(10)


# In[ ]:


## 10 sample reviews of medium ratings
review_43[['reviewText','summary']].sample(10)


# In[ ]:


## 10 sample reviews of LEAST ratings
review_21[['reviewText','summary']].sample(10)


# In[ ]:


## lets look at the overall count of reivews
review_5['summary'].describe()


# In[ ]:


## lets look at the overall count of reivews
review_43['summary'].describe()


# In[ ]:


## lets look at the overall count of reivews
review_21['summary'].describe()


# In[ ]:


review_21_group = review_21.groupby('summary').count()[['reviewText']]
review_21_group.head()


# In[ ]:


## Now lets only consider least rating reviews, so that we can improve the app
review_21_com = review_21[['reviewText','summary']]


# In[ ]:


## new function
def my_tokenizer(x):
    return x.split('.') if x != None else []
review_21_tokens = review_21_com.reviewText.map(my_tokenizer).sum()


# In[ ]:


## worst ratings given
from collections import Counter
counter = Counter(review_21_tokens)
counter.most_common(50)


# In[ ]:


## Checking the Quality of tokens
from spacy.lang.en.stop_words import STOP_WORDS
def remove_token (token):
    return [t for t in token if t not in STOP_WORDS]


# In[ ]:


counter = Counter(remove_token(review_21_tokens))
counter.most_common(50)


# In[ ]:




