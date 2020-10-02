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


from sklearn.feature_extraction.text import CountVectorizer
import collections


# In[ ]:


data = pd.read_csv('/kaggle/input/trump-tweets/trumptweets.csv')


# In[ ]:


data.head()


# In[ ]:


trump_tweets = data['content']


# In[ ]:


string_trump = ""
for string in trump_tweets.values:
    string_trump += string + "."
 


# In[ ]:


import nltk

words = set(nltk.corpus.words.words())

string_trump = " ".join(w for w in nltk.wordpunct_tokenize(string_trump) if w.lower() in words or not w.isalpha())


# In[ ]:


import re
string_trump = re.sub(r'\W+', ' ', string_trump)


# In[ ]:


vectorizer = CountVectorizer(min_df=0, lowercase=True)
vectorizer.fit(trump_tweets)
trump_voc = vectorizer.vocabulary_


# Using favorite numbers we will asses authority value.

# In[ ]:


for index,row in data.iterrows():
    for word in nltk.word_tokenize(row['content']):
        if word in trump_voc:
            trump_voc[word] += trump_voc[word] * (row['favorites'] + 0.01)
        


# In[ ]:


df_trump = pd.DataFrame.from_dict(trump_voc,orient = 'index',columns=['count'])


# In[ ]:


df_trump = df_trump.reset_index()
df_trump.columns = ['word','count']


# In[ ]:


df_trump = df_trump.sort_values('count',ascending = False)


# In[ ]:


trump_most_val_str = ""
for index,row in df_trump.iterrows():
    if row['count'] == float("inf") or row['count'] > 10**24:
        trump_most_val_str += " " + row['word']


# In[ ]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

wordcloud = WordCloud(width = 1000, height = 500).generate(trump_most_val_str)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("your_file_name"+".png", bbox_inches='tight')
plt.show()
plt.close()


# So these are the words that are mostly used and getting likes. We see that political and economic parameters that are put into process are valuable

# # Generate tweets from Most liked words

# In[ ]:


import markovify


# In[ ]:


fav_tweets = ""
for index,row in data.sort_values('favorites',ascending = False).head(100).iterrows():
    fav_tweets += row['content'] + "."
    


# In[ ]:


text_model = markovify.Text(fav_tweets)


# In[ ]:


for i in range(5):
    x = text_model.make_short_sentence(200,tries = 100)
    print(x)


# In[ ]:




