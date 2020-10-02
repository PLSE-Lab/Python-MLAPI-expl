#!/usr/bin/env python
# coding: utf-8

# # In this notebook,i am explaining some basic visualisations on text data.If you are a beginner to NLP,please watch below notebook where i explained some concepts in NLP.

# https://www.kaggle.com/sainathkrothapalli/beginners-approach-to-nlp-problems

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
data.head()


# # Target feature

# In[ ]:


data['target'].value_counts()


# Plotting countplot to see the categories count visually.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.countplot(data['target'])


# We can also use pie chart to percentage of both categories.

# In[ ]:


data['target'].value_counts().head(10).plot.pie(autopct='%1.1f%%')


# # Location and keyword feature

# First checking top 10 location from where the tweets are coming.

# In[ ]:


data['location'].value_counts()[:10].plot(kind='bar')


# Now plotting top 10 keywords,as shown below fatalities is top keyword

# In[ ]:


data['keyword'].value_counts()[:10].plot(kind='bar')


# Now plotting the top 10 locations where disaster tweets are coming,as shown below USA tops the list.

# In[ ]:


data[data['target']==1]['location'].value_counts()[:10].plot(kind='bar')


# # Text feature

# # Word count

# First writing a function to find length of the words in the text and applying it to text feature.

# In[ ]:


def wl(text):
    return len(text.split(" "))
data['word_length']=data['text'].apply(wl)


# Now plotting the word_length,as shown below most of the words in a sentence are between 8 and 22.

# In[ ]:


data['word_length'].hist()


# Now plotting kdeplots,to see the distribution of word_length with respect to disaster and real tweets.As shown below,word_length of both real and disaster tweets lies between 5 and 28,and some of the disaster word count is more than 40.

# In[ ]:


sns.kdeplot(data[data['target']==1]['word_length'],color='g')
sns.kdeplot(data[data['target']==0]['word_length'],color='r')
plt.legend(['disaster','real'])


# Now using the bar plots to check word count in tweets,as shown below disaster tweets has more word count.

# In[ ]:


sns.barplot(x='target',y='word_length',data=data)


# # Character count

# Here also we  are doing the same way as above,first creating a feature called char_length and plotting the distribution using histogram-as shown below many tweets are having char count more than 100.

# In[ ]:


data['char_length']=data['text'].apply(len)
data['char_length'].hist()


# In[ ]:


sns.kdeplot(data[data['target']==1]['char_length'],color='g')
sns.kdeplot(data[data['target']==0]['char_length'],color='r')
plt.legend(['disaster','real'])


# In[ ]:


sns.barplot(x='target',y='char_length',data=data)


# Now using scatterplot to check if there is any relation b/w char count and word count,as shown below there is positive relation that means if char count increases the word count increases.

# In[ ]:


sns.scatterplot(x='char_length',y='word_length',data=data)


# Now using plobability plot to see if there are normally distributed or not.As shown below word_length is some what normally distributes where as char_length is not. 

# In[ ]:


from scipy import stats
import statsmodels.api as sm 
stats.probplot(data['char_length'], plot=plt)


# In[ ]:


stats.probplot(data['word_length'], plot=plt)


# # Unique words in tweets

# Here also we are creating a feature called unique_word_count,and plotting it using histogram,as shown below most unique word count lies b/w 8 and 22.

# In[ ]:


data['unique_word_count'] =data['text'].apply(lambda x: len(set(str(x).split())))


# In[ ]:


data['unique_word_count'].hist()


# In[ ]:


sns.kdeplot(data[data['target']==1]['unique_word_count'],color='g')
sns.kdeplot(data[data['target']==0]['unique_word_count'],color='r')
plt.legend(['disaster','real'])


# Now using scatterplot to check if there is any relation b/w unique word count and word count,as shown below there is almost positive relation that means if unique word count increases the word count increases.

# In[ ]:


sns.scatterplot(x='unique_word_count',y='word_length',data=data)


# Now using scatterplot to check if there is any relation b/w char count and  unique word count,as shown below there is positive relation that means if char count increases the unique word count increases.

# In[ ]:


sns.scatterplot(x='char_length',y='unique_word_count',data=data)


# # Stop words count

# Importing nesessary libraries

# In[ ]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# Now creating feature called stop_words which consist of len of stop words in each tweet.

# In[ ]:


all_stopwords = stopwords.words('english')
data['stop_words']=data['text'].apply(lambda x: len([words for words in str(x).lower().split() if words in all_stopwords]))


# From below histogram,we can say that we can say that,stop words count for most of the tweets is less than 8.

# In[ ]:


data['stop_words'].hist()


# From below bar plot we can say stop words count is less in disaster tweets than real ones.

# In[ ]:


sns.barplot(x='target',y='stop_words',data=data)


# In[ ]:


sns.kdeplot(data[data['target']==1]['stop_words'],color='g')
sns.kdeplot(data[data['target']==0]['stop_words'],color='r')
plt.legend(['disaster','real'])


# Now using scatterplot to check if there is any relation b/w stop words and remaing three created feature,as shown below there is some positive relation b/w stop words and (word_length,unique_word_count)

# In[ ]:


features=['word_length','char_length','unique_word_count']
for i in features:
    sns.scatterplot(x=i,y='stop_words',data=data)
    plt.show()


# Now we can clearly see the correlation b/w the features using below heatmap.

# In[ ]:


corr=data.corr()
sns.heatmap(corr,annot=True)


# As shown above unique_word_count and word_length has 0.97 correlation.
#                unique_word_count and char_length has 0.85 correlation.
#                char_length and word_length has 0.83 correlation.
#                word_length and stop words has 0.75 correlation.

# # Simple code to get bigrams

# In[ ]:


data['text']


# Below is th simple code to get bigrams from text.

# In[ ]:


from nltk.util import ngrams
def get_bigram(text):
    big=''
    token = nltk.word_tokenize(text)
    big=(list(ngrams(token, 2)))
    return str(big)    
data['bigram']=data['text'].apply(get_bigram)       


# In[ ]:


data['bigram']


# # Please upvote if you like,any suggestions and mistakes put it in comments,Thank you.
