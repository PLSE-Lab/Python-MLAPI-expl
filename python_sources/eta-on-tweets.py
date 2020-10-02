#!/usr/bin/env python
# coding: utf-8

# **Exploratory Text Analysis**
# 
# Hi Iam going to do an Exploratory Text Analysis on the obama tweet dataset. I will illustrate the basic Text analysis along with sentiment analysis using TextBlob on the Tweets and classify the tweets as positive or opismistic twees, negative or pessimistic tweets and neutral or general tweets. This would be my first kernel and i would really appreciate your inputs and suggestions to improve my skills.

# First things first, we will import all the required packages and dependencies for our project.

# In[ ]:


import numpy as np 
import pandas as pd 
import os

import matplotlib.pyplot as plt
import matplotlib
import re
from textblob import TextBlob
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import nltk
import seaborn as sns


# In[ ]:


data = pd.read_csv('../input/Tweets-BarackObama.csv')


# As we can see below the dataset consists of the date, username, tweet-text, link and number of retweets and likes. We will consider majorly on the Tweet-text column to perform the sentiment analysis

# In[ ]:


data=data.rename(columns={'Tweet-text' : 'text'})


# In[ ]:


data.head(10)


# Lets explore the top 5 tweets based on number of likes and number of retweets

# In[ ]:


data.sort_values('Likes', ascending=False)[['text','Likes']].head(5)


# In[ ]:


data.sort_values('Retweets', ascending=False)[['text','Retweets']].head(5)


# Splitting the Date column to do further analysis
# 

# In[ ]:


data[['date','time']]=data.Date.str.split("_", expand = True)
data[['year','month','date']] = data.date.str.split("/", expand = True)


# Let's visualise the number of tweets per year over the period of 2012-2019

# In[ ]:


x=data['year'].value_counts()
x=x.sort_index()
sns.barplot(x.index,x.values,alpha=0.8)
matplotlib.rc('figure',figsize=[6,4])
plt.show()


# **Preprocessing**
# 
# The Preprocessing of the text starts with converting the texts to lower case and concatinating the texts with whitespace seperator. In order to remove the links in the tweets i have used the regular expression. Also Regular expression is used to remove the special characters and smileys in the tweets.

# In[ ]:


txt = data['text'].str.lower().str.cat(sep=' ')
text1 = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', txt)
text1 = re.sub('[^A-Za-z]+',' ' , text1)


# Next step involves removing the stopwords from the tweets and counting the frequency of each words

# In[ ]:


words = nltk.tokenize.word_tokenize(text1)
word_dist = nltk.FreqDist(words)
stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 

print('All frequencies, excluding STOPWORDS:')
print('=' * 40)
rslt = pd.DataFrame(words_except_stop_dist.most_common(10),
                    columns=['Word', 'Frequency']).set_index('Word')
print(rslt)
print('=' * 40)
matplotlib.style.use('ggplot')
matplotlib.rc('figure', figsize=[8,5])
rslt.plot.bar(rot=90)


# TextBlob is used to understand the sentiments of the tweets. We are gonna broadly classify the tweets as optimistic, nuetral and pessimistic.  

# In[ ]:


data['polarity'] = data['text'].map(lambda text: TextBlob(text).sentiment.polarity)


# In[ ]:


cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# Plotting the polarity of the tweets helps us to visualize that most of the tweets are neutral in nature with comparetively less pessimistic tweets than optimistic tweets. Lets us further list the top 5 tweets from all the three categories.

# In[ ]:


data['polarity'].iplot(
    kind='hist', bins=20,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')


# In[ ]:


data[data.polarity==-1].text.head(5)


# In[ ]:


data[data.polarity==0].text.head(5)


# In[ ]:


data[data.polarity==1].text.head(5)


# **Conclusion**
# 
# Hence i conclude this kernel with these analysis on the tweeter data and will be further more exploring on other various interesting datasets. Thanks for the support and time.
# 

# In[ ]:




