#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install vaderSentiment')


# In[ ]:


get_ipython().system('pip install textmining')
get_ipython().system('pip install nltk')


# In[ ]:


#Load libraries
import csv
from textblob import TextBlob
import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.corpus import stopwords
import string

import nltk 

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


# In[ ]:



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#Load Text data
post = pd.read_csv("/kaggle/input/post_1.csv")


# In[ ]:


# Select few text
post = post.iloc[:1000]


# In[ ]:


post.rename(columns={'Post':'comments'},inplace=True)


# In[ ]:


# Extract stop words
stop = set(stopwords.words("english"))

# Remove punctuation marks
exclude = set(string.punctuation)


# In[ ]:


exclude


# In[ ]:


# Text pre processing
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = " ".join([ch for ch in stop_free if ch not in exclude])
    num_free = " ".join(i for i in punc_free if not i.isdigit())
    return num_free

post_corpus = [clean(post.iloc[i,1]) for i in range(0, post.shape[0])]


# In[ ]:


post_corpus


# In[ ]:


get_ipython().system('pip install textmining3')


# In[ ]:


import textmining 


# In[ ]:


# Create document term matrix
tdm = textmining.TermDocumentMatrix()
for i in post_corpus:
    
    tdm.add_doc(i)


# In[ ]:


# Write tdm
tdm.write_csv("TDM_DataFRame.csv", cutoff = 1)


# In[ ]:


# Load dataframe for analysis
df = pd.read_csv("TDM_DataFRame.csv")


# In[ ]:


#Plot wordcloud
wordcloud = WordCloud(width = 1000, hieght = 500, stopwords = STOPWORDS, background_color = 'white').generate(
                        ''.join(post['Post']))

plt.figure(figsize = (15,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ## Sentiment Analysis

# In[ ]:


# Sentiment analysis using Text Blob
# Create empty dataframe to store results
FinalResults = pd.DataFrame()

# Run Engine
for i in range(0, post.shape[0]):
    
    blob = TextBlob(post.iloc[i,1])
    
    temp = pd.DataFrame({'Comments': post.iloc[i,1], 'Polarity': blob.sentiment.polarity}, index = [0])
    
    FinalResults = FinalResults.append(temp)    


# In[ ]:


# Sentiment Analysis using Vader
FinalResults_Vader = pd.DataFrame()

# Create engine
analyzer = SentimentIntensityAnalyzer()

# Run Engine
for i in range(0, post.shape[0]):
    
    snt = analyzer.polarity_scores(post.iloc[i,1])
    
    temp = pd.DataFrame({'Comments': post.iloc[i,1], 'Polarity': list(snt.items())[3][1]}, index = [0])

    FinalResults_Vader = FinalResults_Vader.append(temp)    


# In[ ]:





# In[ ]:


import pandas as pd
post_1 = pd.read_csv("../input/post_1.csv")

