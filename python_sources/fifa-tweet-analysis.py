#!/usr/bin/env python
# coding: utf-8

# In[ ]:


**Word Cloud of all the tweets **


# In[ ]:


# this word cloud takes apprx. 1hr and 15 mins to run completely, please take this into consideration
#while running... 

import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# reading the data from csv
data = pd.read_csv('../input/FIFA.csv')

#deleting rows with blank tweets
data['Tweet'].replace('  ', np.nan, inplace=True)
data = data.dropna(subset=['Tweet'])
tweet = data['Tweet']


#apply tokenization
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
comment_words = ' '
stopwords = set(STOPWORDS)

def stem_tokenize(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    #token_sanitize = re.sub("[^a-zA-Z]+","", str(tokens))
    #stems = stem_tokenize(token_sanitize , stemmer)
    stems = stem_tokenize(tokens , stemmer)
    return ' '.join(stems)

corpus = []

for item in tweet:
    item = item.lower()
    tokens = tokenize(item)
    corpus.append(tokens)
    
for words in corpus:
    comment_words = comment_words + words + ' '
 
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
 
# plot the WordCloud image                       
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


# **Top 10 sources for tweets**

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('../input/FIFA.csv')

data['Tweet'].replace('  ', np.nan, inplace=True)
data = data.dropna(subset=['Tweet'])

#combining the iphone and ipad tweet count into a single category

data.loc[data['Source'] == 'Twitter for iPhone', 'Source'] = 'Twitter for iPhone/iPad'
data.loc[data['Source'] == 'Twitter for iPad', 'Source'] = 'Twitter for iPhone/iPad'

#Grouping by source and counts

g1 = data.groupby(["Source"]).size().reset_index(name='counts')
g2 = g1.sort_values('counts', ascending=False).reset_index(drop=True)

#Taking only top 10 sources by count
g3 = g2.head(10)

#Plotting a simple bar chart for view purposes

g3.plot(x="Source", y = "counts", kind="bar")
plt.show()
data.loc[data['Source'] == 'Twitter for iPhone', 'Source'] = 'Twitter for iPhone/iPad'
data.loc[data['Source'] == 'Twitter for iPad', 'Source'] = 'Twitter for iPhone/iPad'

