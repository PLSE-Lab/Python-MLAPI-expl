#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv("../input/Tweets.csv")
data.head(2)
#len(data)


# In[ ]:


# Check the ratio of positive and negative tweets for each airline
data['countval'] = 1
#data.head()
groupby_object = data[['airline','airline_sentiment','countval']]                  .groupby(['airline','airline_sentiment']).aggregate(sum)

#print(groupby_object)
groupby_object.unstack(level=1).plot(kind='bar')
#print(groupby_object)
plt.show()


# In[ ]:


from nltk.corpus import stopwords
import re
print(stopwords)

def tweet_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = raw_review
    print(review_text)
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))

processed_tweets = []
for tweet in data['text']:
    processed = tweet_to_words(tweet)
    processed_tweets.append(processed)
    
print(processed_tweets)


# In[ ]:





# In[ ]:




