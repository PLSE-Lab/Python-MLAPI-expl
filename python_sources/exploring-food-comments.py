#!/usr/bin/env python
# coding: utf-8

# ## Who doesn't love food? : Consuming APIs and utilizing Sentiment Analysis
# 
# @author [gggordon](github.com/gggordon)
# 
# @created 31.10.2018
# 
# The following notebook is an introduction tutorial presented as part of a wider course Data Warehousing and Data Mining. While the notebook does provide sample constructs and resources. The author only intends to presents the concepts and does not  indicate ownership of the resources presented. 
# 
# NB. This notebook is accompanied by a discussion
# 
# ### Topics:
#     - Restful APIs
#     - Wordclouds
#     - Correlation
#     - Sentiment Analysis

# ### Aim: To retrieve comments from a food related API (restful) and perform sentiment analysis on the data set

# In[ ]:


# import the necessary libraries

import pandas as pd # data processing
import matplotlib.pyplot as plt # plotting 
import seaborn as sns # visualizations


# ## The Food API
# 
# We will be consuming/retrieving data from a [restful](https://en.wikipedia.org/wiki/Representational_state_transfer) [api](https://en.wikipedia.org/wiki/Application_programming_interface) provided by [VegGuide.org](https://www.vegguide.org). 
# 
# You may learn more about the API and how to access it here :  https://www.vegguide.org/site/api-docs
# 
# You may test and explore the api here : https://www.vegguide.org/api-explorer/

# In[ ]:


# storing or settings to access the API in variables
api_base_url = 'https://www.vegguide.org' 
region_id = 2 # we can retrieve reviews from different regions
resource_path='/entry/{0}/reviews'.format(region_id) 
api_comments_url = api_base_url+resource_path #actual url that will be used


# In[ ]:


import requests # importing the requests library to perform http requests the api


# In[ ]:


# perform a http GET request to the URL specified earlier with custom HTTP headers
req = requests.get(api_comments_url, headers={
    'User-Agent':'SampleApi/0.01',
    'Accept':'application/json'})


# In[ ]:


#extract the data from the http response
data = req.json()


# In[ ]:


data # just viewing the data


# In[ ]:


# How many rows/records
print('We have {0} rows/records in the retrieved dataset'.format(len(data))) 


# In[ ]:


# What does one row/record look like?
data[0] 


# In[ ]:


# What keys/entries/columns are available in json row?
data[0].keys()


# In[ ]:


data[0]['body'] # a close look at the body key


# In[ ]:


data[0]['body']['text/vnd.vegguide.org-wikitext']


# In[ ]:


# This cell has an error, only kept for discussion purposes


#data_rows=[]
#for index in range(0,len(data)):
#    row = data[index]
#    data_rows.append({
#        'comment':row['body']['text/vnd.vegguide.org-wikitext'],
#        'date':row['last_modified_datetime'],
#        'user_veg_level_num':row['user']['veg_level'],
#        'user_veg_level_desc':row['user']['veg_level_description'],
#        'user_name':row['user']['name'],
#        'rating':row['rating']
#    })
#data_rows


# In[ ]:


# This cell has an error, only kept for discussion purposes


#data_rows=[]
#for index in range(0,len(data)):
#    row = data[index]
#    print(index)
#    print(row.keys())
#    data_rows.append({
#        'comment':row['body']['text/vnd.vegguide.org-wikitext'],
#        'date':row['last_modified_datetime'],
#        'user_veg_level_num':row['user']['veg_level'],
#        'user_veg_level_desc':row['user']['veg_level_description'],
#        'user_name':row['user']['name'],
#        'rating':row['rating']
#   })
#data_rows


# In[ ]:


#extracting the data as flat records to be added to a list `data_rows`
data_rows=[] # create a list to store all records
for index in range(0,len(data)): # iterate for each row in dataset
    row = data[index] #temporary variable to store row
    data_rows.append({ # extracting data from json document and creating dictionary and appending
        'comment':row['body']['text/vnd.vegguide.org-wikitext'] if 'body' in row else '',
        'date':row['last_modified_datetime'] if 'last_modified_datetime' in row else None,
        'user_veg_level_num':row['user']['veg_level'],
        'user_veg_level_desc':row['user']['veg_level_description'],
        'user_name':row['user']['name'],
        'rating':row['rating']
    })
data_rows #previewing results


# In[ ]:


data2 = pd.DataFrame(data_rows) # transform results as dataframe
data2.head() #preview dataframe


# In[ ]:


#utiliy method to draw a wordcloud from list/series
#modified
def wordcloud_draw(data, color = 'black'):
    """
       Draws a wordcloud
       
       params:
           data : list/series - set of sentences to include
        
       requires ```wordcloud``` package
    """
    from wordcloud import WordCloud, STOPWORDS
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


# In[ ]:


#visualize what everyone is saying
wordcloud_draw(data2['comment'])


# ## Sentiment Analysis
# 
# Let us perform [sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis) on the text. What is the relationship between the ratings and the sentiments derived.? 

# In[ ]:


# import library to assist with sentiment analysis
# Currently using https://github.com/cjhutto/vaderSentiment
# NB. There are implications and factors to consider, we will discuss
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[ ]:


# create a SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# ### How to use this implementation?

# In[ ]:


# let us extract ONE (1) senetence
test_sentence = data2['comment'][0]
# analyze that sentence
sentiment_result = analyzer.polarity_scores(test_sentence)
# it returns a python dictionary of values
sentiment_result


# In[ ]:


# we could analyze each row in our data set using the apply method
data2['comment'].apply(analyzer.polarity_scores)


# In[ ]:


# let us define a function to analyze one sentence and return the compound value
def get_how_positive(sentence):
    return analyzer.polarity_scores(sentence)['compound']


# In[ ]:


# testing the application of the method
data2['comment'].apply(get_how_positive)


# In[ ]:


# creating a new column in our data set to store the sentiment value
data2['sentiment'] = data2['comment'].apply(get_how_positive)
# previewing updates
data2.head(10)


# In[ ]:


# Deterimining the correlation between the sentiment values and existing ratings
print("Correlation")
data2[['rating','sentiment']].corr()


# In[ ]:


# Visualizing the correlation on a heatmap
sns.heatmap(data2[['rating','sentiment']].corr())


# ## Questions?

# In[ ]:




