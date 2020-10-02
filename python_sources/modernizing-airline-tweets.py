#!/usr/bin/env python
# coding: utf-8

# #Introduction
# 
# Since Kaggle recently released multi-dataset support, I thought I'd play around with it a bit using two random datasets from the community.
# 
# We'll augment our tweets using definitions from Urban Dictionary. 
# 
# This doesn't have super practical applications, but it's an interesting demonstration of the capabilities

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


airline_tweets = pd.read_csv("../input/twitter-airline-sentiment/Tweets.csv")
ud_definitions = pd.read_csv("../input/urban-dictionary-terms/urban_dictionary.csv")


# In[ ]:


ud_definitions['word_lower'] = ud_definitions.word.str.lower()
word_list = ud_definitions['word_lower'].values


# In[ ]:


# Note: If this was to be actual production code this would need to be updated
# it's extremely slow and inefficient, but it's easy to write it up this way
def contains_ud_word(tweet):
    for word in tweet:
        if word in word_list:
            return True
    return False


# In[ ]:


airline_tweets['text_lower'] = airline_tweets.text.str.lower()


# In[ ]:


airline_tweets['contains_ud'] = airline_tweets.text_lower.apply(contains_ud_word)


# # Percentage Containing Urban Dictionary Words by Sentiment

# In[ ]:


segment_grouping = airline_tweets[['airline_sentiment',
                                   'contains_ud',
                                   'tweet_id']].groupby(['airline_sentiment',
                                                         'contains_ud']).count()
segment_grouping


# In[ ]:


segment_grouping.plot(kind='bar', title="Tweets By Sentiment and Language")


# In[ ]:


toplevel_grouping = airline_tweets[['airline_sentiment',
                                   'tweet_id']].groupby(['airline_sentiment']).count()

final_results = segment_grouping.div(toplevel_grouping, level='airline_sentiment') * 100
final_results


# In[ ]:


final_results.unstack().plot.bar(stacked=True)


# #Generating Some Tweets with Replaced Definitions
# 
# Let's see what happens if we take some of the original tweets and replace any words that show up in Urban Dictionary with their definitions

# In[ ]:


ud_map = ud_definitions[['word','definition']].set_index('word')
replacement_dict = ud_map.to_dict()['definition']


# In[ ]:


ud_tweets = airline_tweets[airline_tweets.contains_ud == True]
for tweet in ud_tweets.sample(10)['text']:
    print("Original Tweet:")
    print(tweet + "\n")
    
    for word in tweet:
        if word in word_list:
            new_tweet = tweet.replace(word, replacement_dict[word])
    print("Replacement Tweet:")
    print(new_tweet + "\n")


# #Findings!
# Taking a look at this, it's very clear we have an issue. Specifically, the digit "7" is in the urban dictionary dataset, that's going to mess with all our results. Let's rerun everything after removing 7

# In[ ]:


ud_definitions[ud_definitions.word == '7']


# In[ ]:


ud_definitions = ud_definitions[ud_definitions.word != '7']
ud_definitions['word_lower'] = ud_definitions.word.str.lower()
word_list = ud_definitions['word_lower'].values

airline_tweets['contains_ud'] = airline_tweets.text_lower.apply(contains_ud_word)
segment_grouping = airline_tweets[['airline_sentiment',
                                   'contains_ud',
                                   'tweet_id']].groupby(['airline_sentiment',
                                                         'contains_ud']).count()
segment_grouping


# #Oh well!

# In[ ]:




