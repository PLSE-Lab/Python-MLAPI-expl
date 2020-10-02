#!/usr/bin/env python
# coding: utf-8

# Data Scientists have made great progress in using machine learning to create fake images and videos of people. The latest generation of  [deepfakes](https://electronics.howstuffworks.com/future-tech/deepfake-videos-scary-good.htm) uses deep neural networks to impersonate people in a way that is very convincing. Well, it shouldn't be too hard then to impersonate someone's tweets, right?
# 
# My method here is fairly simple, thanks to the creators of [Markovify](https://github.com/jsvine/markovify). We can use a Markov Chain generator to create synthetic tweets based on an existing library of actual tweets. 

# In[ ]:


import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100
import markovify as mk

tweets = pd.read_csv('../input/tweets.csv', usecols = ['handle', 'text', 'is_retweet'])
tweets = tweets[tweets.is_retweet == False]
tweets.sample(8)


# In[ ]:


def tweet(tweeter):
    doc = tweets[tweets.handle.str.contains(tweeter)].text.tolist()
    text_model = mk.Text(doc) 
    print('\n', tweeter)
    for i in range(8):
        print(text_model.make_short_sentence(140))
        
tweet('Hillary')
tweet('Donald')


# Sometimes these sound passable (and hilarious) and other times they're a little off.  It might help by restricting the tweets to start with certain words.

# In[ ]:


def subj_tweet(tweeter, subject):
    doc = tweets[tweets.handle.str.contains(tweeter)].text.tolist()
    text_model = mk.Text(doc) 
    print('\n', tweeter)
    for i in range(8):
        print(text_model.make_sentence_with_start(subject, strict=False))

subj_tweet('Hillary', 'They')
subj_tweet('Donald', 'We')


# Sometimes this works and it helps point the fake tweets in a certain direction.  There are several things you can do to further improve the results: more cleanup of the source text, use Spacy's part of speech (POS) tagger to make better sounding sentences, and of course, get more data. The Markovify home page has some great examples of people who made credible twitter bots and the like. Have fun!
# 
# 
