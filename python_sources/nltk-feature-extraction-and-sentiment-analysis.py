#!/usr/bin/env python
# coding: utf-8

# Hello guys, I am just a newbie on Python & ML. While studying NLKT, I faced difficulty in understanding feature extraction, So just treid to walk through some amazing examples. Finally understood, hope it helps you as well.

# # A) Feature Extraction - 
# I picked this example from:
# https://pythonprogramming.net/words-as-features-nltk-tutorial/

# In below script, execution will start from second for-in loop, here we are iterrating for all review categories which will be ('neg', 'pos'), then in second for loop we will itterate for the fileIds (i.e. 'neg/cv000_29416.txt' for 'neg' category & 'pos/cv000_29590.txt' for 'pos' category), after that we will create a list of touples which contains list of words from specific file(negative review or positive review) and it's category. We will store this list of touples in documents and will perform shuffle operation.

# In[ ]:


import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
# documents[10]


# In below script, we are listing out all the words in movie_reviews (pos and neg), and finding the frequency of each word (in form of ({a: 734, b: 500, c: 402, d: 357, ....}) ) and at last considering only top 3000 frequently used words in **word_features**.

# In[ ]:


all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]
# word_features[0:10]


# This is the main feature extraction step, here we are again itterating our **documents** and passing each document containing review words and its category to the find_features function. In this function we are checking if the a review words are present in the complete word_features list, if yes, then we are marking them as 'true' and remaining as 'false' word_features as 'false'.
# 
# 
# For eg : 
# 
# Suppose a - z are total words we have, and are stored in **word_features**
# 
# >     word_features = [a, b, c, d, e, f, ....., z]
# 
# [a, c, f, i] are the words obtained from a positive review file.
#    
# >     documents = [([a, c, f, i], 'pos'), ... ]
#     
# After applying find_features we will get below record in our **featuresets** variable
# 
# >     featuresets = [({a:True, b:False, c:True, d:False, e:False, f:True, g:False, h:False, i:True, j:False, k:False, l:False, m:False, n:False, o:False, p:False, q:False, r:False, s:False, t:False, u:False, v:False, w:False, x:False, y:False, z:False}, 'pos'), ... ]
# 
# So eventually we will get 3000 keys (with True/False based on reviewFileWords presence) for each reviewFile. This is our final feature the needs to given to classifier for training.

# In[ ]:


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words) # will return either True or False

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
# featuresets[0]


# We can use obtained 'featuresets' to train NaiveBayesClassifier algorithm.

# **=================================================================================**

# # B) NLTK Twitter Sentiment Analysis
# In below work I am using Peter Nagy's amazing Kernel:
# https://www.kaggle.com/ngyptr/python-nltk-sentiment-analysis
# ****
# Here, I am just explaining the feature extraction done by him.

# **Sentiment Analysis:**
# the process of computationally identifying and categorizing opinions expressed in a piece of text, especially in order to determine whether the writer's attitude towards a particular topic, product, etc. is positive, negative, or neutral.
# 
#   [1]: https://github.com/nagypeterjob

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


# I decided to only do sentiment analysis on this dataset, therfore I dropped the unnecessary colunns, keeping only *sentiment* and *text*.

# In[ ]:


data = pd.read_csv('../input/Sentiment.csv')
data.head()


# In[ ]:


# Keeping only the neccessary columns
data = data[['text','sentiment']]
data.head()


# First of all, splitting the dataset into a training and a testing set. The test set is the 10% of the original dataset. For this particular analysis I dropped the neutral tweets, as my goal was to only differentiate positive and negative tweets.

# In[ ]:


# Splitting the dataset into train and test set
train, test = train_test_split(data,test_size = 0.1)
# Removing neutral sentiments
train = train[train.sentiment != "Neutral"]


# **Stop Word:** Stop Words are words which do not contain important significance to be used in Search Queries. Usually these words are filtered out from search queries because they return vast amount of unnecessary information. ( the, for, this etc. )

# In[ ]:


tweets = []
stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    # Filtering out the words with less than 4 characters     
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    
    # Here we are filtering out all the words that contains link|@|#|RT     
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    
    # filerting out all the stopwords 
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    
    # finally creating tweets list of tuples containing stopwords(list) and sentimentType 
    tweets.append((words_without_stopwords, row.sentiment))


# ## Creating the list of all unique words(from all tweets): 
# 
# As a next step I extracted the so called features with nltk lib, first by measuring a frequent distribution and by selecting the resulting keys.

# In[ ]:


# It will extract all words from every tweets and will put it into seperate list
def get_words_in_tweets(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    # extracted all the words only    
    return all

# Note that, we are not using this frequency of word occurance anywhere, So it will just return the unique word list. 
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features

all_words_in_tweets = get_words_in_tweets(tweets)

w_features = get_word_features(all_words_in_tweets)

# w_features


# Hereby I plotted the most frequently distributed words. The most words are centered around debate nights.
# 
# **NOTE : *It has nothing to do with sentiment analysis. It just depicts the use of wordcloud. That's all***

# In[ ]:


def wordcloud_draw(data, color = 'black'):
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

wordcloud_draw(w_features)


# ## Feature Extraction:
# 
# Using the NLTK NaiveBayes Classifier I classified the extracted tweet word features.

# In[ ]:


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features[f'contains({word})'] = (word in document_words)
    return features


# In below code, we are calling extract_features for every single tweet and in extract_features function, creating a key for every uniue word (from all tweets) and giving its value True for all words that are present in that specific tweet(passed in function).

# In[ ]:


training_set = nltk.classify.apply_features(extract_features,tweets)
# training_set[0]


# We are basically transforming all the tweets record in specific format, where every list item has been replace with a touple of dictionary and sentimentType. That is why length of tweets and training_set is same. check below code.

# In[ ]:


len(training_set), len(tweets)


# This dictionary(in training_set touple) contains keys(all words from all tweets) and True value for words that are present in that tweet, False for remaining. That is why every dictionary contains same number of keys as the length of total word list(w_features). You can cross verify also by checking the first 5 key in training_set[0] and first 5 items in w_features list, they are same.

# In[ ]:


len(training_set[0][0]), len(w_features)


# Amar - Now, if you have a question about this format of training_set, then I think this is the format which is required by nltk.NaiveBayesClassifier. Now below portion is not much complex.

# ## Building and Traininig Model

# In[ ]:


classifier = nltk.NaiveBayesClassifier.train(training_set)


# Finally, with not-so-intelligent metrics, I tried to measure how the classifier algorithm scored.

# In[ ]:


test_pos = test[ test['sentiment'] == 'Positive']
test_pos = test_pos['text']
test_neg = test[ test['sentiment'] == 'Negative']
test_neg = test_neg['text']


# In[ ]:


neg_cnt = 0
pos_cnt = 0
for obj in test_neg: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Negative'): 
        neg_cnt = neg_cnt + 1
for obj in test_pos: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Positive'): 
        pos_cnt = pos_cnt + 1
        
print('[Negative]: %s/%s '  % (len(test_neg),neg_cnt))        
print('[Positive]: %s/%s '  % (len(test_pos),pos_cnt))    


# ## Epilog ##
# 
# In this project I was curious how well nltk and the NaiveBayes Machine Learning algorithm performs for Sentiment Analysis. In my experience, it works rather well for negative comments. The problems arise when the tweets are ironic, sarcastic has reference or own difficult context.
# 
# Consider the following tweet:
# *"Muhaha, how sad that the Liberals couldn't destroy Trump.  Marching forward."*
# As you may already thought, the words **sad** and **destroy** highly influences the evaluation, although this tweet should be positive when observing its meaning and context. 
# 
# To improve the evalutation accuracy, we need something to take the context and references into consideration. As my project 2.0, I will try to build an LSTM network, and benchmark its results compared to this nltk Machine Learning implementation. Stay tuned. 

# In[ ]:




