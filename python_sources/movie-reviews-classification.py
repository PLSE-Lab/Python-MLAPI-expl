#!/usr/bin/env python
# coding: utf-8

# using **Natural Language Processing with nltk** and **movie_reviews** dataset to classify the reviews positive or negative

# ## Import Libraries**

# In[ ]:


import nltk


# load dataset

# In[ ]:


from nltk.corpus import movie_reviews


# In[ ]:


movie_reviews.fileids()[:5]


# Check number of movie_reviews

# In[ ]:


len(movie_reviews.fileids())


# Print first 5

# In[ ]:


movie_reviews.fileids()[:5]


# last 5

# In[ ]:


movie_reviews.fileids()[-5:]


# fileids can filter the available files based on their category, which is the name of the subfolders they are located in. Therefore we can have lists of positive and negative reviews separately.

# In[ ]:


neg_fileids=movie_reviews.fileids('neg')
pos_fileids=movie_reviews.fileids('pos')


# In[ ]:


len(neg_fileids),len(pos_fileids)


# In[ ]:


print(movie_reviews.raw(fileids=pos_fileids[0]))


# In[ ]:


movie_reviews.words(fileids=pos_fileids[0])


# In[ ]:


import string
string.punctuation


# Using the Python string.punctuation list and the English stopwords we can build better features by filtering out those words that would not help in the classification:

# In[ ]:


useless_words=nltk.corpus.stopwords.words("english")+list(string.punctuation)
useless_words[:10]


# In[ ]:


def build_bag_of_words_features_filtered(words):
    return {
        word:1 for word in words if not word in useless_words}


# **Train a Classifier for Sentiment Analysis**
# Using our build_bag_of_words_features function we can build separately the negative and positive features. Basically for each of the 1000 negative and for the 1000 positive review, we create one dictionary of the words and we associate the label "neg" and "pos" to it.

# In[ ]:


negative_features = [
    (build_bag_of_words_features_filtered(movie_reviews.words(fileids=[f])), 'neg') \
    for f in neg_fileids
]


# In[ ]:


negative_features[:1]


# In[ ]:


positive_features = [
    (build_bag_of_words_features_filtered(movie_reviews.words(fileids=[f])), 'pos') \
    for f in pos_fileids
]


# In[ ]:


positive_features[:1]


# In[ ]:


from nltk.classify import NaiveBayesClassifier


# One of the simplest supervised machine learning classifiers is the Naive Bayes Classifier, it can be trained on 80% of the data to learn what words are generally associated with positive or with negative reviews.

# In[ ]:


split = 800


# In[ ]:


sentiment_classifier = NaiveBayesClassifier.train(positive_features[:split]+negative_features[:split])


#  Check the accuracy on the training set
# 

# In[ ]:


nltk.classify.util.accuracy(sentiment_classifier, positive_features[:split]+negative_features[:split])*100


# Check the accuracy of test dataset

# In[ ]:


nltk.classify.util.accuracy(sentiment_classifier, positive_features[split:]+negative_features[split:])*100

