#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis

# ## 1. Exploring the `movie_reviews` corpus

# In[ ]:


from nltk.corpus import movie_reviews     # These are movie reviews already separated as positive and negative.
movie_reviews.readme().replace('\n', ' ').replace('\t', '').replace('``', '"').replace("''", '"').replace('`', "'")


# In[ ]:


movie_reviews.fileids()


# In[ ]:


len(movie_reviews.fileids())


# In[ ]:


movie_reviews.raw("neg/cv000_29416.txt").replace("\n", "").replace("'", '"').replace('"', "'") # Note here I found a trick to get rid of \' in text. However it only works if there were no " used.


# ## 2. Building and testing the classifier

# In[ ]:


from nltk.corpus import stopwords

stops = stopwords.words('english')
stops.extend('.,[,],(,),;,/,-,\',?,",:,<,>,n\'t,|,#,\'s,\",\'re,\'ve,\'ll,\'d,\'re'.split(','))
stops.extend(',')
stops


# In[ ]:


from nltk.classify import NaiveBayesClassifier
import nltk.classify.util # Utility functions and classes for classifiers. Contains functions such as accuracy(classifier, gold)

# Given a word, returns a dict {word: True}. This will be our feature in the classifier. 
def word_feats(words):
    return dict([(word, True) for word in words if word not in stops and word.isalpha()])

pos_ids = movie_reviews.fileids('pos')
neg_ids = movie_reviews.fileids('neg')

len(pos_ids) + len(neg_ids) 


# In[ ]:


# We take the positive/negative words, create the feature for such words, and store it in a positive/negative features list.
pos_feats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in pos_ids]
neg_feats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in neg_ids]

pos_feats


# In[ ]:


# 3/4 of the features will be used for training.
pos_len_train = int(len(pos_feats) * 3 / 4)
neg_len_train = int(len(neg_feats) * 3 / 4)

pos_len_train


# In[ ]:


train_feats = neg_feats[:neg_len_train] + pos_feats[:pos_len_train]
test_feats = neg_feats[neg_len_train:] + pos_feats[pos_len_train:]

# Training a NaiveBayesClassifier with our training feature words.
classifier = NaiveBayesClassifier.train(train_feats)

print('Accuracy: ', nltk.classify.util.accuracy(classifier, test_feats))


# In[ ]:


# We can see which words fit best in each class.
classifier.show_most_informative_features()


# ## 3. Classifying new data

# In[ ]:


from nltk import word_tokenize, pos_tag

sentence = "I feel so miserable, it makes me amazing"
tokens = [word for word in word_tokenize(sentence) if word not in stops]
tokens


# In[ ]:


feats = word_feats(word for word in tokens)
feats


# In[ ]:


classifier.classify(feats)


# In[ ]:


sentence2 = "You are a pathetic fool, a terrible excuse for a human being."
tokens2 = [word for word in word_tokenize(sentence2) if word not in stops]
tokens2


# In[ ]:


pos_tags2 = [pos for pos in pos_tag(tokens2) if pos[1] == 'JJ']
pos_tags2


# In[ ]:


feats2 = word_feats([word for (word,_) in pos_tags2])
feats2


# In[ ]:


classifier.classify(feats2)


# ## 4. Incorporating bigram features

# In order to improve the classifier, bigram features can be examined using `nltk.util.ngrams`. This is because, for instance, 'not funny' is very different from 'funny'.
