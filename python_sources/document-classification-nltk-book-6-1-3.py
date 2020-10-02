#!/usr/bin/env python
# coding: utf-8

# Document Classification
# ====
# 
# This kernel is the content from [Chapter 6.1.3 of the NLTK book](http://www.nltk.org/book/ch06.html).
# 
# In [Chapter 2](http://www.nltk.org/book/ch02.html#sec-extracting-text-from-corpora), we saw several examples of corpora where documents have been labeled with categories. Using these corpora, we can build classifiers that will automatically tag new documents with appropriate category labels. 
# 
# First, we construct a list of documents, labeled with the appropriate categories. For this example, we've chosen the Movie Reviews Corpus, which categorizes each review as positive or negative.
# 

# In[1]:


import random
import nltk
from nltk.corpus import movie_reviews
from nltk import FreqDist, NaiveBayesClassifier

# Read the documents.
documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents.
random.shuffle(documents)


# Next, we define a feature extractor for documents, so the classifier will know which aspects of the data it should pay attention to the `feature` variable in the `document_features()` function. 
# 
# For document topic identification, we can define a feature for each word, indicating whether the document contains that word. To limit the number of features that the classifier needs to process, we begin by constructing a list of the 2000 most frequent words in the overall corpus [1]. We can then define a feature extractor [2] that simply checks whether each of these words is present in a given document.

# In[2]:


all_words = FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000] # [1]: 2000 most frequent words in the overall corpus

def document_features(document):   # [2]: checks whether word is present in a given document
    document_words = set(document) # [3]: set of all words in a document
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


# **Note:** The reason that we compute the set of all words in a document in [3], rather than just checking if word in document, is that checking whether a word occurs in a set is much faster than checking whether it occurs in a list (See [Chapter 4.7](http://www.nltk.org/book/ch04.html#sec-algorithm-design)).

# Below, we illustrate the document features extracted from the `pos/cv957_8737.txt` file:

# In[3]:


print(document_features(movie_reviews.words('pos/cv957_8737.txt'))) 


# Now that we've defined our feature extractor, we can use it to train a classifier to label new movie reviews. 

# In[4]:


featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))


# 
# To check how reliable the resulting classifier is, we compute its accuracy on the test set [1]. And once again, we can use  show_most_informative_features() to find out which features the classifier found to be most informative [2].

# In[5]:


classifier.show_most_informative_features(5)


# Apparently in this corpus, a review that mentions "schumacher" is almost 7 times more likely to be negative than positive.

# In[ ]:




