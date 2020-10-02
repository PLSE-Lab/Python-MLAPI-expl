#!/usr/bin/env python
# coding: utf-8

# # NLP 101

# In this notebook I show basic concepts appearing in NLP.  
# I also demonstrate how to get basic information about analyzed text using NLTK library.  
# After reading this you will be familiar with terms like:  
# * corpora
# * tokenization
# * stop words
# * lemmatization
# * stemming
# * bag of words
# * TFIDF
# 
# You also will know how to create basic features from text using NLTK and scikit-learn.

# ## What is NLP?

# NLP stands for Natural Language Processing. It is a part of computer science and artificial intelligence. NLP helps using machine learning algorithms  where the input data is text or speach. NLP is used in areas like text summarization, speach recognition, spam detection, named entity recognition, chatboots etc.
# 
# **Corpora (or text corpus)** is simply large set of text prepared in structured way. It is used for statistical analysis and checking hypothesis. Text corpus can contain texts from one or mulitple languages.

# ## What is NLTK?

# NLTK is Python library for working with human language data. It's currently the leading library in this field. NLTK offers interfaces to many corpora and lexical resources. Library contains also suite for typical text processing tasks like steaming, tokenization, parsing, tagging and much more.
# 
# Here is online book covering introduction to NLP based on NLTK library:  
# http://www.nltk.org/book/

# More info:  
# https://www.nltk.org/  
# https://github.com/nltk/nltk/wiki
# 

# ## Let's play

# For introduction of basic consepts of NLP we will use 'My little Pony' synopsis dataset.  
# We will start from prepring the environment.

# In[ ]:


import numpy as np
import pandas as pd
import nltk
import os
print(os.listdir("../input"))


# In[ ]:


dialog = pd.read_csv('../input/clean_dialog.csv')
dialog.head()


# As example for explaining basic terms we will use Narrator introduction from the beginning of first episode.

# In[ ]:


text = dialog['dialog'].loc[0]
print(text)


# ### Sentence tokenization

# **Sentence tokenization** is an action for dividing string containing writting language for sentences. Every language has own punctuation marks which are responsible marking end of sentences. For this problem we will use *sent_tokenize* function from nltk. After that we gather list of sentences

# In[ ]:


sentences = nltk.sent_tokenize(text)
print(sentences)


# In[ ]:


len(sentences)
# Narrator introduction was split into 7 sentences


# ### Word tokenization

# **Word tokenization** (also word segmentation) is very similar to sentence tokenization, but this time the aim of our activity is to divide written language to words.

# In[ ]:


words_list = []
for s in sentences:
    words = nltk.word_tokenize(s)
    words_list.append(words)


# In[ ]:


print(words_list)


# ### Word stemming

# Purpose of **stemming** is reducing the inflectional forms of words and reducing it to base form. For example: reading -> read, dogs -> dog.
# 
# For example we use the most popular lexical database Wordnet which easily accessible in NLTK. Please note that if you work different language you must select apropriate corpus. NLTK allows using own corpus which can be find in th net.

# In[ ]:


from nltk.stem import PorterStemmer
from nltk.corpus import wordnet

def stemm(word):
    stemmer = PorterStemmer()
    print('Word: ', word, ' ... Stemmer: ', stemmer.stem(word))
    
stemm('dogs')
stemm('reading')


# In[ ]:


for w in words_list[2]:
    stemm(w)


# ### Word lemmatization

# Word lemmatization is activity similar to word stemming, buth this time the purpose is to get meaningful base word form so they can be analysed as a single item, identified by the word's lemma or dictionary form.  
# Unlike stemming, lemmatisation depends on correctly identifying the intended part of speech and meaning of a word in a sentence or even in whole document.

# In[ ]:


from nltk.stem import WordNetLemmatizer

def lemmat(word, pos):
    lemmatizer = WordNetLemmatizer()
    print("Word is: :", word, "   ...   Lemmatizer:", lemmatizer.lemmatize(word, pos))

lemmat('caring', wordnet.VERB)


# In[ ]:


for w in words_list[2]:
    lemmat(w, wordnet.VERB)


# ### Stop words

# **Stop words** are words which are normally dropped before or after processing text. They are usually the most common words like *the*, *and*, *a*, etc. NLTK has own lists of stop words. There is also possibility to add own list of stop words, depending of your documents context.
# 
# Let's look at list of English stop words:

# In[ ]:


from nltk.corpus import stopwords
print(stopwords.words("english"))


# Now we apply this list to our texts and remove stop words:

# In[ ]:


stop_words = set(stopwords.words("english"))

words_list_clean = []
for words in words_list:
    words_clean = [w for w in words if not w in stop_words]
    words_list_clean.append(words_clean)


# In[ ]:


print('Sentence before removing stop words:')
print(words_list[0])
print()
print('Sentence after removing stop words:')
print(words_list_clean[0])


# ### Bag of words

# Machine learning algorithms doesn't work with plain text and need features creation. One of the simplest and most popular technique for doing this is bag-of-words.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

# Here we build our model, only words from whole list of lists
words_for_bag = []
for sen in words_list_clean:
    for wrd in sen:
        words_for_bag.append(wrd)
        
print(len(words_for_bag))        

# Preparing vocabulary dictionary based on sklearn 
count_vectorizer = CountVectorizer()

# Creatin bag-of-words model
bag_of_words = count_vectorizer.fit_transform(words_for_bag)

# bag-of-words model as a pandas df
feature_names = count_vectorizer.get_feature_names()

pd.DataFrame(bag_of_words.toarray(), columns = feature_names)


# ### TF-IDF
# **TF-IDF** stands for term frequency - inverse document frequency. It's a value describing how important is word in analyzed document comparing to collection of documents. The higher value means more rare word. When we caclulate TF-IDF we start from removing punctuation and lower casing the words. The next step is to prepare two parts: 
# * TF - term frequency - simply number of times the term appears in the document divide by number of words in the document
# * IDF - inverse document frequency - ln(number of documents / number documents the term appears in)  
# 
# Finally: TFIDF = TF * IDF

# Let look at the example. For calculating TF-IDF we will use TfidfVectorizer from scikit-learn. It contain preprocessing phase: removing stop words and lowercasing.

# In[ ]:


# We will calculate TF-IDF based on Narrator saying from first episode
print(sentences)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Let's create vectorizer
vectorizer = TfidfVectorizer(stop_words = 'english')
X = vectorizer.fit_transform(sentences)

# First vector from first document
first_vector_tfidfvectorizer=X[0]
 
# Preparinf pandas dataframe
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=vectorizer.get_feature_names(), columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)


# ### Sources and more reading:  
# NLP basics:  
# https://towardsdatascience.com/introduction-to-natural-language-processing-for-text-df845750fb63  
# 
# Great list of links covering most NLP areas:  
# https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/91432#latest-539075  
# 
# More about different approaches to lemmatization:  
# https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
# 
# Scikit-learn TFIDF vectorizer:  
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
