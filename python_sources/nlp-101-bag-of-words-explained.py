#!/usr/bin/env python
# coding: utf-8

# # NLP 101 - Bag of Words Explained
# 
# This notebook aims to explain and perform the bag of words approach from scratch. This approach helps convert documents containing texts into representations which can be comprehended by machine learning models. Enjoy reading! :)

# ## Table of Contents
# 
# 1. [Import Packages](#1)
# 2. [Import Data](#2)
# 3. [Preprocessing Steps](#3)
# 4. [Preprocess Texts](#4)
# 5. [Bag of Words Representation](#5)
# 6. [Applying Bag-of-Words on Test Set](#6)
# 7. [Closing Remarks](#7)

# # Import Packages <a class="anchor" id="1"></a>

# In[ ]:


import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from nltk.stem.wordnet import WordNetLemmatizer


# # Import Data <a class="anchor" id="2"></a>

# In[ ]:


train_docs = [
    "He is a lazy boy. She is also lazy.",
    "Edwin is a lazy person."
]

test_docs = [
    "She is extremely lazy."
]


# # Preprocessing Steps <a class="anchor" id="3"></a>
# 
# 1. Remove markup
# 2. Obtain only alphabets
# 3. Convert all to lowercase
# 4. Tokenize (A blunt defintion would be extracting words from dcuments and each word would be an element of the list which represents a single document)
# 5. Lemmatize
# 6. Remove stopwords

# In[ ]:


lemmatizer = WordNetLemmatizer() # Instantiate lemmatizer

def proc_text(messy): #input is a single string
    first = BeautifulSoup(messy, "lxml").get_text() #gets text without tags or markup, remove html
    second = re.sub("[^a-zA-Z]"," ",first) #obtain only letters
    third = second.lower() #obtains a list of words in lower case
    fourth = word_tokenize(third) # tokenize
    fifth = [lemmatizer.lemmatize(str(x)) for x in fourth] #lemmatizing
    stops = set(stopwords.words("english")) #faster to search through a set than a list
    final = [w for w in fifth if not w in stops] #remove stop words
    return final


# In[ ]:


train = [proc_text(text) for text in train_docs] # preprocess training set
test = [proc_text(text) for text in test_docs] # preprocess test set

print("Training Documents:\n{}\n\n".format(train))
print("Test Documents:\n{}\n\n".format(test))


# # Bag of Words Representation <a class="anchor" id="4"></a>
# 
# Now that the documents are preprocessed, we shall proceed to implementing the bag of words approach. Note that each of train and test is a list of lists, where each element in each list is a token. 
# 
# We have to convert all each document into a list or a vector and when you do this to all documents, they together will form a matrix. This matrix is also called
# 
# ## Document-Term Matrix
# 
# Let $d_i$ and $w_j$ denote i-th row and j-th column of this matrix respectively. Each entry of this matrix, which can be expressed as $ ({d_i}, {w_j}) $, would be represented by a non-negative interger which is greater than equal to zero. This integer would also mean the number of times word $j$ appears in document $i$. 
# 
# To elaborate further, the numbers of rows and columns of this matrix will be determined by the number of documents and number of distinct tokens (or words) in your training set respectively. All the tokens (or words) extracted from all training documents make up what we usually call, vocabulary.
# 
# ## Procedure
# 
# Note that these steps are only performed on the training set.
# 
# The steps are as follows:
# 
# 1. Construct vocabulary-index dictionary, i.e. create a Python dictionary with distinct tokens as keys and corresponding values would be indices ranging from 0 to number of distinct tokens.
# 2. Construct a matrix of zeros, where the numbers of rows and columns correspond to the numbers of training documents and distinct tokens respectively.
# 3. Iterate through the tokens of each document and update the counts of tokens for that partcular document.

# In[ ]:


# Construct Vocabulary-Index Dictionary

vocab_idx = dict()
num_vocab = len(vocab_idx)

for doc in train:
    for token in set(doc):
        if token not in vocab_idx.keys():
            vocab_idx[token] = num_vocab
            num_vocab += 1


# In[ ]:


print("A Peek at Vocabulary-Index Dictionary\n\n")
print(vocab_idx)


# In[ ]:


# Construct Matrix of Zeroes

train_matrix = np.zeros((len(train), num_vocab)) # Document Term Matrix for training set


# In[ ]:


# Update Counts of Terms in Each Document for Training Set

for idx, doc in enumerate(train):
    for token in doc:
        train_matrix[idx, vocab_idx[token]] += 1


# In[ ]:


print("A Peek at Train Matrix\n\n")
print(train_matrix)


# Lets reiterate what the training documents were, they are: 
# 
# 1. He is a lazy boy. She is also lazy.
# 2. Edwin is a lazy person.
# 
# They have been processed and the tokens extracted were:
# 
# 1. 'lazy', 'boy', 'also', 'lazy'
# 2. 'edwin', 'lazy', 'person'
# 
# The distinct tokens obtained from training documents are:
# 
# 1. boy
# 2. also
# 3. lazy
# 4. person
# 5. edwin
# 
# and their corresponding indices are from 0 to 4 respectively.
# 
# Looking at the document-term matrix constructed based on training set, the first row corresponds to the first document and the columns represent the distinct tokens mentioned previously and the order is preserved. To elaborate further, for the first document, we can see that:
# 
# - The word 'boy' appeared 1 time.
# - The word 'also' appeared 1 time.
# - The word 'lazy' appeared 2 times.
# - The word 'person' appeared 0 times.
# - The word 'edwin' appeared 0 times.
# 
# Thus the vector representation for the first document would be (1, 1, 2, 0, 0).

# # Applying Bag-of-Words on Test Document <a class="anchor" id="6"></a>
# 
# Now, we would have to apply similar steps on the test document, but there are some differences to take note of:
# 
# - Vocabulary-index dictionary is only built using the tokens from training set as the model would be built using distinct tokens seen in the training set only. This is because there may be new tokens found in the test set, which are not seen in the training set. Since the number of features used to train the model is fixed, the building of the vocabulary-index dictionary would be restricted to just those tokens from the training set.
# 
# - As a result of the pervious pointer, the number of rows and columns in the document-term matrix for the test set would be the numbers of documents and distinct tokens in the test set respectively.
# 
# In summary, the steps are as follows:
# 
# 1. Construct a matrix of zeros for the test set where the numbers of rows and columns would be determined by the number of documents in test set and the number of distinct tokens found in training set.
# 2. Iterate through the tokens of each document and update the counts of tokens for that partcular document. 

# In[ ]:


# Construct a matrix of zeros

test_matrix = np.zeros((len(test), num_vocab))


# In[ ]:


# Update Counts of Terms in Each Document for Test Set

for idx, doc in enumerate(test):
    for token in doc:
        if token in vocab_idx.keys():
            test_matrix[idx, vocab_idx[token]] += 1


# In[ ]:


print("A Peek at Test Matrix\n\n")
print(test_matrix)


# Let's reiterate what the test document is:
# 
# - She is extremely lazy.
# 
# The test document has been processed and the tokens extracted were:
# 
# - 'extremely', 'lazy'
# 
# Note that the word 'extremely' was not found in the vocabulary of tokens extracted from the training documents, hence it will not be represented in the document-term matrix constructed for the test set. Furthermore, for the test set, we can see that:
# 
# - The word 'boy' appeared 0 time.
# - The word 'also' appeared 0 time.
# - The word 'lazy' appeared 1 time.
# - The word 'person' appeared 0 times.
# - The word 'edwin' appeared 0 times.
# 
# Thus the vector representation for the first document would be (0, 0, 1, 0, 0).

# # Closing Remarks <a class="anchor" id="7"></a>
# 
# Bag of Words approach is a frequency-based approach to convert each document into a vector which can be used to train a machine learning model. However, this approach does not capture the syntactic and semantic relationships present within documents and there exist many other more effective ways of representation. Just to name a few:
# 
# 1. Tfidf
# 2. Word Vector Averaging
# 3. Sentence Encoding
# 
# Hope this kernel has been helpful in some ways :)
