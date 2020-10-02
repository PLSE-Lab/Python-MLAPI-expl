#!/usr/bin/env python
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[ ]:


import nltk
import pandas as pd
import numpy as np
import re

# If you would like to work with the raw text you can use 'moby_raw'
with open('../input/moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[ ]:


def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[ ]:


def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[ ]:


from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[ ]:


def answer_one():
    
    
    return example_two() / example_one()

answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[ ]:


def answer_two():
    tokens = nltk.word_tokenize(moby_raw)
    whales = [w for w in tokens if  w =='whale' or w == 'Whale'] 
    return len(whales) / example_one()

answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[85]:


def answer_three():
    tokens = nltk.word_tokenize(moby_raw)
    dist = nltk.FreqDist(tokens)

    return dist.most_common(20)

answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return a sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[98]:


def answer_four():
    tokens = nltk.word_tokenize(moby_raw)
    dist = nltk.FreqDist(tokens)
    vocab1 = list(dist.keys())
    sel_list = [t for t in vocab1 if len(t)>5 and dist[t]>150]
    return sorted(sel_list)

answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[ ]:


def answer_five():
    tokens = nltk.word_tokenize(moby_raw)
    tokens_uni = set(tokens)
    longest_word = max(tokens_uni, key=len)
    result = longest_word, len(longest_word)
    
    return result

answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[105]:


def answer_six():
    tokens = nltk.word_tokenize(moby_raw)
    dist = nltk.FreqDist(tokens)
    freq_words = [w for w in list(dist.keys()) if w.isalpha() and dist[w] > 2000]
    
    dict_I_want = { key: dist[key] for key in freq_words } # make new dict
    result = sorted(dict_I_want.items(), key=lambda kv: kv[1], reverse=True)
    # read more about *sorted* function

    return result
    
answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[107]:


def answer_seven():
    # count tokens
    tokens = len(nltk.word_tokenize(moby_raw))
    # count sentences
    sents = len(nltk.sent_tokenize(moby_raw))
    # divide # of tokens by # of sentences
    
    return tokens/sents

answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[110]:


def answer_eight():
    tokens = nltk.word_tokenize(moby_raw)
    tags = nltk.pos_tag(tokens)
    frequencies = nltk.FreqDist([tag for (word, tag) in tags])
    return frequencies.most_common(5)

answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[123]:


from nltk.corpus import words
from nltk.metrics.distance import jaccard_distance, edit_distance
from nltk.util import ngrams

correct_spellings = words.words()


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[121]:


entries=['cormulent', 'incendenece', 'validrate']


# In[119]:


def jaccard(entries, gram_number):
    """find the closet words to each entry

    Args:
     entries: collection of words to match
     gram_number: number of n-grams to use

    Returns:
     list: words with the closest jaccard distance to entries
    """
    outcomes = []
    for entry in entries:
        spellings = [s for s in correct_spellings if s.startswith(entry[0])]
        distances = ((jaccard_distance(set(ngrams(entry, gram_number)),
                                       set(ngrams(word, gram_number))), word) for word in spellings)
        closest = min(distances)
        outcomes.append(closest[1])
    return outcomes

jaccard(entries, 3)


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[122]:


jaccard(entries, 4)


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[125]:


def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    """gets the nearest words based on Levenshtein distance

    Args:
     entries (list[str]): words to find closest words to

    Returns:
     list[str]: nearest words to the entries
    """
    outcomes = []
    for entry in entries:
        distances = ((edit_distance(entry,
                                    word), word)
                     for word in correct_spellings)
        closest = min(distances)
        outcomes.append(closest[1])
    return outcomes

print(answer_eleven())


# It took much longer.
