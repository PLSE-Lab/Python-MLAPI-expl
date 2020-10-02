#!/usr/bin/env python
# coding: utf-8

# # Detecting Text Language by Counting Stop Words

# 
# *Stop words* are words which are filtered out before processing because they are mostly grammatical as opposed to semantic in nature e.g. search engines remove words like 'want'.

# ## 1. Tokenizing

# In[ ]:


text = "Yo man, it's time for you to shut yo' mouth! I ain't even messin' dawg."


# In[ ]:


import sys

try:
    from nltk.tokenize import wordpunct_tokenize # RE-based tokenizer which splits text on whitespace and punctuation (except for underscore)
except ImportError:
    print('[!] You need to install nltk (http://nltk.org/index.html)')


# In[ ]:


test_tokens = wordpunct_tokenize(text)
test_tokens


# There are other tokenizers e.g. `RegexpTokenizer` where you can enter your own regexp, `WhitespaceTokenizer` (similar to Python's `string.split()`) and `BlanklineTokenizer`.

# ## 2. Exploring NLTK's stop words corpus

# NLTK comes with a corpus of stop words in various languages.

# In[ ]:


from nltk.corpus import stopwords
stopwords.readme().replace('\n', ' ') # Since this is raw text, we need to replace \n's with spaces for it to be readable.


# In[ ]:


stopwords.fileids() # Most corpora consist of a set of files, each containing a piece of text. A list of identifiers for these files is accessed via fileids().


# Corpus readers provide a variety of methods to read data from the corpus:

# In[ ]:


stopwords.raw('greek')


# In[ ]:


stopwords.raw('greek').replace('\n', ' ') # Better


# In[ ]:


stopwords.words('english')[:10]


# We can also use `.sents()` which returns sentences. However, in our particular case, this will cause an error:

# In[ ]:


stopwords.sents('greek')


# The erro is because the `stopwords` corpus reader is of type `WordListCorpusReader` so there are no sentences.
# It's the same for `.paras()`.

# In[ ]:


len(stopwords.words(['english', 'greek'])) # There is a total of 444 Greek and English stop words


# ## 3. The classification

# We loop through the list of stop words in all languages and check how many stop words our test text contains in each language. The text is then classified to be in the language in which it has the most stop words.

# In[ ]:


language_ratios = {}

test_words = [word.lower() for word in test_tokens] # lowercase all tokens
test_words_set = set(test_words)

for language in stopwords.fileids():
    stopwords_set = set(stopwords.words(language)) # For some languages eg. Russian, it would be a wise idea to tokenize the stop words by punctuation too.
    common_elements = test_words_set.intersection(stopwords_set)
    language_ratios[language] = len(common_elements) # language "score"
    
language_ratios


# In[ ]:


most_rated_language = max(language_ratios, key=language_ratios.get) # The key parameter to the max() function is a function that computes a key. In our case, we already have a key so we set key to languages_ratios.get which actually returns the key.
most_rated_language


# In[ ]:


test_words_set.intersection(set(stopwords.words(most_rated_language))) # We can see which English stop words were found.

