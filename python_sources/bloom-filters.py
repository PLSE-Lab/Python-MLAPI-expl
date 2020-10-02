#!/usr/bin/env python
# coding: utf-8

# Bloom filters are memory efficent & fast ways to check if an item appears in a set. It won't return false negatives (item is in set and it says that it isn't) but does have a chance to return a false positive. You can set your tolerance for this with a specific filter using error_rate.
# 
# This kernel has an example of how to create a bloom filter for both unigrams (aka single words) and 5-grams (five words that occur congruently). 

# In[ ]:


get_ipython().system(' pip install bloom_filter')


# In[ ]:


# import bloom filters
from bloom_filter import BloomFilter
from nltk.util import ngrams

# bloom filter with default # of max elements and 
# acceptable false positive rate
bloom = BloomFilter(max_elements=1000, error_rate=0.1)

# sample text
text = '''The numpy sieve with trial division is actually a pretty fast Python
implementation. I've done some benchmarks in the past and saw around of 2-3x or so slower 
than a similar C++ implementation and less than an order of magnitude slower than C.'''
text = text.lower()

# split by word & add to filter
for i in text.split():
    bloom.add(i)

# check if word in filter
"sieve" in bloom 


# In[ ]:


# bloom filter to store our ngrams in 
bloom_ngram = BloomFilter(max_elements=1000, error_rate=0.1)

# get 5 grams from our text
tokens = [token for token in text.split(" ") if token != ""]
output = list(ngrams(tokens, 5))

# add each 5gram to our bloom filter
for i in output:
    bloom_ngram.add(" ".join(i))

# check if word in filter
print("check unigram:")
print("sieve" in bloom_ngram)

# check if ngram in filter
print("check 5gram:")
print("numpy sieve with trial division" in bloom_ngram)

