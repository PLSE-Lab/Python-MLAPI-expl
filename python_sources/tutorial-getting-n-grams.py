#!/usr/bin/env python
# coding: utf-8

# ### The order words are used in is important
# 
# The order that words are used in text is not random. In English, for example, you can say "the red apple" but not "apple red the". The relationships between words in text is very complex. So complex, in fact, that there's an entire field of linguistics devoted to it: syntax. (You can learn more about syntax [here](https://pdfs.semanticscholar.org/ba33/a9656f43a6b7420a180bdeccd5be98fc05eb.pdf), if you're interested.)
# 
# However, there is a relatively simply way of capturing some of these relationships between words, originally proposed by Warren Weaver. You can capture quite a bit of information by just looking at which words tend to show up next to each other more often. 
# 
# 
# ### What are ngrams? 
# 
# The general idea is that you can look at each pair (or triple, set of four, etc.) of words that occur next to each other. In a sufficently-large corpus, you're likely to see "the red" and "red apple" several times, but less likely to see "apple red" and "red the". This is useful to know if, for example, you're trying to figure out what someone is more likely to say to help decide between possible output for an automatic speech recognition system.
# 
# These co-occuring words are known as "n-grams", where "n" is a number saying how long a string of words you considered. (Unigrams are single words, bigrams are two words, trigrams are three words, 4-grams are four words, 5-grams are five words, etc.)
# 
# 
# > **Learning goal:** In this tutorial, you'll learn how to find all the n-grams in a corpus & count how often each is used.

# In[ ]:


# load in all the modules we're going to need
import nltk, re, string, collections
from nltk.util import ngrams # function for making ngrams

# this corpus is pretty big, so let's look at just one of the files in it
with open("../input/spanish_corpus/spanishText_15000_20000", "r", encoding='latin-1') as file:
    text = file.read()

# check to make sure the file read in alright; let's print out the first 1000 characters
text[0:1000]


# Looking at the text, you can see that there's some extra xml markup that we don't really want to analyze (the suff in <pointy brackets>). Let's get rid of that.

# In[ ]:


# let's do some preprocessing. We don't care about the XML notation, new lines 
# or punctuation marks other than periods. (We'll consider the end of the sentence
# a "word") We also don't want to consider capitalization. 

# get rid of all the XML markup
text = re.sub('<.*>','',text)

# get rid of the "ENDOFARTICLE." text
text = re.sub('ENDOFARTICLE.','',text)

# get rid of punctuation (except periods!)
punctuationNoPeriod = "[" + re.sub("\.","",string.punctuation) + "]"
text = re.sub(punctuationNoPeriod, "", text)

# make sure it looks ok
text[0:1000]


# Ok, now onto making the n-grams! The first thing we want to do is "tokenize", or break the  text into indvidual words (you can find more information on tokenization [here](https://www.kaggle.com/rtatman/tokenization-tutorial)).

# In[ ]:


# first get individual words
tokenized = text.split()

# and get a list of all the bi-grams
esBigrams = ngrams(tokenized, 2)

# If you like, you can uncomment the next like to take a look at 
# the first ten to make sure they look ok. Please note that doing so 
# will consume the generator & will break the next block of code, so you'll
# need to re-comment it and run this block again to get it to work.
# list(esBigrams)[:10]


# Yay, we've got our n-grams! Just a list of all the bigrams in a text isn't that useful, though. Let's collapse it a little bit & get a count of how frequently we see each bigram in our corpus. 

# In[ ]:


# get the frequency of each bigram in our corpus
esBigramFreq = collections.Counter(esBigrams)

# what are the ten most popular ngrams in this Spanish corpus?
esBigramFreq.most_common(10)


# I'm not a fluent Spanish speaker, but those do look like some reasonable very frequent two-word phrases in Spanish.
# 
# And that's all there is to it! Here are some exercises to help you try it out yourself.
# 
# 
# ### Your turn!
# 
# 1. Find the most frequent n-grams in another file in this corpus. You can find the other files by entering "../input/spanish_corpus/" in a code chunk and then hitting the Tab key. All of the files will be listed in a drop-down menu. Are the most frequent bigrams the same as they were in this file?
# 2. Instead of bigrams (two word phrases), can you find trigrams (three words)?
# 3. Find the most frequent ngrams in another corpus. You can find some [here](https://www.kaggle.com/datasets?sortBy=relevance&group=featured&search=corpus) to start you out.
