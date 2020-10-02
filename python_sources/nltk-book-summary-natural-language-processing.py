#!/usr/bin/env python
# coding: utf-8

# # NLTK book exploration
# I'm beginner at programming and at Python (graduated this year as a lawyer). However, I'm not working as a lawyer, I create websites and sell adverts there. I have a lot text processing and thats why I got highly interested in this challange! But I'm not able to complete any ML task yet and I want to study NLTK + help others to do it.
# 
# So I decided to take the [book](http://www.nltk.org/book/), which Rachael reccomended in her [kernel](https://www.kaggle.com/rtatman/beginner-s-tutorial-python) and take out all the most important staff in one helpfull kernel. **This will by nltk notebook** =) 
# 
# Today I did the first chapter and I hope to finish the book untill the end of the challange. However, right now I can not quite understand how do I use data from challange? I imported .csv but I can not apply NLTK to it. I read about data processing, but still can not do it with that particular dataset. 
# 
# For now I will use text from the book, after I figure out how to process it, I will correct all the data.
# 
# So first chapter is about:
# 1.  **Counting words and punctuation**
# 1. **Plotting word's frequency**
# 1. **Basic text understanding**

# In[ ]:


#importing all neccesary libraries
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.book import *
texts = pd.read_csv("../input/train.csv")


# # Counting words and punctuation

# Its all pretty simple here.
# * **len** - counts all the words and punctuation marks.
# * **set** - all  words and punctiation marks without repition
# 
# We can combine them to determine how rich is authors text by deviding number of unique words by total number of words (shown below).
# 
# We can also apply that to count how often author uses one word in relation to others.
# 
# *Functions for that that are ready for use are below.*
# 
# * **FreqDist** - shows how many times particular word appear.
# * **most_common()** - shows words that get repeated the most

# In[ ]:


# getting lenght of the text in words and punctuation symbols
print('Words + punctuation:', len(text4))

# getting all the distinc words in text
set(text4)

# we can also combine them and see how many distinct words are exactly there 
distinct_words = len(set(text4))
print('Distinc words and punctuation symbols:', distinct_words)

# based on that we can determine how 'Rich' is the text
rich = len(set(text4)) / len(text4)
print('My book is rich ratio is', rich)

# what part of the text takes specific word in percents?
text4.count("I")
word_percent = 100 * text4.count('I') / len(text4)
print('Authors talks about himselft', word_percent, 'percent of time!' )

# This will help to get the most repetetive words and their count
fdist = FreqDist(text4)
print(fdist.most_common(5))

### HELPFUL FUNCTIONS TO DO THE SAME FASTER
def lexical_diversity(text):
    return len(set(text)) / len(text) [2]

def percentage(count, total):
    return 100 * count / total


# # Word frequency plotting

# Visualization is always important =) 
# * **fdist.plot** - plots the most repetetive words and punctuation marks by their count
# * **dispersion_plot** - shows where does text have particular word in the begging or in the end.. And how often authors uses it.

# In[ ]:


# Now while those words do not make much sense, we should determine which ones are just common words for every text
plt.figure(figsize=(15 , 8))
fdist.plot(50, cumulative=True)

# This plot basically shows word position in the text. We can see here that the word "please" was used only once at the end of the text.
plt.figure(figsize=(14 , 8))
text4.dispersion_plot(["love", "hate", "duty", "tax", "please"])


# # How can we understand what text is about

# As I understand there different way to get the meaning of the text. Frequent words do not tell much of the story. We  can also look at  words that do not repeat - **hapaxes**. That will be a little better, but still would not make too much sense. We can also subset list by getting out longest words with some frequence. That also gives some more information.
# 
# But In order to succeed we need to combine both methods, that will give much clearer picture of the text. We need to get the most repeated **bigrams** - two consecutive words in a text.

# In[ ]:


# getting all words that do not repeat itself
hapaxes = fdist.hapaxes()
print('Unique words:')
print(hapaxes[:8])
# getting combined
combined = sorted(w for w in set(text4) if len(w) > 7 and fdist[w] > 7)
print('--------------------------------------------')
print('Counting words that are longer than 7 symbols and are repeated at least 7 times:')
print(combined[:10])
#lets get the collocation
print('--------------------------------------------')
print('Two consecutive words (bigrams) that get repeated most of the times:')
text4.collocations()


# * **concordance** - shows context of concrete word in the text. 
# * **similar** - shows what other words appear in a similar contexts. With that we can understand how particular author feels about certain word. From the example below we can see that author uses 'love' in the same context with 'whales', 'sea', 'ships', which can tell us what author likes the most.
# * **common_contexts** - same as concordance, but shows mathcing contexts of several words.

# In[ ]:


# Thats how we can find any particular word
concordance = text4.concordance("please")
print(concordance)
print('--------------------------------------------')
text1.similar("love")
print('and')
text2.similar('love')
print('--------------------------------------------')
# Now. Thats the list of words which were used in there same context as 'think'! That is actually very interesting.
text1.common_contexts(["love", "sea"])


# Now we can more o less tell smth about what text is about.

# # Asessing Text Corpora and Lexical Resources
# * **corpora** - larg body of linguistic data

# In[ ]:


from nltk.corpus import gutenberg
gutenberg.fileids()
#picking up texts from the Projest Gutenberg electronic Text archive
nltk.corpus.gutenberg.fileids()
# After we get a list of names we can put in argument just the text we need. Notice .words
hamlet = gutenberg.words('shakespeare-hamlet.txt')


# There is a convinient way to compare texts by: average word length, average sentence length and the lexical diversity score.
# * **raw**() -  gives us the contents of the file without any linguistic processing

# In[ ]:


for fileid in gutenberg.fileids():
    # contents of the file without any linguistic processing
     num_chars = len(gutenberg.raw(fileid)) 
    # average word lenght (!it counts spaces, so you have to assume number is 1 less)
     num_words = len(gutenberg.words(fileid))
    # average sentence length
     num_sents = len(gutenberg.sents(fileid))
     num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
     print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)


#  # Plotting word appearance over time
#  Inaugural Address Corpus is  a collection of 55 texts with **time demension**.
#  
#  Lets plot how frequent was the use of the word "America' through out the time.

# In[ ]:


from nltk.corpus import inaugural
# extracting the first four characters, using fileid[:4] to get the year 
[fileid[:4] for fileid in inaugural.fileids()]
cfd = nltk.ConditionalFreqDist(
           (target, fileid[:4])
           for fileid in inaugural.fileids()
           for w in inaugural.words(fileid)
           for target in ['america', 'me']
           if w.lower().startswith(target)) 
cfd.plot()


# # Main Corpus Functionality
# * **fileids**()	the files of the corpus
# * ** fileids([categories])**	the files of the corpus corresponding to these categories
# * **categories()**	the categories of the corpus
# * **categories([fileids])	**the categories of the corpus corresponding to these files
# * **raw()	**the raw content of the corpus
# * **raw(fileids=[f1,f2,f3])**	the raw content of the specified files
# * **raw(categories=[c1,c2])**	the raw content of the specified categories
# * **words()**	the words of the whole corpus
# * **words(fileids=[f1,f2,f3])**	the words of the specified fileids
# * **words(categories=[c1,c2])**	the words of the specified categories
# * **sents()**	the sentences of the whole corpus
# * **sents(fileids=[f1,f2,f3])**	the sentences of the specified fileids
# * ** sents(categories=[c1,c2])**	the sentences of the specified categories
# * **abspath(fileid)**	the location of the given file on disk
# * **encoding(fileid)**	the encoding of the file (if known)
# * ** open(fileid)	**open a stream for reading the given corpus file
# * **root**	if the path to the root of locally installed corpus
# * **readme()**	the contents of the README file of the corpus

# # Loading your own Corpus (kaggle one for example)

# In[ ]:


from nltk.corpus import PlaintextCorpusReader
corpus_root = '../input/'
wordlists = PlaintextCorpusReader(corpus_root, '.*') 
wordlists.words('test.csv')


# # Conditional Frequency Distributions
# * **ConditionalFreqDist()** takes a list of pairs.

# In[ ]:


from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist(
           (genre, word)
           for genre in brown.categories()
           for word in brown.words(categories=genre))

