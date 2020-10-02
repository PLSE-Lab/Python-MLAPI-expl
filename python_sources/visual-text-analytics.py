#!/usr/bin/env python
# coding: utf-8

# In[61]:


#This is a usual set of Visualizations we can use while working with Text data.
#The idea was to build a repository for future correspondence
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS


# In[1]:


import nltk


# # Visual Text Analytics

# ## 1: WordCloud

# * Text dataset used is the US presidential inaugural addresses which are part of nltk.corpus package.
# * WordClouds help in detecting the words that occur frequently.

# In[8]:


# import the dataset

from nltk.corpus import inaugural
# extract the datataset in raw format, you can also extract it in other formats as well
text = inaugural.raw()
wordcloud = WordCloud(max_font_size=60).generate(text)
plt.figure(figsize=(16,12))
# plot wordcloud in matplotlib
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# ## 2: Lexical dispersion plot

# * This is the plot of a word vs the offset of the word in the text corpus.
# * The y-axis represents the word. 
# * Each word has a strip representing entire text in terms of offset, 
# * and a mark on the strip indicates the occurrence of the word at that offset, a strip is an x-axis. 
# * The positional information can indicate the focus of discussion in the text. 
# * So if you observe the plot below the words America, 
# * Democracy and freedom occur more often at the end of the speeches and words like duties and some words have somewhat uniform distribution in the middle. 

# In[6]:


from  nltk.book import text4 as inaugural_speeches
plt.figure(figsize=(16,5))
topics = ['citizens', 'democracy', 'freedom', 'duties', 'America','principle','people', 'Government']
inaugural_speeches.dispersion_plot(topics)


# ## 3: Frequency distribution plot:

# * This plot tries to communicate the frequency of the vocabulary in the text. 
# * Frequency distribution plot is word vs the frequency of the word. 
# * The frequency of the word can help us understand the topic of the corpus. 
# * Different genre of text can have a different set of frequent words, for example, 
# * If we have news corpus then sports news may have a different set of frequent words as compared to news related to politics, 
# * nltk has FreqDist class that helps to create a frequency distribution of the text corpus. 

# In[10]:


from nltk.corpus import brown
stop_words = set(STOPWORDS)
topics = ['government', 'news', 'religion','adventure','hobbies']
for topic in topics:
    # filter out stopwords and punctuation mark and only create array of words
    words = [word for word in brown.words(categories=topic)
            if word.lower() not in stop_words and word.isalpha() ]
    freqdist = nltk.FreqDist(words)
    # print 5 most frequent words
    print(topic,'more :', ' , '.join([ word.lower() for word, count in freqdist.most_common(5)]))
    # print 5 least frequent words
    print(topic,'less :', ' , '.join([ word.lower() for word, count in freqdist.most_common()[-5:]]))


# * It will be surprising to see that in least frequent table words belonging to a category of text corpus 
# * are more informative compared to the words found in the most frequent table 
# * which is the core idea behind TF-IDF algorithm. 
# * Most frequent words convey little information about text compared to less frequent words.

# ### The code below will plot frequency distribution for a government text, you can change the genre to see distribution for a different genre like try humor, new, etc.

# In[11]:


# get all words for government corpus
corpus_genre = 'government'
words = [word for word in brown.words(categories=corpus_genre) if word.lower() not in stop_words and word.isalpha() ]
freqdist = nltk.FreqDist(words)
plt.figure(figsize=(16,5))
freqdist.plot(50)


# ## 4: Lexical diversity dispersion plot

# * Lexical diversity lets us what is the percentage of the unique words in the text corpus 
# * For example if there are 100 words in the corpus and there are only 20 unique words then lexical diversity is 20/100=0.2. 
# * The formula for calculating lexical diversity is as below :

# In[13]:


def lexical_diversity(text):
    return round(len(set(text)) / len(text),2) #Measure of uniqueness

def get_brown_corpus_words(category, include_stop_words=False):
    '''helper method to get word array for a particular category
     of brown corpus which may/may not include the stopwords that can be toggled
     with the include_stop_words flag in the function parameter'''
    if include_stop_words:
        words = [word.lower() for word in brown.words(categories=category) if word.isalpha() ]
    else:
        words = [word.lower() for word in brown.words(categories=category)
                 if word.lower() not in stop_words and word.isalpha() ]
    return words

# calculate and print lexical diversity for each genre of the brown corpus
for genre in brown.categories():
    lex_div_with_stop = lexical_diversity(get_brown_corpus_words(genre, True))
    lex_div = lexical_diversity(get_brown_corpus_words(genre, False))
    print(genre ,lex_div , lex_div_with_stop)


# * It would also be interesting to see how to lexical diversity changes in the corpus. 
# * To visualise this we can divide a text corpus into small chunks and calculate the diversity for that chuck and plot it. 
# * Corpus can be divided by sentence or we can consider each paragraph as chunks but for sake of simplicity 
# * We can consider a batch of 1000 words as a chunk and plot its lexical diversity. 

# In[57]:


#Function to sort the words of a given corpus and category lexicographically
def Lexo_sort(corpus,category):
    words1 = sorted([wrd for wrd in list(set(corpus.words(categories=category))) if wrd.isalpha()])
    return (words1) 


# ## 5: Word length distribution plot:

# * This plot is word length on x-axis vs number of words of that length on the y-axis. 
# * This plot helps to visualise the composition of different word length in the text corpus. 

# In[58]:


cfd = nltk.ConditionalFreqDist(
           (genre, len(word))
           for genre in brown.categories()
           for word in get_brown_corpus_words(genre))

plt.figure(figsize=(16,8))
cfd.plot()


# ## 6: N-gram frequency distribution plot

# * n-grams is the continuous sequences of n words that occur very often 
# * for example for n=2 we are looking for 2 words that occur very often together 
# * like New York, Butter milk, etc. such pair of words are also called bigram, for n=3 its called trigram and so on. 
# * N-gram distribution plot tries to visualise distribution n-grams for different value of n, 
# * for this example, we consider n from 1 to 5. In the plot, x-axis has the different value of n 
# * and y-axis has the number of time n-gram sequence has occurred.

# In[60]:


from nltk.util import ngrams
plt.figure(figsize=(16,8))
for genre in brown.categories():
    sol = []
    for i in range(1,6):
        count = 0
        fdist = nltk.FreqDist(ngrams(get_brown_corpus_words(genre), i))
        sol.append(len([cnt for ng,cnt in fdist.most_common() if cnt > 1]))
    plt.plot(np.arange(1,6), sol, label=genre)
plt.legend()
plt.show()

