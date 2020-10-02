#!/usr/bin/env python
# coding: utf-8

# Simple analysis of Wikipedia titles with the aim to practice the use of nltk library and regex in Python.
# 
# The first goal was to identify most common words that appear in Wikipedia titles.
# 
# The second goal was to write a function, that takes a given word as an argument and produces a list of most common words that accompany it in Wikipedia titles. E.g. are words that appear most frequently with "Poland"  the same that appear with a word "Polska" (Poland in Polish language), or are there differences?
# 
# Any comments and suggestions are warmly welcomed.

# In[ ]:


import pandas as pd
import re

from nltk.tokenize import RegexpTokenizer
import nltk

import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[ ]:


df = pd.read_csv('../input/titles.txt', sep='\n')


# In[ ]:


df.head()


# In[ ]:


df.columns = ['articles']


# In[ ]:


df.info()


# In[ ]:


# starting with duplicates drop to avoid unnecessary processing
df.drop_duplicates(inplace=True)


# In[ ]:


df.info()


# In[ ]:


# changing all row values to strings
df.articles = df.articles.apply(str)


# In[ ]:


# using lower on each row to remove values differing only by case
df.articles = [article.lower() for article in df.articles]


# In[ ]:


# changing data frame to a list, it will allow quick list comprehensions
articles = df.articles.tolist()
articles = list(set(articles))
len(articles)


# In[ ]:


# changing underscores to spaces
articles = [re.sub('_', ' ', article) for article in articles]


# In[ ]:


# removing special-characters-only rows
articles = [re.sub('[^A-Za-z0-9\s]+', '', article) for article in articles]


# In[ ]:


articles[:10]


# In[ ]:


# removing empty strings left
final_articles = [article for article in articles if article != '']


# In[ ]:


# removing unnecessary spaces
final_articles = [article.strip() for article in final_articles]


# In[ ]:


final_articles[:10]


# In[ ]:


len(final_articles)


# The data is now in much nicer and cleaner shape. We're down from 14846974 entries to 12675048. Now I am going to prepare the data to tokenize it.

# In[ ]:


# splitting each article so that in next steps I can create one string of all the text
all_lines = [article.split(' ') for article in final_articles]


# In[ ]:


all_lines[:10]


# In[ ]:


all_words = ' '.join(word for line in all_lines for word in line if word != '')


# In[ ]:


all_words[:1]


# In[ ]:


# tokenizing words
tokenizer = RegexpTokenizer('\w+')
tokens = tokenizer.tokenize(all_words)
print(tokens[:5])


# In[ ]:


# creating freq dist and plot
freqdist1 = nltk.FreqDist(tokens)
freqdist1.plot(25)


# Although the titles are in different languages, the top words include numerous English stopwords. It will be a good idea to remove them, as they carry no meaning.

# In[ ]:


nltk.download('stopwords')
sw = nltk.corpus.stopwords.words('english')
sw[:5]


# In[ ]:


all_no_stopwords = []
for word in tokens:
    if word not in sw:
        all_no_stopwords.append(word)

freqdist2 = nltk.FreqDist(all_no_stopwords)
freqdist2.plot(25)


# Result, most common words in Wikipedia titles:

# In[ ]:


freqdist2.most_common(25)


# In[ ]:


def contextWords(search_word, list_of_strings):
    '''
    Takes a string as an argument and checks it's presence in each string from a given list.
    Returns a frequency distribution plot of words that accompany the search word in all strings, excluding English 
    stopwords.
    '''
    search_list = [word for word in list_of_strings if re.findall(search_word, word)]
    search_words = []
    for expression in search_list:
        for w in expression.split(' '):
            if w not in nltk.corpus.stopwords.words('english'):
                search_words.append(w)
    new_search_words = ' '.join(word for word in search_words if word != '')
    new_tokens = tokenizer.tokenize(new_search_words)
    search_freqdist = nltk.FreqDist(new_tokens)
    search_freqdist.plot(20)   


# In[ ]:


contextWords('poland', final_articles)


# In[ ]:


contextWords('polska', final_articles)


# I'm finding this quite surprising. In English version we see a lot of neutral words such as: voivoidship (regional unit), national, greater and lesser (which constitute some of the voivodship names). Some words corresponding to Poland's history: kingdom, republic, infantry, war. On the other hand the Polish version is what I find really interesting, as it includes hockey twice! I wouldn't guess this is the one sport to be associated with Poland. Other than that there are: league, uprising, energy, newspaper, national, movie and socialistic and some administrative units' names.

# In[ ]:


contextWords('germany', final_articles)


# In[ ]:


contextWords('deutschland', final_articles)


# English version of articles with 'Germany' include a lot of historical connotations: nazi, east, west, infantry, army, whereas German version include none of such words. Most common words in articles' titles are then, except for stopwords: party, rally, addiction, superstar, gmbh (l.l.c.), season and a few geographical terms. This creates a significant contrast.

# In[ ]:




