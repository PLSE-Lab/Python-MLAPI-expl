#!/usr/bin/env python
# coding: utf-8

# # P5M1L5 Text Processing - Capturing Text Data
# 

# ### Steps of Natural Language Processing:
# 1. Capturing Text Data (file, web scraping etc)
# 2. Cleaning Raw Data (removing tags (of html), extracting useful text from html, removing extra white spaces,etc)
# 3. Normalization - Lowercase, removing punctuation
# 4. Tokenization - Token is a fancy word for 'words'. Generally we break text document in words.
# 5. Stop Word removal
# 6. Part of Speech Tagging
# 7. Named Entity Recognition
# 8. Stemming and Lemmatization
#     * Stemming doesn't use dictionary. Changing, Changed, Change ----- converted to chang
#     * Lammatization uses dictionary. Changing, Changed, Change ------- converted to change

# In[ ]:


# 1. Python open()

import os
with open('../input/poetry/adele.txt','r') as f:
    adele = f.read()


# In[ ]:


# 2. Web Scraping

import requests
from bs4 import BeautifulSoup


# In[ ]:


from time import sleep
sleep(5)


# In[ ]:


url = 'https://www.udacity.com/courses/all'
page = requests.get(url)


# In[ ]:


# There are two methods of parsing the html page. 1st - Regular Expressions. 2nd - BeautifulSoup
# Regular Expression:

import re


# In[ ]:


re.findall('[^.,?/''"":;!@#$%&*() ]+', page.text)


# In[ ]:


pattern = re.compile(r'<,.*?>')
# page_clean = print(pattern.sub(' ', page.text)) # Replaces patterns with space


# In[ ]:


# Beautiful Soup

soup = BeautifulSoup(page.text, 'html5lib')
print(soup.getText())


# In[ ]:


soup = BeautifulSoup(page.content, 'html.parser')
for i in soup.select('a'):
    print(i.getText())


# # 3. Normalization

# In[ ]:





# In[ ]:


#1. Lower Case

adele = adele.lower()

#2. Punctuation Removal
import string
punc = string.punctuation
adele = ''.join([char for char in adele if char not in punc]) #for loops may be inefficient sometimes


# In[ ]:


adele[:500]


# In[ ]:


#Removing newlines
#strip function does not remove whitespaces from the middle part of the text. It removes white spaces only from the 
#beginnings and end. Let's use replace to get rid of the newlines.

adele = adele.replace('\n', ' ')
adele[:500]


# In[ ]:


# using regular expression to get rid of punctuations:
with open('../input/poetry/beatles.txt') as g:
    beatles = g.read()

#Normalization
#Lower case

beatles = beatles.lower()

#Removing Punctuation
import re
beatles = re.sub('[^a-zA-Z0-9\s]','',beatles) #replacing not a-z, A-Z, 0-9 and whitespaces with ''.


# In[ ]:


beatles = beatles.replace('\n', ' ')


# # 4. Tokenization 
# 2 methods:
# 1. str.split() - but it converts Dr. to 'Dr'
# 2. nltk - it's smart enough to consider Dr. as Dr.

# In[ ]:


#1. split()

adele_words = adele.split()
adele_words[:5]


# In[ ]:


#2. nltk

from nltk.tokenize import word_tokenize
adele_nltk_word =word_tokenize(adele)
adele_nltk_word[:2]


# # 5. Stopwords Removal

# In[ ]:


from nltk.corpus import stopwords

adele_stopwords_removed = [w for w in adele_nltk_word if w not in stopwords.words('english')]
adele_stopwords_removed[:5]


# # 6. Parts of Speech Tagging

# In[ ]:


from nltk import pos_tag
adele_pos_tag = pos_tag(adele_stopwords_removed)
adele_pos_tag[:5]


# # 7. Named Entity Recognition

# In[ ]:


from nltk import ne_chunk
adele_named_entity = ne_chunk(adele_pos_tag)


# In[ ]:


print(ne_chunk(pos_tag(word_tokenize("Don't Bullshit me"))))


# In[ ]:


print(ne_chunk(pos_tag(word_tokenize("This is bullshit"))))


# # 8. Stemming and Lemmatization
# 
# 1. Stemming:
#        It does not require any context. The words are stripped off the suffixes based on the fixed rules. Sometimes, 
#        it produces words without meaninng. Two methods for English is available in nltk. PorterStemmer and
#        LancasterStemmer. Porter is older one, follows 5 rules. Lancaster is the newer one, follows 120 rules. Lancaster 
#        causes overstemming due to aggressive iterations.
# 2. Lemmatization:
#         It connects to wordnet database and produces meaningful words.

# In[ ]:


from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer

ps = PorterStemmer()
list(map(ps.stem,adele_stopwords_removed))


# In[ ]:




