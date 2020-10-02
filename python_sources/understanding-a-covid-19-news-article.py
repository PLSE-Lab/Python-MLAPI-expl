#!/usr/bin/env python
# coding: utf-8

# Covid-19 has spread all the over the world. Economic situations of countries are not well. In an attempt to understand the current situation, we are analysing a Times of India economic article.
# 
# Article link- https://timesofindia.indiatimes.com/blogs/transforming-india/episode-11-economic-impact-of-coronavirus/
# 

# In[ ]:



#Importing the essential libraries
#Beautiful Soup is a Python library for pulling data out of HTML and XML files
#The Natural Language Toolkit

import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from bs4 import BeautifulSoup
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random
from wordcloud import WordCloud
from html.parser import HTMLParser

import bs4 as bs
import urllib.request
import re


# In[ ]:


#we are using request package to make a GET request for the website, which means we're getting data from it.
#It is a "Times Of India" article on the economic impact of COVID-19

r=requests.get('https://timesofindia.indiatimes.com/blogs/transforming-india/episode-11-economic-impact-of-coronavirus/')


# In[ ]:



#Setting the correct text encoding of the HTML page
r.encoding = 'utf-8'


# In[ ]:


#Extracting the HTML from the request object
html = r.text


# In[ ]:


# Printing the first 500 characters in html
print(html[:500])


# In[ ]:



# Creating a BeautifulSoup object from the HTML
soup = BeautifulSoup(html)

# Getting the text out of the soup
text = soup.get_text()


# In[ ]:


#total length
len(text)


# In[ ]:


#We are taking this range, as this contains the actual news article.
text=text[48700:70000]


# In[ ]:


text[10000:12000]


# In[ ]:


#The text of the novel contains a lot of unwanten stuff, we need to remove them
#We will start by tokenizing the text, that is, remove everything that isn't a word (whitespace, punctuation, etc.) 
#Then then split the text into a list of words


# In[ ]:


#CLEANING THE PUNCTUATION


# In[ ]:


len(text)


# In[ ]:


import string
string.punctuation


# In[ ]:


text_nopunct=''

text_nopunct= "".join([char for char in text if char not in string.punctuation])


# In[ ]:


len(text_nopunct)


# In[ ]:


#We see a large part has been cleaned


# In[ ]:


#Creating the tokenizer
tokenizer = nltk.tokenize.RegexpTokenizer('\w+')


# In[ ]:


#Tokenizing the text
tokens = tokenizer.tokenize(text_nopunct)


# In[ ]:


len(tokens)


# In[ ]:


print(tokens[0:20])


# In[ ]:


#now we shall make everything lowercase for uniformity
#to hold the new lower case words

words = []

# Looping through the tokens and make them lower case
for word in tokens:
    words.append(word.lower())


# In[ ]:


print(words[0:500])


# In[ ]:


#Stop words are generally the most common words in a language.
#English stop words from nltk.

stopwords = nltk.corpus.stopwords.words('english')


# In[ ]:


words_new = []

#Now we need to remove the stop words from the words variable
#Appending to words_new all words that are in words but not in sw

for word in words:
    if word not in stopwords:
        words_new.append(word)


# In[ ]:


len(words_new)


# # Lemmatization

# Lemmatization is the process of grouping together the different inflected forms of a word so they can be analysed as a single item. Lemmatization is similar to stemming but it brings context to the words. So it links words with similar meaning to one word.
# Lemmatization is preferred over Stemming because lemmatization does morphological analysis of the words.

# In[ ]:


from nltk.stem import WordNetLemmatizer 
  
wn = WordNetLemmatizer() 


# In[ ]:


lem_words=[]

for word in words_new:
    word=wn.lemmatize(word)
    lem_words.append(word)
    


# In[ ]:


len(lem_words)


# In[ ]:


len(words_new)


# In[ ]:


same=0
diff=0

for i in range(0,1832):
    if(lem_words[i]==words_new[i]):
        same=same+1
    elif(lem_words[i]!=words_new[i]):
        diff=diff+1


# In[ ]:


print('Number of words Lemmatized=', diff)
print('Number of words not Lemmatized=', same)


# In[ ]:


#This shows that lemmatization helps to process the data efficiently.
#It is used in- 
# 1. Comprehensive retrieval systems like search engines.
# 2. Compact indexing


# In[ ]:


#The frequency distribution of the words
freq_dist = nltk.FreqDist(lem_words)


# In[ ]:


#Frequency Distribution Plot
plt.subplots(figsize=(20,12))
freq_dist.plot(50)


# In[ ]:


#converting into string

res=' '.join([i for i in lem_words if not i.isdigit()])


# In[ ]:


plt.subplots(figsize=(16,10))
wordcloud = WordCloud(
                          background_color='black',
                          max_words=100,
                          width=1400,
                          height=1200
                         ).generate(res)


plt.imshow(wordcloud)
plt.title('Economic News (100 words)')
plt.axis('off')
plt.show()


# In[ ]:


plt.subplots(figsize=(16,10))
wordcloud = WordCloud(
                          background_color='black',
                          max_words=200,
                          width=1400,
                          height=1200
                         ).generate(res)


plt.imshow(wordcloud)
plt.title('Economic News (200 words)')
plt.axis('off')
plt.show()


# In[ ]:





# # Text Summarzation

# In[ ]:


# Removing Square Brackets and Extra Spaces
clean_text = re.sub(r'\[[0-9]*\]', ' ', text)
clean_text = re.sub(r'\s+', ' ', clean_text)


# In[ ]:





# In[ ]:


clean_text[0:500]


# In[ ]:


#We need to tokenize the article into sentences
#Sentence tokenization

sentence_list = nltk.sent_tokenize(clean_text)


# In[ ]:


sentence_list


# In[ ]:


#Weighted Frequency of Occurrence

stopwords = nltk.corpus.stopwords.words('english')

word_frequencies = {}
for word in nltk.word_tokenize(clean_text):
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1


# In[ ]:


type(word_frequencies)


# In[ ]:


maximum_frequncy = max(word_frequencies.values())

for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)


# In[ ]:





# In[ ]:


sentence_scores = {}
for sent in sentence_list:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 30:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]


# In[ ]:


sentence_scores


# In[ ]:


import heapq
summary_sentences = heapq.nlargest(20, sentence_scores, key=sentence_scores.get)

summary = ' '.join(summary_sentences)
print(summary)


# In[ ]:





# In[ ]:


#Thank You


# In[ ]:





# In[ ]:




