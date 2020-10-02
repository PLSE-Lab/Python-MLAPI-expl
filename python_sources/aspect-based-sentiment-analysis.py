#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Greetings! This is the first public kernel. In this kernel, I  will demonstrate few steps of NLP, few plots like wordcloud, frequency plot.

# In[ ]:


#general imports
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import seaborn as sns # plotting
import matplotlib.pyplot as plt # plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import os # accessing directory structure

#NLP processing imports
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import re
import spacy

###Vader Sentiment
#To install vaderSentiment
get_ipython().system('pip install vaderSentiment ')
from vaderSentiment import vaderSentiment
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

####Lemmatization
from nltk.stem import WordNetLemmatizer
# Lemmatize with POS Tag
from nltk.corpus import wordnet


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv("/kaggle/input/twcs.csv")


# In[ ]:


data.shape


# In[ ]:


data = data.loc[:10000]


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


pd.set_option('display.max_colwidth', -1)


# In[ ]:


data.head(10)


# In[ ]:


nRow, nCol = data.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


#DataTypes
data.dtypes


# In[ ]:


data["text"] = data["text"].astype(str)


# # TEXT CLEANING

# ### Remove urls,@mention, https

# In[ ]:


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+|@[^\s]+')
    return url_pattern.sub(r'', text)


# In[ ]:


data["textclean"] = data["text"].apply(lambda text: remove_urls(text))


# In[ ]:


data.head()


# # Find out the top 100 words which are getting used in the text of the data

# In[ ]:


top_N = 100 #top 100 words

#convert list of list into text
a = data['textclean'].str.lower().str.cat(sep=' ')

# removes punctuation,numbers and returns list of words
b = re.sub('[^A-Za-z]+', ' ', a)


# In[ ]:


#remove all the stopwords from the text
stop_words = list(get_stop_words('en'))         
nltk_words = list(stopwords.words('english'))   
stop_words.extend(nltk_words)


# In[ ]:


word_tokens = word_tokenize(b) # Tokenization
filtered_sentence = [w for w in word_tokens if not w in stop_words]
filtered_sentence = []
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)


# In[ ]:


# Remove characters which have length less than 2  
without_single_chr = [word for word in filtered_sentence if len(word) > 2]

# Remove numbers
cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]


# **#### *Lemmatization is the process of converting a word to its base form. The difference between stemming and lemmatization is, lemmatization considers the context and converts the word to its meaningful base form, whereas stemming just removes the last few characters, often leading to incorrect meanings and spelling errors.*

# # Lemmatization

# #### I am using Wordnet Lemmatizer with appropriate POS tag. 
# #### Function to map word with its POS tag

# In[ ]:


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# 1. Init Lemmatizer
lemmatizer = WordNetLemmatizer()


# In[ ]:


lemmatized_output = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in cleaned_data_title]
lemmatized_output = [word for word in lemmatized_output if not word.isnumeric()]


# # Frequency distribution

# In[ ]:


word_dist = nltk.FreqDist(lemmatized_output)
top100_words = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['Word', 'Frequency'])


# In[ ]:


plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.barplot(x="Frequency",y="Word", data=top100_words.head(10))


# # WordCloud

# In[ ]:


def wc(data,bgcolor,title):
    plt.figure(figsize = (80,80))
    wc = WordCloud(background_color = bgcolor, max_words = 100,  max_font_size = 50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')


# In[ ]:


wc(lemmatized_output,'black','Common Words' )


# # VADER + TEXTBLOB Sentiment Analysis

# In[ ]:


sent_analyser = SentimentIntensityAnalyzer()
def sentiment(text):
    return (sent_analyser.polarity_scores(text)["compound"])


# In[ ]:


data["Polarity"] = data["textclean"].apply(sentiment)


# In[ ]:


data.head()


# In[ ]:


data.dtypes


# In[ ]:


def senti(data):
    if data['Polarity'] >= 0.05:
        val = "Positive"
    elif data['Polarity'] <= -0.05:
        val = "Negative"
    else:
        val = "Neutral"
    return val


# In[ ]:


data['Sentiment'] = data.apply(senti, axis=1)


# In[ ]:


plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment", data=data, 
                  palette=dict(Neutral="blue", Positive="Green", Negative="Red"))


# # ASPECT MINING/ OPINION MINING

# In[ ]:


#import spacy
nlp = spacy.load("en_core_web_sm")


# In[ ]:


def pos(text):
    doc = nlp(text)
    # You want list of Verb tokens 
    aspects = [token.text for token in doc if token.pos_ == "NOUN"]
    return aspects


# In[ ]:


data["Aspects"] = data["textclean"].apply(pos)


# In[ ]:


data.head()


# There is scope of improvement. I will learn and update it accordingly

# **P.S This is my first Kaggle Kernel and I am fairly new to python programming as well, hence my non usage of list comprehensions and functions might be evident. I highly encourage everyone to fork my code and add your own twists to increase the accuracy of both aspect extractions and sentiment analysis.**
