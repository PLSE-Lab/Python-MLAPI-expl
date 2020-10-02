#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import nltk
import string

from nltk.corpus import stopwords


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_trn = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

print('Training data shape: ', df_trn.shape)
print('Testing data shape: ', df_test.shape)


# In[ ]:


# First few rows of the training dataset
df_trn.head()


# In[ ]:


df_trn['sentiment'].unique()


# In[ ]:


# First few rows of the testing dataset
df_test.head()


# The columns denote the following:
# 
# *     The textID of a tweet
# *     The text of a tweet
# *     The selected text which determines the polarity of the tweet
# *     sentiment of the tweet
# 
# The test dataset doesn't have the selected text column which needs to be identified.

# In[ ]:


#Missing values in training set
df_trn.isnull().sum()


# In[ ]:


#Missing values in test set
df_test.isnull().sum()


# In[ ]:


df_trn[df_trn.text.isnull()]


# In[ ]:


# Dropping missing values
df_trn.dropna(axis=0, inplace=True)

print('Training data shape: ', df_trn.shape)


# Now, let's analyse and see how the Sentiment column looks like. I have only used the training dataset but the process will remain the same if we wish to to do it for the test dataset as well.
# 
# Let's look at an example of each sentiment: Positive, negative and neutral

# In[ ]:


# Positive tweet
print("Positive Tweet example :", df_trn[df_trn['sentiment']=='positive']['text'].values[0])
#negative_text
print("Negative Tweet example :", df_trn[df_trn['sentiment']=='negative']['text'].values[0])
#neutral_text
print("Neutral tweet example  :", df_trn[df_trn['sentiment']=='neutral']['text'].values[0])


# In[ ]:


# Distribution of the Sentiment Column

df_trn['sentiment'].value_counts()


# In[ ]:


# It'll be better if we could get a relative percentage instead of the count.

df_trn['sentiment'].value_counts(normalize=True)


# In[ ]:


categories = df_trn['sentiment'].value_counts(normalize=True)
plt.figure(figsize=(10,5))
sns.barplot(categories.index, categories.values, alpha=0.8)
plt.title('Distribution of Sentiment column in the training set')
plt.ylabel('Percentage', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()


# In[ ]:


# It'll be better if we could get a relative percentage instead of the count.

df_test['sentiment'].value_counts(normalize=True)


# In[ ]:


categories = df_test['sentiment'].value_counts(normalize=True)
plt.figure(figsize=(10,5))
sns.barplot(categories.index, categories.values, alpha=0.8)
plt.title('Distribution of Sentiment column in the testing set')
plt.ylabel('Percentage', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()


# Before we start with any NLP project we need to pre-process the data to get it all in a consistent format.We need to clean, tokenize and convert our data into a matrix. Let's create a function which will perform the following tasks on the text columns:
# 
# *     Make text lowercase,
# *     removes hyperlinks,
# *     remove punctuation
# *     removes numbers
# *     tokenizes
# *     removes stopwords

# In[ ]:


sentences = ' '.join(x.lower() for x in df_trn['text'].values)
print(sentences)


# In[ ]:


special_words = re.findall('[0-9a-z%s]+' % string.punctuation, sentences)
print(sorted(set(special_words)))


# In[ ]:


df_trn['text_clean'] = df_trn['text'].str.lower()

df_trn.head()


# In[ ]:


df_trn['text_clean'] = df_trn['text_clean'].apply(lambda x: re.sub('https?://\S+|www\.\S+', ' url ', x))

clean_sentences = ' '.join(x for x in df_trn['text_clean'].values)
special_words = re.findall('[0-9a-z%s]+' % string.punctuation, clean_sentences)
print(sorted(set(special_words)))


# In[ ]:


df_trn['text_clean'] = df_trn['text_clean'].apply(lambda x: re.sub('#[\w\d]+', ' hashtag ', x))

clean_sentences = ' '.join(x for x in df_trn['text_clean'].values)
special_words = re.findall('[0-9a-z%s]+' % string.punctuation, clean_sentences)
print(sorted(set(special_words)))


# In[ ]:


df_trn['text_clean'] = df_trn['text_clean'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))

clean_sentences = ' '.join(x.lower() for x in df_trn['text_clean'].values)
special_words = re.findall('[0-9a-z%s]+' % string.punctuation, clean_sentences)
print(sorted(set(special_words)))


# In[ ]:


df_trn['text_clean'] = df_trn['text_clean'].apply(lambda x: re.sub('\n', ' ', x))
df_trn['text_clean'] = df_trn['text_clean'].apply(lambda x: re.sub('\w*\d\w*', ' number ', x))
df_trn['text_clean'] = df_trn['text_clean'].apply(lambda x: re.sub('\s+', ' ', x))

df_trn.head(20)


# In[ ]:


print(stopwords.words('english'))


# In[ ]:


print(nltk.tokenize.word_tokenize(df_trn['text_clean'][5]))


# In[ ]:


tk = nltk.tokenize.word_tokenize(df_trn['text_clean'][5])
rs = []
for w in tk:
    if w not in stopwords.words('english'):
        rs.append(w)

print(df_trn['text_clean'][5])
print(' '.join(rs))


# In[ ]:


def text_preprocessing(text):
    """
    Cleaning and parsing the text.

    """
    tokenized_text = nltk.tokenize.word_tokenize(text)
    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(remove_stopwords)
    return combined_text


# In[ ]:


# Applying the cleaning function to both test and training datasets
df_trn['text_clean'] = df_trn['text_clean'].apply(str).apply(lambda x: text_preprocessing(x))


# In[ ]:


df_trn.head()


# In[ ]:


# Number of words
df_trn['word_count'] = df_trn['text'].apply(lambda x: len(str(x).split(" ")))
df_trn[['text','word_count']].head()


# In[ ]:


# Number of char
df_trn['char_count'] = df_trn['text'].str.len() ## this also includes spaces
df_trn[['text','char_count']].head()


# In[ ]:


pos = df_trn[df_trn['sentiment']=='positive']
neg = df_trn[df_trn['sentiment']=='negative']
neutral = df_trn[df_trn['sentiment']=='neutral']


# In[ ]:


categories = pos['char_count'].value_counts(normalize=True)
plt.figure(figsize=(50, 25))
sns.barplot(categories.index, categories.values, alpha=0.8)
plt.title('Positive Text Length Distribution')
plt.ylabel('Count', fontsize=20)
plt.xlabel('Char length', fontsize=20)
plt.show()


# In[ ]:


categories = neg['char_count'].value_counts(normalize=True)
plt.figure(figsize=(50, 25))
sns.barplot(categories.index, categories.values, alpha=0.8)
plt.title('Negative Text Length Distribution')
plt.ylabel('Count', fontsize=20)
plt.xlabel('Char length', fontsize=20)
plt.show()


# In[ ]:


categories = neutral['char_count'].value_counts(normalize=True)
plt.figure(figsize=(50, 25))
sns.barplot(categories.index, categories.values, alpha=0.8)
plt.title('Neutral Text Length Distribution')
plt.ylabel('Count', fontsize=20)
plt.xlabel('Char length', fontsize=20)
plt.show()


# In[ ]:


# Average word length

def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

df_trn['avg_word'] = df_trn['text'].apply(lambda x: avg_word(x))
df_trn[['text','avg_word']].head()


# In[ ]:


# Stop words present in tweet
stop = stopwords.words('english')

df_trn['stopwords'] = df_trn['text'].apply(lambda x: len([x for x in x.split() if x in stop]))
df_trn[['text','stopwords']].head()


# In[ ]:


# Hashtag present in tweet

df_trn['hastags'] = df_trn['text'].apply(lambda x: len([w for w in x.split() if w.startswith('#')]))
df_trn[['text','hastags']].head()


# In[ ]:


df_trn['numerics'] = df_trn['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
df_trn[['text','numerics']].head()


# In[ ]:




