#!/usr/bin/env python
# coding: utf-8

# In[17]:


#importing necessary libraries

import json
import pandas as pd
import numpy as np
import collections, re

#NLP libraries
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS

#for visualization
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# # Importing English SMS Data 

# In[21]:


with open("../input/smsCorpus_en_2015.03.09_all.json") as f:
    data = json.load(f)


# In[22]:


type(data)


# In[ ]:


listofDict = data['smsCorpus']['message']


# In[ ]:


len(listofDict)


# In[ ]:


listofDict[0]


# In[ ]:


fullData = pd.DataFrame(listofDict)


# Extracting text messages

# In[ ]:


smsData = fullData[['@id','text']]


# In[ ]:


smsData = pd.DataFrame(smsData)
smsData.head()


# # Word Count
# 
# Count of words in each text message

# In[ ]:


smsData['word_count'] = smsData['text'].apply(lambda x: len(str(x).split(" ")))
smsData[['text','word_count']].head()
smsData.head()


# average word count

# In[ ]:


def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

smsData['avg_word'] = smsData['text'].apply(lambda x: avg_word(str(x)))
smsData[['text','avg_word']].head()


# Number of stopwords

# In[ ]:


from nltk.corpus import stopwords
stop = stopwords.words('english')

smsData['stopwords'] = smsData['text'].apply(lambda x: len([x for x in str(x).split() if x in stop]))
smsData[['text','stopwords']].head()


# # Pre Processing SMS text

# # Converting text messages to lowercase

# In[ ]:


smsData['text'] = smsData['text'].apply(lambda x: " ".join(str(x).lower() for x in str(x).split()))
smsData['text'].head()


# In[ ]:


smsData['upper'] = smsData['text'].apply(lambda x: len([x for x in str(x).split() if x.isupper()]))
smsData[['text','upper']].head()


# # Removing Punctuations

# In[ ]:


smsData['text'] = smsData['text'].str.replace('[^\w\s]','')
smsData['text'].head()


# # Removing Stopwords

# In[ ]:


from nltk.corpus import stopwords
stop = stopwords.words('english')
smsData['text'] = smsData['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in stop))
smsData['text'].head()


# # Removing common words

# In[ ]:


freq = pd.Series(' '.join(smsData['text']).split()).value_counts()[:10]
freq


# In[ ]:


freq = list(freq.index)
smsData['text'] = smsData['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in freq))
smsData['text'].head()


# # Removing rare words

# In[ ]:


rare = pd.Series(' '.join(smsData['text']).split()).value_counts()[-10:]
rare


# In[ ]:


rare = list(rare.index)
smsData['text'] = smsData['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in freq))
smsData['text'].head()


# # Spell Check using TextBlob
# 
# Corrects the spelling of words with the most matched words

# In[ ]:


from textblob import TextBlob
smsData['text'][:5].apply(lambda x: str(TextBlob(x).correct()))


# # Stemming
# Strips affixes using Porter's stemming algorith to reduce inflections or variant forms 

# In[ ]:


from nltk.stem import PorterStemmer
st = PorterStemmer()
smsData['text'][:5].apply(lambda x: " ".join([st.stem(word) for word in str(x).split()]))


# # Lemmatization
# Replaces with the corrects dictionary base form of a word

# In[ ]:


from textblob import Word
smsData['text'] = smsData['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in str(x).split()]))
smsData['text'].head()


# # Extracting Bigrams

# In[ ]:


TextBlob(smsData['text'][3]).ngrams(2)


# Term Frequency

# In[ ]:


tf1 = (smsData['text'][1:2]).apply(lambda x: pd.value_counts(str(x).split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1


# Inverse Document Frequency

# In[ ]:


for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(smsData.shape[0]/(len(smsData[smsData['text'].str.contains(word)])))

tf1.sort_values(by=['idf'],ascending=False)


# # Term Frequency-Inverse Document Frequency
# 
# * Measure of availabilty of a word within a text message as well as the scarcity of the word over the entire collection of text messages 
# * More the Tf-Idf more important the word is.

# In[ ]:


tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1.sort_values(by=['idf'],ascending=False)


# # Barplot of Term Frequency Inverse Document Frequency against words 

# In[ ]:


topvacab = tf1.sort_values(by='tfidf',ascending=False)
top_vacab = topvacab.head(20)
sns.barplot(x='tfidf',y='words', data=top_vacab)


# In[ ]:


top_vacab.plot(x ='words', kind='bar') 


# # Bag of Words

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
BagOfWords = bow.fit_transform(smsData['text'])
BagOfWords


# # Sentiment Analysis
# Classifies texts based on the sentiment they represent .

# In[ ]:


smsData['sentiment'] = smsData['text'].apply(lambda x: TextBlob(str(x)).sentiment[0] )
sentiment = smsData[['text','sentiment']]

sentiment.head()


# In[ ]:


pos_texts = [ text for index, text in enumerate(smsData['text']) if smsData['sentiment'][index] > 0]
neu_texts = [ text for index, text in enumerate(smsData['text']) if smsData['sentiment'][index] == 0]
neg_texts = [ text for index, text in enumerate(smsData['text']) if smsData['sentiment'][index] < 0]

possitive_percent = len(pos_texts)*100/len(smsData['text'])
neutral_percent = len(neu_texts)*100/len(smsData['text'])
negative_percent = len(neg_texts)*100/len(smsData['text'])


# # Pie chart of Sentiment Analysis 

# In[ ]:


percent_values = [possitive_percent, neutral_percent, negative_percent]
labels = 'Possitive', 'Neutral', 'Negative'

plt.pie(percent_values, labels=labels, autopct='%3.3f')


# # Creating Word Cloud

# # Word Cloud of possitive texts
# 

# In[ ]:


k= (' '.join(pos_texts))

wordcloud = WordCloud(width = 1000, height = 500).generate(k)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# # Wordcloud of neutral texts

# In[ ]:


k= (' '.join(neu_texts))

wordcloud = WordCloud(width = 1000, height = 500).generate(k)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# # Word Cloud of Negative texts

# In[ ]:


k= (' '.join(neg_texts))

wordcloud = WordCloud(width = 1000, height = 500).generate(k)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# In[ ]:




