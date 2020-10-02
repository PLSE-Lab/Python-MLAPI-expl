#!/usr/bin/env python
# coding: utf-8

# <h1>End-to-End NLP Process with Sentiment Analysis</h1>

# In this notebook we are going to go through on how to perform text classification using logistic regression and several text encoding techniques such as bag of words and tf-idf. Our task will be to classify text to determine it's sentiment class. Our dataset contains the movie review data with labeled sentiment class of 0,1,2,3 and 4 where 0 is negative, 1 somehow negative, 2 neutral, 3 somehow positive and 4 positive.<br><br>
# We will start with Exploratory Data Analysis then perform machine learning modeling.

# <div align='center'><h1>Import required libraries</h1>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import ngrams
import string,re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import warnings, os


# In[ ]:


plt.figure(figsize=(16,7))
plt.style.use('ggplot')
warnings.filterwarnings('ignore')


# Load data

# In[ ]:


# Locate the data directories
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train=pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip',sep='\t')
test=pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv.zip',sep='\t')


# <div align='center'><H1>Part 1 Exploratory Data Analysis</H1></div>

# In[ ]:


train.shape, test.shape


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


train.head()


# Sentiment Description

# In[ ]:


train['sentiment_class'] = train['Sentiment'].map({0:'negative',1:'somewhat negative',2:'neutral',3:'somewhat positive',4:'positive'})
train.head()


# Remove punctuations

# In[ ]:


def remove_punctuation(text):
    return "".join([t for t in text if t not in string.punctuation])


# In[ ]:


train['Phrase']=train['Phrase'].apply(lambda x:remove_punctuation(x))
train.head()


# Remove words with less than 2 characters

# In[ ]:


def words_with_more_than_three_chars(text):
    return " ".join([t for t in text.split() if len(t)>3])


# In[ ]:


train['Phrase']=train['Phrase'].apply(lambda x:words_with_more_than_three_chars(x))
train.head()


# Remove stopwords

# In[ ]:


stop_words=stopwords.words('english')
train['Phrase']=train['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
train.head()


# check sentiment categories

# In[ ]:


train.groupby('Sentiment')['Sentiment'].count()


# Visualize the target variables

# In[ ]:


train.groupby('sentiment_class')['sentiment_class'].count().plot(kind='bar',title='Target class',figsize=(16,7),grid=True)


# Get percentages of each class

# In[ ]:


((train.groupby('sentiment_class')['sentiment_class'].count()/train.shape[0])*100).plot(kind='pie',figsize=(7,7),title='% Target class', autopct='%1.0f%%')


# Adding Phrase length

# In[ ]:


train['PhraseLength']=train['Phrase'].apply(lambda x: len(x))


# In[ ]:


train.sort_values(by='PhraseLength', ascending=False).head()


# Distribution of phrase length on each class

# In[ ]:


plt.figure(figsize=(16,7))
bins=np.linspace(0,200,50)
plt.hist(train[train['sentiment_class']=='negative']['PhraseLength'],bins=bins,density=True,label='negative')
plt.hist(train[train['sentiment_class']=='somewhat negative']['PhraseLength'],bins=bins,density=True,label='somewhat negative')
plt.hist(train[train['sentiment_class']=='neutral']['PhraseLength'],bins=bins,density=True,label='neutral')
plt.hist(train[train['sentiment_class']=='somewhat positive']['PhraseLength'],bins=bins,density=True,label='somewhat positive')
plt.hist(train[train['sentiment_class']=='positive']['PhraseLength'],bins=bins,density=True,label='positive')
plt.xlabel('Phrase length')
plt.legend()
plt.show()


# Common words with word cloud

# In[ ]:


# Install wordcoud library
# !pip install wordcloud


# In[ ]:


from wordcloud import WordCloud, STOPWORDS 
stopwords = set(STOPWORDS) 


# In[ ]:


word_cloud_common_words=[]  
for index, row in train.iterrows(): 
    word_cloud_common_words.append((row['Phrase'])) 
word_cloud_common_words

wordcloud = WordCloud(width = 1600, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 5).generate(''.join(word_cloud_common_words)) 
  
# plot the WordCloud image                        
plt.figure(figsize = (16, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# Word Frequency

# In[ ]:


text_list=[]  
for index, row in train.iterrows(): 
    text_list.append((row['Phrase'])) 
text_list

total_words=''.join(text_list)
total_words=word_tokenize(total_words)


# In[ ]:


freq_words=FreqDist(total_words)
word_frequency=FreqDist(freq_words)


# In[ ]:


# 10 common words
print(word_frequency.most_common(10))


# In[ ]:


# visualize 
pd.DataFrame(word_frequency,index=[0]).T.sort_values(by=[0],ascending=False).head(20).plot(kind='bar',figsize=(16,6),grid=True)


# Common words used for negative sentiment

# In[ ]:


neg_text_list=[]  
for index, row in train[train['Sentiment']==0].iterrows(): 
    neg_text_list.append((row['Phrase'])) 
neg_text_list

neg_total_words=' '.join(neg_text_list)
neg_total_words=word_tokenize(neg_total_words)

neg_freq_words=FreqDist(neg_total_words)
neg_word_frequency=FreqDist(neg_freq_words)


# In[ ]:


# visualize 
pd.DataFrame(neg_word_frequency,index=[0]).T.sort_values(by=[0],ascending=False).head(20).plot(kind='bar',figsize=(16,6),grid=True)


# Common words used for positive sentiment

# In[ ]:


pos_text_list=[]  
for index, row in train[train['Sentiment']==4].iterrows(): 
    pos_text_list.append((row['Phrase'])) 
pos_text_list

pos_total_words=' '.join(pos_text_list)
pos_total_words=word_tokenize(pos_total_words)

pos_freq_words=FreqDist(pos_total_words)
pos_word_frequency=FreqDist(pos_freq_words)


# In[ ]:


# visualize 
pd.DataFrame(pos_word_frequency,index=[0]).T.sort_values(by=[0],ascending=False).head(20).plot(kind='bar',figsize=(16,6),grid=True)


# Common bigram words used for positive sentiment

# In[ ]:


text="Tom and Jerry love mickey. But mickey dont love Tom and Jerry. What a love mickey is getting from these two friends"
bigram_frequency = FreqDist(ngrams(word_tokenize(text),3))
bigram_frequency.most_common()[0:5]


# In[ ]:


text_list=[]  
for index, row in train.iterrows(): 
    text_list.append((row['Phrase'])) 
text_list

total_words=' '.join(text_list)
total_words=word_tokenize(total_words)

freq_words=FreqDist(total_words)
word_frequency=FreqDist(ngrams(freq_words,2))
word_frequency.most_common()[0:5]


# In[ ]:


# visualize 
pd.DataFrame(word_frequency,index=[0]).T.sort_values(by=[0],ascending=False).head(20).plot(kind='bar',figsize=(16,6),grid=True)


# <div align='center'><H1>Part 2 Machine Learning Modeling</H1></div>

# Prepare Training data

# Create Bag of words with CountVectorizer

# In[ ]:


train['tokenized_words']=train['Phrase'].apply(lambda x:word_tokenize(x))
train.head()


# In[ ]:


count_vectorizer=CountVectorizer()
phrase_dtm=count_vectorizer.fit_transform(train['Phrase'])


# In[ ]:


phrase_dtm.shape


# Split data into training and validation sets (70:30) ratio

# In[ ]:


X_train,X_val,y_train,y_val=train_test_split(phrase_dtm,train['Sentiment'],test_size=0.3, random_state=38)
X_train.shape,y_train.shape,X_val.shape,y_val.shape


# Train Logistic Regression model

# In[ ]:


model=LogisticRegression()


# In[ ]:


model.fit(X_train,y_train)


# Measure model performance

# In[ ]:


accuracy_score(model.predict(X_val),y_val)*100


# Free up memory for tf-idf

# In[ ]:


del X_train
del X_val
del y_train
del y_val


# Preparing data with tf-idf

# In[ ]:


tfidf=TfidfVectorizer()
tfidf_dtm=tfidf.fit_transform(train['Phrase'])


# In[ ]:


X_train,X_val,y_train,y_val=train_test_split(tfidf_dtm,train['Sentiment'],test_size=0.3, random_state=38)
X_train.shape,y_train.shape,X_val.shape,y_val.shape


# In[ ]:


tfidf_model=LogisticRegression()


# In[ ]:


tfidf_model.fit(X_train,y_train)


# In[ ]:


accuracy_score(tfidf_model.predict(X_val),y_val)*100


# Predict on test data

# In[ ]:


print(tfidf_model.predict(X_val)[0:10])


# new data prediction function

# In[ ]:


def predict_new_text(text):
    tfidf_text=tfidf.transform([text])
    return tfidf_model.predict(tfidf_text)


# In[ ]:


predict_new_text("The movie is bad and sucks!")


# Prepare Test Data

# In[ ]:


test['Phrase']=test['Phrase'].apply(lambda x:remove_punctuation(x))
test['Phrase']=test['Phrase'].apply(lambda x:words_with_more_than_three_chars(x))
test['Phrase']=test['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
test_dtm=tfidf.transform(test['Phrase'])


# In[ ]:


# Predict with test data
test['Sentiment']=tfidf_model.predict(test_dtm)
test.set_index=test['PhraseId']
test.head()


# In[ ]:


# save results to csv file
# test.to_csv('Submission.csv',columns=['PhraseId','Sentiment'],index=False)

