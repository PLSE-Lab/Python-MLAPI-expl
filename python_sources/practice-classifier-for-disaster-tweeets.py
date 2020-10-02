#!/usr/bin/env python
# coding: utf-8

# # Practice Classifier for Disaster Tweeets

# In[ ]:


#REFERENCES
#https://tinyurl.com/r7z663r -- Twitter Sentiment Analysis usig Naive Bayes
#https://www.kaggle.com/arefoestrada/nlp-classifying-disaster-tweets
#https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert
#https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove
#https://www.kaggle.com/manjeetsingh/eda-for-disaster-tweets
#https://www.kaggle.com/arefoestrada/nlp-classifying-disaster-tweets/notebook


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from nltk import corpus
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
import re
import string



#importing dataset
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")
train = pd.read_csv("../input/nlp-getting-started/train.csv")


# ##### 1. A glimpse of the training data:

# In[ ]:


train.head()


# #### 2. A glimpse of test data:

# In[ ]:


test.head()


# ### 3. Missing Values
# In both training and test data, we can see that the location and keyword column has missing values.
# 
# 1. Total number of null values in keywords & location columns in training set

# In[ ]:


train.isnull().sum()


# 2. Total number of null values in keywords & location columns in test set

# In[ ]:


test.isnull().sum()


# Since the number of missing values are too large to just omit/drop the rows altogether. 
# Missing values in those features are filled with no_keyword and no_location respectively.

# In[ ]:


values = {'keyword': "no_keyword", 'location': "no_location"}
train.fillna(value=values)


# In[ ]:


test.fillna(value=values)


# ### Basic EDA
# 
# 1. Finding class distribution

# In[ ]:


x=train.target.value_counts()
sns.barplot(x.index,x)
plt.gca().set_ylabel('samples')


# The distribution shows that there are more non-disaster tweeets than disaster ones.

# ### Data Cleaning

# Most common word:
# 
# The most common words here are mostly stopwords. Words that have no meaning. 
# This indicates the data needs to be cleaned.

# In[ ]:


freq = pd.Series(' '.join(train['text']).split()).value_counts()[:20]
freq


# #### Ngram Analysis:
# The most common bigrams in tweets.

# In[ ]:


def get_top_tweet_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[ ]:


plt.figure(figsize=(10,5))
top_tweet_bigrams=get_top_tweet_bigrams(train['text'])[:10]
x,y=map(list,zip(*top_tweet_bigrams))
sns.barplot(x=y,y=x)


# The analysis of top Bi-gram shows that those words mostly have links/urls.

# #### Cleaning the text column:

# Removing URLS:

# In[ ]:


#removes URL from text
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

train['text']=train['text'].apply(lambda x : remove_URL(x))


# In[ ]:


#removes HTML tags from text
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

train['text']=train['text'].apply(lambda x : remove_html(x))


# In[ ]:


#removing emojis
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

train['text']=train['text'].apply(lambda x: remove_emoji(x))


# In[ ]:


#removing punctuations

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

train['text']=train['text'].apply(lambda x : remove_punct(x))


# In[ ]:


#using pyspellchecker to remove spelling errors

from spellchecker import SpellChecker

spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

train['text']=train['text'].apply(lambda x : correct_spellings(x))


# In[ ]:


# Importing train_test_split
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train['text'], train['target'])


# #### Vectorizing Dataset
# 
# Tfidf vectorizer is used to create a corpus and vectorize the train's train dataset and test dataset.

# In[ ]:


# Initializing the vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()


# In[ ]:


# Learning the vocabulary inside the datasets and transform the train's train dataset into matrix
X_train_vect = vect.fit_transform(X_train)
X_test_vect = vect.transform(X_test)


# #### Classifying Each Tweet
# 
# Multinomial Naive Bayes classifier, which is suitable for classification on text.

# In[ ]:


# Importing Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()


# In[ ]:


# Training the train dataset
model.fit(X_train_vect, y_train)


# In[ ]:


# Predicting the target (0 for non-disaster tweet, 1 for disaster tweet)
y_predict = model.predict(X_test_vect)


# In[ ]:


# Estimating the accuracy of the model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[ ]:


# Classification report
print(classification_report(y_test, y_predict))


# In[ ]:


# Confusion matrix
print(confusion_matrix(y_test, y_predict))


# In[ ]:


# Accuracy score
print(accuracy_score(y_test, y_predict))


# ### Inferencing from Test Dataset

# In[ ]:


# Extracting the tweets from the test dataset
text_test = test['text']


# In[ ]:


# Transforming the tweets into matrix
text_test_trans = vect.transform(text_test)

# Predicting the tweets
result = model.predict(text_test_trans)

# Putting the result into the submission's dataframe
sample_submission['target'] = result


# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission.to_csv('submission.csv', index = False)


# In[ ]:


import pandas as pd
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")
train = pd.read_csv("../input/nlp-getting-started/train.csv")

